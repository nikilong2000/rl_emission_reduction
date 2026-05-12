"""
Single-seed phase-2 fine-tune with explicit (w_emission, w_soc_squared).
Designed for SLURM array dispatch (10 seeds) via submit_phase2_seeds.sh.

Each seed continues from the phase-1 best-seed checkpoint with the
Optuna-selected reward weights, varies only training/env seed.

Usage:
    python run_phase2_seed.py --algorithm ppo --seed 0 \
        --phase2_config 02_rl_control/logs/ppo/phase2_optuna/best_params_phase2.json
    python run_phase2_seed.py --algorithm sac --seed 3 \
        --w_emission 0.7 --w_soc_squared 120
"""

import os
import sys
import json
import argparse
import datetime
import time
import warnings
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

current_dir = os.path.dirname(os.path.abspath(__file__))
rl_control_dir = os.path.dirname(current_dir)
models_dir = os.path.join(rl_control_dir, "models")
sys.path.insert(0, rl_control_dir)
sys.path.insert(0, models_dir)
sys.path.insert(0, current_dir)
from env import EmissionControlEnv
from env_thermal import EmissionControlEnvThermal
from models.eval import evaluate_model
from utils.config_utils import load_config
from utils.checkpoint_utils import VecNormalizeCheckpointCallback
from hpo_search_spaces import apply_config_override
from select_best_seed import find_best_seed
from tune_phase2_hpo import _build_continue_kwargs

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

ALGO_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}
ALGO_ON_POLICY = {"ppo": True, "sac": False, "td3": False}
N_ENVS = 40


def _resolve_weights(args):
    """Read weights from --phase2_config JSON or fall back to CLI flags."""
    if args.phase2_config:
        with open(args.phase2_config) as f:
            cfg = json.load(f)
        we = cfg["w_emission"]
        wsq = cfg["w_soc_squared"]
        print(f"Loaded weights from {args.phase2_config}: we={we} wsq={wsq}")
        return float(we), float(wsq)
    if args.w_emission is None or args.w_soc_squared is None:
        sys.exit(
            "Either --phase2_config or both --w_emission and --w_soc_squared required."
        )
    return float(args.w_emission), float(args.w_soc_squared)


def main(args):
    algo_key = args.algorithm.lower()
    if algo_key not in ALGO_CLASSES:
        print(f"Unknown algorithm '{algo_key}'.")
        sys.exit(1)
    AlgoClass = ALGO_CLASSES[algo_key]
    is_on_policy = ALGO_ON_POLICY[algo_key]

    try:
        from utils.platform_utils import configure_environment, configure_tf_devices

        configure_environment()
        configure_tf_devices()
    except ImportError:
        pass

    set_random_seed(args.seed)
    print(f"Random seed: {args.seed}")

    w_emission, w_soc_squared = _resolve_weights(args)

    # Config with phase-1 HPO + injected phase-2 weights
    config = load_config(current_dir=models_dir, algo_key=algo_key)
    hpo_config_path = args.hpo_config or os.path.join(
        rl_control_dir,
        "logs_cluster_phase1",
        "logs",
        algo_key,
        "optuna",
        "best_params.json",
    )
    with open(hpo_config_path) as f:
        overrides = json.load(f)
    apply_config_override(config, overrides)
    config.W_EMISSION = w_emission
    config.W_SOC_SQUARED = w_soc_squared
    config.W_SOC = 0.0
    config.W_FUEL = 0.0
    print(f"Applied HPO overrides from {hpo_config_path}")
    print(f"Reward weights: we={w_emission} wsq={w_soc_squared}")

    # Phase-1 best-seed ckpt
    seeds_dir = args.seeds_dir or os.path.join(
        rl_control_dir,
        "logs_cluster_phase1",
        "logs",
        algo_key,
        "optuna",
        "seeds",
    )
    best, _ = find_best_seed(seeds_dir, algo_key)
    ckpt_path = best["model_zip"]
    vec_norm_path = os.path.join(best["seed_dir"], "vec_normalize.pkl")
    print(f"Continuing from phase-1 seed {best['seed']}: {ckpt_path}")

    # Output dir
    output_root = args.output_dir or os.path.join(
        rl_control_dir, "logs", algo_key, "phase2_seeds"
    )
    seed_dir = os.path.join(output_root, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)
    print(f"Output dir: {seed_dir}")

    # Env
    env_cls = EmissionControlEnvThermal if args.use_thermal else EmissionControlEnv

    def make_env(rank):
        def _init():
            e = env_cls(config_module=config, random_target=True)
            e.reset(seed=args.seed * 1000 + rank)
            return Monitor(e, os.path.join(seed_dir, str(rank)))

        return _init

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = True
        env.norm_reward = is_on_policy
        print(f"Loaded VecNormalize stats from {vec_norm_path}")
    else:
        print(f"Warning: vec_normalize.pkl missing at {vec_norm_path}; starting fresh.")
        env = VecNormalize(
            env, norm_obs=is_on_policy, norm_reward=is_on_policy, clip_obs=10.0
        )

    model_kwargs = _build_continue_kwargs(
        algo_key, config, env, seed_dir, args.agent_device
    )
    print(f"Loading {algo_key.upper()} ckpt from {ckpt_path}")
    model = AlgoClass.load(ckpt_path, env=env, **model_kwargs)

    # Callbacks
    checkpoint_freq = 100_000
    name_prefix = f"{algo_key}_phase2_seed_{args.seed}"
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(seed_dir, "checkpoints"),
        name_prefix=name_prefix,
    )
    vec_norm_cb = VecNormalizeCheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(seed_dir, "checkpoints"),
        name_prefix=name_prefix,
        vec_normalize=env,
    )

    print(
        "=" * 40,
        f"Seed {args.seed}: fine-tune {algo_key.upper()} ({args.total_timesteps:,} steps)",
    )
    t0 = time.perf_counter()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, vec_norm_cb],
    )
    duration = time.perf_counter() - t0
    duration_hms = str(datetime.timedelta(seconds=int(duration)))
    print(f"Seed {args.seed}: training finished in {duration_hms}")

    final_name = f"{algo_key}_phase2_seed_{args.seed}_final"
    model_path = os.path.join(seed_dir, final_name)
    model.save(model_path)
    env.save(os.path.join(seed_dir, "vec_normalize.pkl"))

    train_config = {
        "algorithm": algo_key.upper(),
        "phase": "2_seed_validation",
        "seed": args.seed,
        "continued_from": ckpt_path,
        "phase1_best_seed": best["seed"],
        "phase2_config_path": args.phase2_config,
        "total_timesteps": args.total_timesteps,
        "n_envs": N_ENVS,
        "w_speed": config.W_SPEED,
        "w_emission": config.W_EMISSION,
        "w_fuel": config.W_FUEL,
        "w_brake": config.W_BRAKE,
        "w_soc": config.W_SOC,
        "w_soc_squared": config.W_SOC_SQUARED,
        "w_flicker": config.W_FLICKER,
        "training_duration_seconds": round(duration, 3),
        "training_duration_hms": duration_hms,
    }
    with open(os.path.join(seed_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    env.close()

    print("=" * 40, f"Seed {args.seed}: running full evaluation")
    evaluate_model(
        model_path,
        eval_log_dir=seed_dir,
        train_config=train_config,
        algorithm=algo_key,
        use_thermal=args.use_thermal,
        random_target=True,
    )
    print(f"Seed {args.seed}: complete -> {seed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--phase2_config",
        default=None,
        help="Path to Optuna best_params_phase2.json. Alternative to --w_emission/--w_soc_squared.",
    )
    parser.add_argument("--w_emission", type=float, default=None)
    parser.add_argument("--w_soc_squared", type=float, default=None)
    parser.add_argument(
        "--hpo_config",
        default=None,
        help="Phase-1 best_params.json. Default: logs_cluster_phase1/logs/<algo>/optuna/best_params.json.",
    )
    parser.add_argument(
        "--seeds_dir",
        default=None,
        help="Phase-1 seeds dir. Default: logs_cluster_phase1/logs/<algo>/optuna/seeds.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--total_timesteps", type=int, default=4_000_000)
    parser.add_argument("--agent_device", default="auto")
    parser.add_argument("--use_thermal", action="store_true", default=False)
    main(parser.parse_args())
