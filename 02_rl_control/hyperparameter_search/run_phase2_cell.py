"""
Run one phase-2 fine-tune cell: continue from a phase-1 best-seed checkpoint
with new reward weights (W_EMISSION, W_SOC_SQUARED).

Designed for SLURM array dispatch via submit_phase2_sweep.sh — array index =
cell_id (0..8). Each cell maps to one (W_EMISSION, W_SOC_SQUARED) combo on
a coarse 3x3 grid.

Usage:
    python run_phase2_cell.py --algorithm ppo --cell_id 4
    python run_phase2_cell.py --algorithm sac --cell_id 0 --total_timesteps 50000
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

current_dir = os.path.dirname(os.path.abspath(__file__))
rl_control_dir = os.path.dirname(current_dir)
models_dir = os.path.join(rl_control_dir, "models")
sys.path.insert(0, rl_control_dir)
sys.path.insert(0, models_dir)
sys.path.insert(0, current_dir)
from env import EmissionControlEnv
from misc.env_thermal import EmissionControlEnvThermal
from models.eval import evaluate_model
from utils.config_utils import load_config
from utils.checkpoint_utils import VecNormalizeCheckpointCallback
from hpo_search_spaces import apply_config_override
from select_best_seed import find_best_seed

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

ALGO_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}
ALGO_ON_POLICY = {"ppo": True, "sac": False, "td3": False}
N_ENVS = 40

W_EMISSION_GRID = [0.25, 0.5, 1.0]
W_SOC_SQUARED_GRID = [50.0, 150.0, 400.0]


def cell_id_to_weights(cell_id: int):
    n = len(W_SOC_SQUARED_GRID)
    if cell_id < 0 or cell_id >= len(W_EMISSION_GRID) * n:
        raise ValueError(
            f"cell_id {cell_id} out of range " f"[0, {len(W_EMISSION_GRID) * n - 1}]"
        )
    i_e, i_s = divmod(cell_id, n)
    return W_EMISSION_GRID[i_e], W_SOC_SQUARED_GRID[i_s]


def _build_continue_kwargs(algo_key, config, env, log_dir, device):
    """Match models/train.py continue-from path: skip policy_kwargs and
    learning_starts so the loaded checkpoint's architecture and replay-buffer
    state are preserved."""
    kwargs = {
        "verbose": 1,
        "tensorboard_log": log_dir,
        "device": device,
    }
    if algo_key == "ppo":
        kwargs.update(
            {
                "learning_rate": config.LEARNING_RATE,
                "n_steps": config.N_STEPS,
                "batch_size": config.BATCH_SIZE,
                "n_epochs": config.N_EPOCHS,
                "gamma": config.GAMMA,
                "gae_lambda": config.GAE_LAMBDA,
                "clip_range": config.CLIP_RANGE,
            }
        )
    elif algo_key == "sac":
        kwargs.update(
            {
                "learning_rate": config.LEARNING_RATE,
                "buffer_size": config.BUFFER_SIZE,
                "batch_size": config.BATCH_SIZE,
                "tau": config.TAU,
                "gamma": config.GAMMA,
                "train_freq": config.TRAIN_FREQ,
                "gradient_steps": config.GRADIENT_STEPS,
                "ent_coef": config.ENT_COEF,
                "target_entropy": config.TARGET_ENTROPY,
                "use_sde": config.USE_SDE,
                "sde_sample_freq": config.SDE_SAMPLE_FREQ,
            }
        )
    elif algo_key == "td3":
        n_actions = env.action_space.shape[-1]
        kwargs.update(
            {
                "learning_rate": config.LEARNING_RATE,
                "buffer_size": config.BUFFER_SIZE,
                "batch_size": config.BATCH_SIZE,
                "tau": config.TAU,
                "gamma": config.GAMMA,
                "train_freq": config.TRAIN_FREQ,
                "gradient_steps": config.GRADIENT_STEPS,
                "policy_delay": config.POLICY_DELAY,
                "target_policy_noise": config.TARGET_POLICY_NOISE,
                "target_noise_clip": config.TARGET_NOISE_CLIP,
                "action_noise": NormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=config.ACTION_NOISE_SIGMA * np.ones(n_actions),
                ),
            }
        )
    return kwargs


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

    # Resolve grid cell -> reward weights
    w_emission, w_soc_squared = cell_id_to_weights(args.cell_id)
    print(
        f"Cell {args.cell_id}: W_EMISSION={w_emission}, W_SOC_SQUARED={w_soc_squared}"
    )

    # Load base config + apply phase-1 best HPO params (same alg hyperparams)
    config = load_config(current_dir=models_dir, algo_key=algo_key)
    if args.hpo_config:
        with open(args.hpo_config) as f:
            overrides = json.load(f)
        apply_config_override(config, overrides)
        print(f"Applied HPO overrides from {args.hpo_config}")

    # Inject phase-2 reward weights
    config.W_EMISSION = w_emission
    config.W_SOC_SQUARED = w_soc_squared
    config.W_SOC = 0.0
    config.W_FUEL = 0.0

    # Resolve phase-1 best-seed checkpoint
    seeds_dir = args.seeds_dir or os.path.join(
        rl_control_dir,
        "logs_cluster",
        "logs",
        algo_key,
        "optuna",
        "seeds",
    )
    best, _ = find_best_seed(seeds_dir, algo_key)
    ckpt_path = best["model_zip"]
    vec_norm_path = os.path.join(best["seed_dir"], "vec_normalize.pkl")
    print(f"Continuing from seed {best['seed']}: {ckpt_path}")

    # Output dir: logs/<algo>/phase2/cell_<id>_we<we>_wsq<wsq>/
    output_root = args.output_dir or os.path.join(
        rl_control_dir, "logs", algo_key, "phase2"
    )
    cell_name = f"cell_{args.cell_id:02d}_we{w_emission:g}_wsq{w_soc_squared:g}"
    cell_dir = os.path.join(output_root, cell_name)
    os.makedirs(cell_dir, exist_ok=True)
    print(f"Output dir: {cell_dir}")

    # Environment
    env_cls = EmissionControlEnvThermal if args.use_thermal else EmissionControlEnv

    def make_env(rank):
        def _init():
            e = env_cls(config_module=config, random_target=True)
            e.reset(seed=args.cell_id * 1000 + rank)
            return Monitor(e, os.path.join(cell_dir, str(rank)))

        return _init

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # Load VecNormalize stats from phase-1 seed; norm_reward stale but reset by
    # SB3's running-stats accumulator within first ~10k steps for PPO.
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

    # Build kwargs and load model
    model_kwargs = _build_continue_kwargs(
        algo_key, config, env, cell_dir, args.agent_device
    )
    print(f"Loading {algo_key.upper()} model from {ckpt_path}...")
    model = AlgoClass.load(ckpt_path, env=env, **model_kwargs)

    # Callbacks
    checkpoint_freq = 100_000
    name_prefix = f"{algo_key}_phase2_cell{args.cell_id:02d}"
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(cell_dir, "checkpoints"),
        name_prefix=name_prefix,
    )
    vec_norm_cb = VecNormalizeCheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(cell_dir, "checkpoints"),
        name_prefix=name_prefix,
        vec_normalize=env,
    )

    print(
        "=" * 40,
        f"Cell {args.cell_id}: fine-tune {algo_key.upper()} ({args.total_timesteps:,} steps)",
    )
    t0 = time.perf_counter()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, vec_norm_cb],
    )
    duration = time.perf_counter() - t0
    duration_hms = str(datetime.timedelta(seconds=int(duration)))
    print(f"Cell {args.cell_id}: training finished in {duration_hms}")

    # Save final model and vec_normalize
    final_name = f"{algo_key}_phase2_cell{args.cell_id:02d}_final"
    model_path = os.path.join(cell_dir, final_name)
    model.save(model_path)
    env.save(os.path.join(cell_dir, "vec_normalize.pkl"))

    train_config = {
        "algorithm": algo_key.upper(),
        "phase": 2,
        "cell_id": args.cell_id,
        "continued_from": ckpt_path,
        "phase1_best_seed": best["seed"],
        "phase1_baseline": {
            "rmse_speed_kmph": best["rmse_speed_kmph"],
            "total_nox_g": best["total_nox_g"],
            "delta_soc": best["delta_soc"],
        },
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
    with open(os.path.join(cell_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    env.close()

    # Full evaluation
    print("=" * 40, f"Cell {args.cell_id}: running full evaluation")
    evaluate_model(
        model_path,
        eval_log_dir=cell_dir,
        train_config=train_config,
        algorithm=algo_key,
        use_thermal=args.use_thermal,
        random_target=True,
    )
    print(f"Cell {args.cell_id}: complete -> {cell_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument(
        "--cell_id",
        type=int,
        required=True,
        help="Index into the 3x3 grid (0..8). See cell_id_to_weights().",
    )
    parser.add_argument(
        "--hpo_config",
        default=None,
        help="Path to phase-1 best_params.json. Default: logs_cluster/logs/<algo>/optuna/best_params.json.",
    )
    parser.add_argument(
        "--seeds_dir",
        default=None,
        help="Phase-1 seeds dir. Default: logs_cluster/logs/<algo>/optuna/seeds.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Parent output dir. Default: logs/<algo>/phase2.",
    )
    parser.add_argument("--total_timesteps", type=int, default=4_000_000)
    parser.add_argument("--agent_device", default="auto")
    parser.add_argument("--use_thermal", action="store_true", default=False)
    args = parser.parse_args()

    if args.hpo_config is None:
        args.hpo_config = os.path.join(
            rl_control_dir,
            "logs_cluster",
            "logs",
            args.algorithm,
            "optuna",
            "best_params.json",
        )
    main(args)
