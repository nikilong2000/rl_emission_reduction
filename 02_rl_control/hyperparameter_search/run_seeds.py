"""
Train a single seed with the best HPO config, then run full evaluation.
Intended to be called from a SLURM array job (submit_seeds.sh) or manually.
Usage:
    python run_seeds.py --algorithm ppo --config best_params.json --seed 0
    python run_seeds.py --algorithm sac --config best_params.json --seed 3 \\
           --output_dir logs/sac/optuna/seeds
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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
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

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALGO_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}
ALGO_ON_POLICY = {"ppo": True, "sac": False, "td3": False}
N_ENVS = 20


# ---------------------------------------------------------------------------
# Model kwargs builder (mirrors models/train.py logic)
# ---------------------------------------------------------------------------
def _build_model_kwargs(algo_key, config, env, log_dir, device):
    kwargs = {
        "verbose": 1,
        "tensorboard_log": log_dir,
        "device": device,
    }
    if hasattr(config, "POLICY_KWARGS"):
        kwargs["policy_kwargs"] = config.POLICY_KWARGS
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
                "learning_starts": config.LEARNING_STARTS,
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
                "learning_starts": config.LEARNING_STARTS,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    algo_key = args.algorithm.lower()
    if algo_key not in ALGO_CLASSES:
        print(f"Unknown algorithm '{algo_key}'. Choose from: {list(ALGO_CLASSES)}")
        sys.exit(1)
    # TF device setup
    try:
        from utils.platform_utils import configure_environment, configure_tf_devices

        configure_environment()
        configure_tf_devices()
    except ImportError:
        pass
    # Load base config and apply HPO overrides
    config = load_config(current_dir=models_dir, algo_key=algo_key)
    with open(args.config) as f:
        overrides = json.load(f)
    apply_config_override(config, overrides)
    print(f"Applied HPO overrides from {args.config}")
    # Set all random seeds
    set_random_seed(args.seed)
    print(f"Random seed: {args.seed}")
    # Output directory
    output_dir = args.output_dir or os.path.join(
        rl_control_dir, "logs", algo_key, "optuna", "seeds"
    )
    seed_dir = os.path.join(output_dir, f"seed_{args.seed}")
    os.makedirs(seed_dir, exist_ok=True)
    print(f"Seed output: {seed_dir}")
    AlgoClass = ALGO_CLASSES[algo_key]
    is_on_policy = ALGO_ON_POLICY[algo_key]
    # Environment
    env_cls = EmissionControlEnvThermal if args.use_thermal else EmissionControlEnv

    def make_env(rank):
        def _init():
            e = env_cls(config_module=config, random_target=True)
            e.reset(seed=args.seed * 1000 + rank)
            return Monitor(e, os.path.join(seed_dir, str(rank)))

        return _init

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecNormalize(
        env, norm_obs=is_on_policy, norm_reward=is_on_policy, clip_obs=10.0
    )
    # Build model
    model_kwargs = _build_model_kwargs(
        algo_key, config, env, seed_dir, args.agent_device
    )
    model = AlgoClass("MlpPolicy", env, **model_kwargs)
    # Callbacks
    checkpoint_freq = 100_000
    name_prefix = f"{algo_key}_seed_{args.seed}"
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
    # Train
    total_timesteps = args.total_timesteps
    print(
        "=" * 40,
        f"Seed {args.seed}: {algo_key.upper()} training ({total_timesteps:,} steps)",
    )
    t0 = time.perf_counter()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, vec_norm_cb],
    )
    duration = time.perf_counter() - t0
    duration_hms = str(datetime.timedelta(seconds=int(duration)))
    print(f"Seed {args.seed}: training finished in {duration_hms}")
    # Save final model + normaliser
    final_name = f"{algo_key}_seed_{args.seed}_final"
    model_path = os.path.join(seed_dir, final_name)
    model.save(model_path)
    env.save(os.path.join(seed_dir, "vec_normalize.pkl"))
    # Save train config for provenance
    train_config = {
        "algorithm": algo_key.upper(),
        "seed": args.seed,
        "total_timesteps": total_timesteps,
        "config_override_path": os.path.abspath(args.config),
        "overrides": overrides,
        "training_duration_seconds": round(duration, 3),
        "training_duration_hms": duration_hms,
        "n_envs": N_ENVS,
        # Reward weights (from config, unchanged)
        "w_speed": config.W_SPEED,
        "w_emission": config.W_EMISSION,
        "w_fuel": config.W_FUEL,
        "w_brake": config.W_BRAKE,
        "w_soc": config.W_SOC,
        "w_soc_squared": config.W_SOC_SQUARED,
        "w_flicker": config.W_FLICKER,
    }
    with open(os.path.join(seed_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    env.close()
    # Full evaluation (reuses models/eval.py)
    print("=" * 40, f"Seed {args.seed}: running full evaluation")

    evaluate_model(
        model_path,
        eval_log_dir=seed_dir,
        train_config=train_config,
        algorithm=algo_key,
        use_thermal=args.use_thermal,
        random_target=True,
    )
    print(f"Seed {args.seed}: complete → {seed_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one seed with best HPO config + full evaluation."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["ppo", "sac", "td3"],
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to best_params.json from HPO.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed (0-9 for 10-seed validation).",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=4_000_000,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Parent dir for seed_N/ folders (default: logs/{algo}/optuna/seeds).",
    )
    parser.add_argument(
        "--agent_device",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--use_thermal",
        action="store_true",
        default=False,
    )
    main(parser.parse_args())
