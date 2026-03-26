"""
Generic training script for PPO, SAC, and TD3 emission-control agents.

Usage:
    python train.py --algorithm ppo --random_target
    python train.py --algorithm sac --random_target --use_thermal
    python train.py --algorithm td3 --continue_from path/to/model.zip
"""

import os
import sys
import json
import datetime
import time
import warnings
import argparse

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(current_dir)

from env import EmissionControlEnv
from env_thermal import EmissionControlEnvThermal
from plotting import TrainingLivePlotCallback, ExplorationEntropyCallback
from utils.config_utils import config_check, load_config
from utils.checkpoint_utils import VecNormalizeCheckpointCallback

# Suppress sklearn warnings about feature names
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------
ALGORITHM_REGISTRY = {
    "ppo": {
        "class": PPO,
        "on_policy": True,
    },
    "sac": {
        "class": SAC,
        "on_policy": False,
    },
    "td3": {
        "class": TD3,
        "on_policy": False,
    },
}


def _build_train_config(algo_key: str, config, args) -> dict:
    """Build a JSON-serialisable dict with all hyperparameters for reproducibility."""
    tc = {
        "algorithm": algo_key.upper(),
        "env": "thermal" if args.use_thermal else "base",
        "total_timesteps": config.TOTAL_TIMESTEPS,
        # Reward weights (shared across all algorithms)
        "w_speed": config.W_SPEED,
        "w_emission": config.W_EMISSION,
        "w_fuel": config.W_FUEL,
        "w_brake": config.W_BRAKE,
        "w_soc": config.W_SOC,
        "w_soc_squared": config.W_SOC_SQUARED,
        "w_flicker": config.W_FLICKER,
        "use_onnx": bool(getattr(config, "USE_ONNX", False)),
        "use_thermal": args.use_thermal,
        "random_target": args.random_target,
        "continued_run": args.continue_from is not None,
        "continued_from": args.continue_from,
    }

    # Algorithm-specific hyperparameters
    if algo_key == "ppo":
        tc.update(
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
        tc.update(
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
        tc.update(
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
                "action_noise_sigma": config.ACTION_NOISE_SIGMA,
            }
        )

    return tc


def _build_model_kwargs(algo_key: str, config, env, args, log_dir) -> dict:
    """Return the keyword arguments for the SB3 model constructor."""
    common = {
        "verbose": 1,
        "tensorboard_log": log_dir,
        "device": args.agent_device,
    }

    if algo_key == "ppo":
        common.update(
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
        common.update(
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
        if not args.continue_from:
            common["learning_starts"] = config.LEARNING_STARTS
    elif algo_key == "td3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=config.ACTION_NOISE_SIGMA * np.ones(n_actions),
        )
        common.update(
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
                "action_noise": action_noise,
            }
        )
        if not args.continue_from:
            common["learning_starts"] = config.LEARNING_STARTS

    return common


def main(args):
    algo_key = args.algorithm.lower()
    if algo_key not in ALGORITHM_REGISTRY:
        print(
            f"Error: Unknown algorithm '{algo_key}'. Choose from: {list(ALGORITHM_REGISTRY.keys())}"
        )
        sys.exit(1)

    algo_info = ALGORITHM_REGISTRY[algo_key]
    AlgoClass = algo_info["class"]
    is_on_policy = algo_info["on_policy"]

    # GPU setup for the LSTM environment models (TensorFlow)
    try:
        from utils.platform_utils import configure_environment, configure_tf_devices

        configure_environment()
        configure_tf_devices()
    except ImportError:
        print(f"Error: utils.platform_utils not available. Check the import.")
        pass

    # Load algorithm-specific config
    config = load_config(current_dir=current_dir, algo_key=algo_key)

    # Allow CLI flag to override config value for SAC's SDE
    if hasattr(args, "use_sde") and args.use_sde:
        config.USE_SDE = True

    # Create Log Directory
    base_log_dir = os.path.join(os.path.dirname(current_dir), "logs", algo_key)
    run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")

    # Create Environment
    env_cls = EmissionControlEnvThermal if args.use_thermal else EmissionControlEnv

    def make_env():
        e = env_cls(config_module=config, random_target=args.random_target)
        return Monitor(e, os.path.join(log_dir, "monitor.csv"))

    env = DummyVecEnv([make_env])

    # Build config snapshot for reproducibility
    train_config = _build_train_config(algo_key, config, args)
    print(json.dumps(train_config, indent=4))  # print to validate

    # VecNormalize
    # On-policy (PPO): normalise both obs and rewards
    # Off-policy (SAC/TD3): normalise obs only (raw rewards in replay buffer)
    norm_reward = is_on_policy  # TODO: check this because i think its simply a mistake; however sac was performing pretty well

    if args.continue_from:
        config_check(args.continue_from, train_config)

        vec_norm_path = os.path.join(
            os.path.dirname(args.continue_from), "vec_normalize.pkl"
        )
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = True
            env.norm_reward = norm_reward
            print(f"Loaded VecNormalize stats from {vec_norm_path}")
        else:
            print(
                "Warning: Could not find vec_normalize.pkl. Starting fresh normalizer."
            )
            env = VecNormalize(
                env, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0
            )
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)

    # Build model
    model_kwargs = _build_model_kwargs(algo_key, config, env, args, log_dir)

    if args.continue_from:
        print(
            f"Loading existing model from {args.continue_from} to continue training..."
        )
        model = AlgoClass.load(args.continue_from, env=env, **model_kwargs)
    else:
        model = AlgoClass("MlpPolicy", env, **model_kwargs)

    # Callbacks
    CHECKPOINT_FREQ = 100_000
    name_prefix = f"{algo_key}_emission_model"
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix=name_prefix,
    )
    vec_normalize_checkpoint_callback = VecNormalizeCheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix=name_prefix,
        vec_normalize=env,
    )
    plot_callback = TrainingLivePlotCallback(check_freq=1_000, log_dir=log_dir)
    entropy_callback = ExplorationEntropyCallback(plot_freq=10, log_dir=log_dir)

    ### TRAINING ###
    print(40 * "=", f"Starting {algo_key.upper()} Training...")
    training_start_time = time.perf_counter()
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[
            checkpoint_callback,
            vec_normalize_checkpoint_callback,
            plot_callback,
            entropy_callback,
        ],
    )
    training_duration_seconds = time.perf_counter() - training_start_time
    training_duration_hms = str(
        datetime.timedelta(seconds=int(training_duration_seconds))
    )
    train_config["training_duration_seconds"] = round(training_duration_seconds, 3)
    train_config["training_duration_hms"] = training_duration_hms
    print(
        f"\nTraining finished in {training_duration_seconds:.2f}s "
        f"({training_duration_hms})."
    )

    # Save
    final_model_name = f"{algo_key}_emission_final"
    model.save(os.path.join(log_dir, final_model_name))
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    with open(os.path.join(log_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    print(f"\nModel and VecNormalize stats saved to {log_dir}")

    ### EVALUATING ###
    from eval import evaluate_model

    evaluate_model(
        os.path.join(log_dir, final_model_name),
        eval_log_dir=log_dir,
        train_config=train_config,
        algorithm=algo_key,
        use_thermal=args.use_thermal,
        random_target=args.random_target,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or continue training a PPO/SAC/TD3 emission-control model."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["ppo", "sac", "td3"],
        help="RL algorithm to use.",
    )
    parser.add_argument(
        "--continue_from",
        type=str,
        default=None,
        help="Path to an existing model (.zip) to continue training from.",
    )
    parser.add_argument(
        "--agent_device",
        type=str,
        default="auto",
        help="PyTorch device for the agent (e.g. 'cpu', 'cuda', 'auto').",
    )
    parser.add_argument(
        "--use_thermal",
        action="store_true",
        default=False,
        help="Use EmissionControlEnvThermal (10-dim obs with aftertreatment temps).",
    )
    parser.add_argument(
        "--random_target",
        action="store_true",
        default=True,
        help="Use random target speeds instead of CSV trajectories.",
    )
    parser.add_argument(
        "--use_sde",
        action="store_true",
        default=False,
        help="Enable State-Dependent Exploration (SAC only).",
    )
    args = parser.parse_args()

    main(args)
