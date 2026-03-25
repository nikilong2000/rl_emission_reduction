# DEPRECATED: Use the generic train.py in models/ instead:
#   python models/train.py --algorithm ppo [--random_target] [--use_thermal]
import os
import time
import json
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import warnings
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from env import EmissionControlEnv
from env_thermal import EmissionControlEnvThermal
from plotting import (
    TrainingLivePlotCallback,
    ExplorationEntropyCallback,
)
import config
from eval_ppo import evaluate_model
from utils import safety_utils
from utils.checkpoint_utils import VecNormalizeCheckpointCallback

# suppress sklearn warnings about feature names
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)


def main(args):
    # Create Log Directory
    base_log_dir = os.path.join(
        os.path.dirname(os.path.dirname(current_dir)), "logs", "ppo"
    )
    run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # Create Environment
    # Use Monitor to log episode rewards/lengths to csv for the callback
    def make_env():
        env_cls = EmissionControlEnvThermal if args.use_thermal else EmissionControlEnv
        e = env_cls(config_module=config, random_target=args.random_target)
        return Monitor(e, os.path.join(log_dir, "monitor.csv"))

    env = DummyVecEnv([make_env])

    # Hyperparameters; keep for file writing
    train_config = {
        "learning_rate": config.LEARNING_RATE,
        "n_steps": config.N_STEPS,
        "batch_size": config.BATCH_SIZE,
        "n_epochs": config.N_EPOCHS,
        "gamma": config.GAMMA,
        "gae_lambda": config.GAE_LAMBDA,
        "clip_range": config.CLIP_RANGE,
        "total_timesteps": config.TOTAL_TIMESTEPS,
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

    if args.continue_from:
        safety_utils.config_check(args.continue_from, train_config)

        vec_norm_path = os.path.join(
            os.path.dirname(args.continue_from), "vec_normalize.pkl"
        )
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = True
            env.norm_reward = True
            print(f"Loaded VecNormalize stats from {vec_norm_path}")
        else:
            print(
                "Warning: Could not find vec_normalize.pkl. Starting fresh normalizer."
            )
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Instantiate the agent
    if args.continue_from:
        print(
            f"Loading existing model from {args.continue_from} to continue training..."
        )
        model = PPO.load(
            args.continue_from,
            env=env,
            verbose=1,
            learning_rate=train_config["learning_rate"],
            n_steps=train_config["n_steps"],
            batch_size=train_config["batch_size"],
            n_epochs=train_config["n_epochs"],
            gamma=train_config["gamma"],
            gae_lambda=train_config["gae_lambda"],
            clip_range=train_config["clip_range"],
            tensorboard_log=log_dir,
            device=args.agent_device,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=train_config["learning_rate"],
            n_steps=train_config["n_steps"],
            batch_size=train_config["batch_size"],
            n_epochs=train_config["n_epochs"],
            gamma=train_config["gamma"],
            gae_lambda=train_config["gae_lambda"],
            clip_range=train_config["clip_range"],
            tensorboard_log=log_dir,
            device=args.agent_device,
        )

    # Callbacks
    CHECKPOINT_FREQ = 100000
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_emission_model",
    )
    vec_normalize_checkpoint_callback = VecNormalizeCheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_emission_model",
        vec_normalize=env,
    )

    # Plot callback: updates training_progress.png every 1000 steps
    plot_callback = TrainingLivePlotCallback(check_freq=1000, log_dir=log_dir)
    entropy_callback = ExplorationEntropyCallback(plot_freq=10, log_dir=log_dir)

    # Train the agent
    print("Starting Training...")
    training_start_time = time.perf_counter()
    model.learn(
        total_timesteps=train_config["total_timesteps"],
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
        f"Training finished in {training_duration_seconds:.2f}s "
        f"({training_duration_hms})."
    )

    # Save the final model
    model.save(os.path.join(log_dir, "ppo_emission_final"))
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    with open(os.path.join(log_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    print(f"Model and VecNormalize stats saved to {log_dir}")

    # --- Evaluation Phase ---
    print("Evaluating Model...")

    evaluate_model(
        os.path.join(log_dir, "ppo_emission_final"),
        eval_log_dir=log_dir,
        train_config=train_config,
        use_thermal=args.use_thermal,
        random_target=args.random_target,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or continue training a PPO model."
    )
    parser.add_argument(
        "--continue_from",
        type=str,
        default=None,
        help="Path to an existing model (.zip) to continue training from",
    )
    parser.add_argument(
        "--agent_device",
        type=str,
        default="auto",
        help="Device for the PPO agent (e.g. 'cpu', 'cuda', 'auto'). "
        "Use 'cpu' when GPU is reserved for environment model inference.",
    )
    parser.add_argument(
        "--use_thermal",
        action="store_true",
        help="Use EmissionControlEnvThermal instead of EmissionControlEnv.",
    )
    parser.add_argument(
        "--random_target",
        action="store_true",
        default=True,
        help="Use a random constant target speed (0-250 km/h) instead of CSV trajectories.",
    )
    args = parser.parse_args()

    main(args)
