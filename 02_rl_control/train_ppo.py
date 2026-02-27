import os
import time
import datetime
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import warnings
import argparse

# Suppress sklearn warnings about feature names
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

# Adjust sys.path to ensure imports work if running from this directory
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

try:
    from .env import EmissionControlEnv
    from .plotting import TrainingLivePlotCallback, plot_evaluation, plot_actions
    from . import config
    from .eval_ppo import evaluate_model
    from .utils import safety_utils
except ImportError:
    from env import EmissionControlEnv
    from plotting import TrainingLivePlotCallback, plot_evaluation, plot_actions
    import config
    from eval_ppo import evaluate_model
    from utils import safety_utils


def main(args):
    # Create Log Directory
    base_log_dir = os.path.join(current_dir, "logs")
    run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")

    # Create Environment
    # Use Monitor to log episode rewards/lengths to csv for the callback
    env = EmissionControlEnv()
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))

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
        "w_flicker": config.W_FLICKER,
        "continued_run": args.continue_from is not None,
        "continued_from": args.continue_from,
    }

    if args.continue_from:
        safety_utils.config_check(args.continue_from, train_config)

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
        )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_emission_model",
    )

    # Plot callback: updates training_progress.png every 1000 steps
    plot_callback = TrainingLivePlotCallback(check_freq=1000, log_dir=log_dir)

    # Train the agent
    print("Starting Training...")
    model.learn(
        total_timesteps=train_config["total_timesteps"],
        callback=[checkpoint_callback, plot_callback],
    )

    print("Training finished.")

    # Save the final model
    model.save(os.path.join(log_dir, "ppo_emission_final"))
    print(f"Model saved to {log_dir}")

    # --- Evaluation Phase ---
    print("Evaluating Model...")

    evaluate_model(
        os.path.join(log_dir, "ppo_emission_final"),
        eval_log_dir=log_dir,
        train_config=train_config,
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
    args = parser.parse_args()

    main(args)
