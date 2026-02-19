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
    from . import config
except ImportError:
    from env import EmissionControlEnv
    import config


class TrainingLivePlotCallback(BaseCallback):
    """
    Callback for plotting training progress (Reward) in real-time (saved to file).
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(TrainingLivePlotCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve mean reward from the monitor file or accumulated rewards
            # SB3 Monitor wrapper writes to a csv file. We can read that or just track episode rewards if we want.
            # But SB3 callbacks don't easily give "current episode reward".
            # Easiest is to read the monitor.csv if it exists.
            try:
                monitor_path = os.path.join(self.log_dir, "monitor.csv")
                if os.path.exists(monitor_path):
                    # Skip first 2 lines (metadata)
                    df = pd.read_csv(monitor_path, skiprows=1)
                    if len(df) > 0:
                        # Use a moving average for smoother plotting
                        rewards = df["r"].values
                        if len(rewards) > 0:
                            mean_reward = np.mean(rewards[-100:])  # Last 100 episodes
                            self.rewards.append(mean_reward)
                            self.timesteps.append(self.num_timesteps)

                            self._plot()
            except Exception as e:
                pass  # Ignore errors during plotting

        return True

    def _plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.timesteps, self.rewards, label="Mean Reward (Last 100 Eps)")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, "training_progress.png"))
        plt.close()


def plot_evaluation(results, log_dir):
    """
    Plot evaluation results: Speed, SOC, Emissions.
    """
    time_steps = np.arange(len(results["speed_actual"]))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 1. Speed
    axes[0].plot(
        time_steps,
        results["speed_target"],
        label="Target Speed",
        color="black",
        alpha=0.9,
        linestyle="--",
    )
    axes[0].plot(
        time_steps,
        results["speed_actual"],
        label="Actual Speed",
        color="blue",
        alpha=0.7,
    )
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].set_title("Speed Tracking")
    axes[0].legend()
    axes[0].grid(True)

    # 2. SOC
    axes[1].plot(time_steps, results["soc"], label="SOC", color="green")
    axes[1].set_ylabel("SOC")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("State of Charge")
    axes[1].axhline(y=0.2, color="r", linestyle=":", alpha=0.5)
    axes[1].axhline(y=0.9, color="r", linestyle=":", alpha=0.5)
    axes[1].grid(True)

    # 3. Emissions
    axes[2].plot(
        time_steps, results["nox"], label="NOx (Tailpipe)", color="orange", alpha=0.8
    )
    axes[2].plot(
        time_steps, results["co"], label="CO (Tailpipe)", color="red", alpha=0.8
    )
    axes[2].set_ylabel("Emissions (g/s)")
    axes[2].set_xlabel("Time Step")
    axes[2].set_title("Emissions")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "evaluation_results.png"))
    plt.close()
    print(
        f"Evaluation plots saved to {os.path.join(log_dir, 'evaluation_results.png')}"
    )


def main():
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

    # Instantiate the agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
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
    TIMESTEPS = 300000  # Increased slightly for better plotting demo

    model.learn(
        total_timesteps=TIMESTEPS, callback=[checkpoint_callback, plot_callback]
    )

    print("Training finished.")

    # Save the final model
    model.save(os.path.join(log_dir, "ppo_emission_final"))
    print(f"Model saved to {log_dir}")

    # --- Evaluation Phase ---
    print("Evaluating Model...")

    # We need to recreate the env without Monitor to avoid overwriting logs,
    # or just use the existing one but reset.
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    # Data collection for plotting
    eval_results = {
        "speed_actual": [],
        "speed_target": [],
        "soc": [],
        "ice_torque": [],
        "nox": [],
        "co": [],
        "fuel": [],
    }

    # Store initial state
    # Obs: [Car_Speed, Target_Speed, SOC, ICE_Torque, NOx, CO]
    eval_results["speed_actual"].append(obs[0])
    eval_results["speed_target"].append(obs[1])
    eval_results["soc"].append(obs[2])
    eval_results["ice_torque"].append(obs[3])
    eval_results["nox"].append(obs[4])
    eval_results["co"].append(obs[5])
    eval_results["fuel"].append(0.0)  # No fuel consumed at step 0

    while not (terminated or truncated):
        action, _states = model.predict(
            obs, deterministic=True
        )  # Deterministic for evaluation
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Obs: [Car_Speed, Target_Speed, SOC, ICE_Torque, NOx, CO]
        eval_results["speed_actual"].append(obs[0])
        eval_results["speed_target"].append(obs[1])
        eval_results["soc"].append(obs[2])
        eval_results["ice_torque"].append(obs[3])
        eval_results["nox"].append(obs[4])
        eval_results["co"].append(obs[5])
        eval_results["fuel"].append(info.get("fuel", 0.0))

    print(f"Evaluation finished. Total Reward: {total_reward}")

    # --- Calculate Metrics ---
    # Convert lists to arrays for calculation
    speed_actual = np.array(eval_results["speed_actual"])
    speed_target = np.array(eval_results["speed_target"])
    fuel_mg = np.array(eval_results["fuel"])
    nox_gs = np.array(eval_results["nox"])
    co_gs = np.array(eval_results["co"])
    soc = np.array(eval_results["soc"])

    # 1. Total Cumulative Values
    total_fuel_g = np.sum(fuel_mg) / 1000.0  # Convert mg to g
    total_nox_g = np.sum(nox_gs)  # Assuming 1Hz (step time = 1s), g/s * 1s = g
    total_co_g = np.sum(co_gs)  # Assuming 1Hz

    # 2. Speed Tracking Metrics
    # Exclude the first few steps if needed, but here we take all
    speed_error = speed_actual - speed_target
    mae_speed = np.mean(np.abs(speed_error))
    rmse_speed = np.sqrt(np.mean(speed_error**2))

    # 3. SOC Metrics
    initial_soc = soc[0]
    final_soc = soc[-1]
    delta_soc = final_soc - initial_soc

    metrics = {
        "total_reward": float(total_reward),
        "total_fuel_g": float(total_fuel_g),
        "total_nox_g": float(total_nox_g),
        "total_co_g": float(total_co_g),
        "mae_speed_kmph": float(mae_speed),
        "rmse_speed_kmph": float(rmse_speed),
        "initial_soc": float(initial_soc),
        "final_soc": float(final_soc),
        "delta_soc": float(delta_soc),
    }

    # Save Metrics to JSON
    import json

    metrics_path = os.path.join(log_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Save Time Series Data to CSV
    df_res = pd.DataFrame(eval_results)
    csv_path = os.path.join(log_dir, "evaluation_data.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Evaluation data saved to {csv_path}")

    # Generate Plots
    plot_evaluation(eval_results, log_dir)


if __name__ == "__main__":
    main()
