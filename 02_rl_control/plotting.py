import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


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
    Plot evaluation results: Speed, SOC, Emissions (NOx, CO).
    """
    time_steps = np.arange(len(results["speed_actual"]))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)

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
    axes[0].set_ylim(-10, 140)
    axes[0].legend()
    axes[0].grid(True)

    # 2. SOC
    axes[1].plot(time_steps, results["soc"], label="SOC", color="green")
    axes[1].set_ylabel("SOC")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("State of Charge (1 = full, 0 = empty)")
    axes[1].axhline(y=0.2, color="r", linestyle=":", alpha=0.5)
    axes[1].axhline(y=0.9, color="r", linestyle=":", alpha=0.5)
    axes[1].grid(True)

    # 3. NOx
    nox_mg = np.array(results["nox"]) * 0.5 * 1000.0
    axes[2].plot(time_steps, nox_mg, label="NOx (Tailpipe)", color="orange", alpha=0.9)
    axes[2].set_ylabel("NOx (mg)")
    axes[2].set_xlabel("Time Step (0.5s)")
    axes[2].set_title("NOx Emissions per Step (mg)")
    axes[2].set_ylim(-1, 30)
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "evaluation_results.png"))
    plt.close()
    print(
        f"Evaluation plots saved to {os.path.join(log_dir, 'evaluation_results.png')}"
    )


def plot_actions(results, log_dir, window_start=300, window_size=30):
    """
    Plot control actions [ICE Speed, EM2 Torque, Fuel, Brake] for a specific time window.
    Default window: 300s to 330s.
    """
    # Slice the data
    start_idx = window_start
    end_idx = window_start + window_size

    # Ensure indices are within bounds
    total_len = len(results["speed_actual"])
    if start_idx >= total_len:
        print(
            f"Window start {start_idx} is beyond data length {total_len}. Plotting last {window_size} steps."
        )
        start_idx = max(0, total_len - window_size)
        end_idx = total_len

    end_idx = min(end_idx, total_len)

    time_steps = np.arange(start_idx, end_idx)

    # Extract data slices
    engine_on = results.get("engine_on", [False] * total_len)[start_idx:end_idx]
    ice_speed = results["ice_speed_rpm"][start_idx:end_idx]
    em2_torque = results["em2_torque_nm"][start_idx:end_idx]
    fuel = results["fuel"][start_idx:end_idx]
    brake = results["brake_perc"][start_idx:end_idx]

    fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=False)

    # 1. Engine State
    axes[0].step(time_steps, engine_on, label="Engine On", color="purple", where="mid")
    axes[0].set_ylabel("Boolean")
    axes[0].set_title("ICE Engine State")
    axes[0].set_ylim(-0.2, 1.2)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Off", "On"])
    axes[0].grid(True)
    axes[0].legend()

    # 2. ICE Speed
    axes[1].plot(time_steps, ice_speed, label="ICE Speed (RPM)", color="blue")
    axes[1].set_ylabel("RPM")
    axes[1].set_title("ICE Speed")
    axes[1].grid(True)
    axes[1].legend()

    # 3. EM2 Torque
    axes[2].plot(time_steps, em2_torque, label="EM2 Torque (Nm)", color="green")
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_title("EM2 Torque")
    axes[2].grid(True)
    axes[2].legend()

    # 4. Fuel Injection
    axes[3].plot(time_steps, fuel, label="Fuel Injection (mg)", color="orange")
    axes[3].set_ylabel("Fuel (mg)")
    axes[3].set_title("Fuel Injection per Step")
    axes[3].grid(True)
    axes[3].legend()

    # 5. Brake
    axes[4].plot(time_steps, brake, label="Brake (%)", color="red")
    axes[4].set_ylabel("Brake (%)")
    axes[4].set_xlabel("Time Step (s)")
    axes[4].set_title("Brake Command")
    axes[4].set_ylim(-5.0, 105.0)
    axes[4].grid(True)
    axes[4].legend()

    plt.tight_layout()
    save_path = os.path.join(log_dir, "action_results.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Action plots saved to {save_path}")
