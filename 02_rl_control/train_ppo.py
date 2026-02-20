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
    from .plotting import TrainingLivePlotCallback, plot_evaluation, plot_actions
    from . import config
except ImportError:
    from env import EmissionControlEnv
    from plotting import TrainingLivePlotCallback, plot_evaluation, plot_actions
    import config


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
    }

    # Instantiate the agent
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
        "ice_speed_rpm": [],
        "em2_torque_nm": [],
        "brake_perc": [],
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
    # Initial actions are zero or undefined, appending 0 for consistency
    eval_results["ice_speed_rpm"].append(0.0)
    eval_results["em2_torque_nm"].append(0.0)
    eval_results["brake_perc"].append(0.0)

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

        # Collect actions
        eval_results["ice_speed_rpm"].append(info.get("ice_speed_rpm", 0.0))
        eval_results["em2_torque_nm"].append(info.get("em2_torque_nm", 0.0))
        eval_results["brake_perc"].append(info.get("brake_perc", 0.0))

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
    dt = 0.5  # 1 step = 0.5s
    # fuel is already per step, so no conversion needed for dt
    total_fuel_g = np.sum(fuel_mg) / 1000.0  # Convert mg to g
    total_nox_g = np.sum(nox_gs) * dt  # dt = 0.5s, g/s * 0.5s = g
    total_co_g = np.sum(co_gs) * dt  # dt = 0.5s

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
        "configuration": train_config,
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

    # Calculate and save emissions per km
    calculate_emissions_per_km(eval_results, log_dir)

    # Generate Plots
    plot_evaluation(eval_results, log_dir)
    plot_actions(eval_results, log_dir, window_start=300, window_size=30)


def calculate_emissions_per_km(results, log_dir):
    speed_actual = np.array(results["speed_actual"])
    nox_gs = np.array(results["nox"])
    co_gs = np.array(results["co"])

    dt = 0.5  # 1 step = 0.5 seconds

    distance_km = 0.0
    accumulated_nox_mg = 0.0
    accumulated_co_mg = 0.0

    km_counter = 1

    out_path = os.path.join(log_dir, "emissions_per_km.txt")
    with open(out_path, "w") as f:
        f.write(
            "Kilometer, NOx (mg/km), CO (mg/km), NOx_Pass (<=80 mg/km), CO_Pass (<=500 mg/km)\n"
        )

        for v, nox, co in zip(speed_actual, nox_gs, co_gs):
            dist_step = v * dt / 3600.0

            # Emissions in mg for this step = rate (g/s) * dt (s) * 1000 (mg/g)
            nox_step_mg = nox * dt * 1000.0
            co_step_mg = co * dt * 1000.0

            distance_km += dist_step
            accumulated_nox_mg += nox_step_mg
            accumulated_co_mg += co_step_mg

            if distance_km >= 1.0:
                nox_pass = accumulated_nox_mg <= 80.0
                co_pass = accumulated_co_mg <= 500.0

                f.write(
                    f"{km_counter}, {accumulated_nox_mg:.2f}, {accumulated_co_mg:.2f}, {nox_pass}, {co_pass}\n"
                )

                # Reset accumulators
                distance_km = 0.0
                accumulated_nox_mg = 0.0
                accumulated_co_mg = 0.0
                km_counter += 1

        # Handle the remaining partial kilometer
        if distance_km > 0.1:  # Only report if at least 100m driven
            nox_per_km = accumulated_nox_mg / distance_km
            co_per_km = accumulated_co_mg / distance_km
            nox_pass = nox_per_km <= 80.0
            co_pass = co_per_km <= 500.0
            f.write(
                f"Partial ({distance_km:.2f} km), {nox_per_km:.2f}, {co_per_km:.2f}, {nox_pass}, {co_pass}\n"
            )


if __name__ == "__main__":
    main()
