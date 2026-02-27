import os
import sys
import argparse
import datetime
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

try:
    from .env import EmissionControlEnv
    from .plotting import TrainingLivePlotCallback, plot_evaluation, plot_actions
except ImportError:
    from env import EmissionControlEnv
    from plotting import TrainingLivePlotCallback, plot_evaluation, plot_actions


def calculate_emissions_per_km(results, log_dir):
    speed_actual = np.array(results["speed_actual"])
    nox_gs = np.array(results["nox"])

    dt = 0.5  # 1 step = 0.5 seconds

    distance_km = 0.0
    accumulated_nox_mg = 0.0

    km_counter = 1

    out_path = os.path.join(log_dir, "emissions_per_km.txt")
    with open(out_path, "w") as f:
        f.write("Kilometer, NOx (mg/km), NOx_Pass (<=80 mg/km)\n")

        for v, nox in zip(speed_actual, nox_gs):
            dist_step = v * dt / 3600.0

            # Emissions in mg for this step = rate (g/s) * dt (s) * 1000 (mg/g)
            nox_step_mg = nox * dt * 1000.0

            distance_km += dist_step
            accumulated_nox_mg += nox_step_mg

            if distance_km >= 1.0:
                nox_pass = accumulated_nox_mg <= 80.0

                f.write(f"{km_counter}, {accumulated_nox_mg:.2f}, {nox_pass}\n")

                # Reset accumulators
                distance_km = 0.0
                accumulated_nox_mg = 0.0
                km_counter += 1

        # Handle the remaining partial kilometer
        if distance_km > 0.1:  # Only report if at least 100m driven
            nox_per_km = accumulated_nox_mg / distance_km
            nox_pass = nox_per_km <= 80.0
            f.write(f"Partial ({distance_km:.2f} km), {nox_per_km:.2f}, {nox_pass}\n")


def evaluate_model(model_path, eval_log_dir=None, train_config=None):
    # Validate model path (Stable-Baselines3 usually adds .zip automatically, so we check both)
    if not (os.path.exists(model_path) or os.path.exists(model_path + ".zip")):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    if eval_log_dir is None:
        # Create Log Directory for evaluation
        base_log_dir = os.path.join(current_dir, "logs")
        run_name = datetime.datetime.now().strftime("eval_%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = eval_log_dir
    print(f"Logging evaluation results to {log_dir}")

    # Initialize environment
    print("Evaluating Model...")
    env = EmissionControlEnv()
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
        "fuel": [],
        "engine_on": [],
        "ice_speed_rpm": [],
        "em2_torque_nm": [],
        "brake_perc": [],
    }

    # Store initial state
    eval_results["speed_actual"].append(obs[0])
    eval_results["speed_target"].append(
        obs[0] + obs[1]
    )  # Target Speed = Car_Speed + Speed_Error
    eval_results["soc"].append(obs[2])
    eval_results["ice_torque"].append(obs[3])
    eval_results["nox"].append(obs[4])
    eval_results["fuel"].append(0.0)
    eval_results["engine_on"].append(False)
    eval_results["ice_speed_rpm"].append(0.0)
    eval_results["em2_torque_nm"].append(0.0)
    eval_results["brake_perc"].append(0.0)

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        eval_results["speed_actual"].append(obs[0])
        eval_results["speed_target"].append(
            obs[0] + obs[1]
        )  # Target Speed = Car_Speed + Speed_Error
        eval_results["soc"].append(obs[2])
        eval_results["ice_torque"].append(obs[3])
        eval_results["nox"].append(obs[4])
        eval_results["fuel"].append(info.get("fuel", 0.0))
        eval_results["engine_on"].append(info.get("engine_on", False))
        eval_results["ice_speed_rpm"].append(info.get("ice_speed_rpm", 0.0))
        eval_results["em2_torque_nm"].append(info.get("em2_torque_nm", 0.0))
        eval_results["brake_perc"].append(info.get("brake_perc", 0.0))

    print(f"Evaluation finished. Total Reward: {total_reward}")

    # --- Calculate Metrics ---
    speed_actual = np.array(eval_results["speed_actual"])
    speed_target = np.array(eval_results["speed_target"])
    fuel_mg = np.array(eval_results["fuel"])
    nox_gs = np.array(eval_results["nox"])
    soc = np.array(eval_results["soc"])

    dt = 0.5
    total_fuel_g = np.sum(fuel_mg) / 1000.0
    total_nox_g = np.sum(nox_gs) * dt

    speed_error = speed_actual - speed_target
    mae_speed = np.mean(np.abs(speed_error))
    rmse_speed = np.sqrt(np.mean(speed_error**2))

    initial_soc = soc[0]
    final_soc = soc[-1]
    delta_soc = final_soc - initial_soc

    metrics = {
        "model_path": model_path,
    }

    continued_run = False
    continued_from = None

    if train_config is not None:
        metrics["configuration"] = train_config
        continued_run = train_config.get("continued_run", False)
        continued_from = train_config.get("continued_from", None)

    metrics.update(
        {
            "continued_run": continued_run,
            "continued_from": continued_from,
            "total_reward": float(total_reward),
            "total_fuel_g": float(total_fuel_g),
            "total_nox_g": float(total_nox_g),
            "mae_speed_kmph": float(mae_speed),
            "rmse_speed_kmph": float(rmse_speed),
            "initial_soc": float(initial_soc),
            "final_soc": float(final_soc),
            "delta_soc": float(delta_soc),
            "custom_notes": "",
        }
    )

    metrics_path = os.path.join(log_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    df_res = pd.DataFrame(eval_results)
    csv_path = os.path.join(log_dir, "evaluation_data.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Evaluation data saved to {csv_path}")

    calculate_emissions_per_km(eval_results, log_dir)
    plot_evaluation(eval_results, log_dir)
    plot_actions(eval_results, log_dir, window_start=1, window_size=3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model.")
    parser.add_argument(
        "model_path", type=str, help="Path to the trained PPO model (.zip)"
    )
    args = parser.parse_args()

    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    evaluate_model(args.model_path, eval_log_dir=model_dir)
