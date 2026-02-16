import os
import sys
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import shutil
from datetime import datetime

# Add the current directory to sys.path to import from Modular_NN
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

sys.path.append(
    os.path.join(current_dir, "..", "internal_lstm_models", "NN_Application")
)

# Import helper functions from Modular_NN
from Modular_NN import load_network, set_states, load_scaler

import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

# Configuration
ICE_MODEL_DIR = os.path.join(
    current_dir, "../internal_lstm_models/NN_Application/Nets/ICE"
)
PG_MODEL_DIR = os.path.join(
    current_dir, "../internal_lstm_models/NN_Application/Nets/Drivetrain"
)
INPUT_DATA_PATH = os.path.join(
    current_dir, "../internal_lstm_models/NN_Application/Input_data/WLTC.csv"
)
OUTPUT_DIR = os.path.join(current_dir, "output/RL_Loop_Simulation")

# Input columns definition (ORDER MATTERS!)
ICE_INPUTS = ["ICE_Speed_rpm", "fuel_mg", "T_amb_K", "p_amb_bar"]
PG_INPUTS_CSV = [
    "ICE_Speed_rpm",
    "EM2_Torque_Nm",
    "Brake_perc",
]  # ICE_Torque_Nm comes from ICE model

# Initial outputs for state initialization (taken from config.txt)
# ICE: 0, 0, 0, 0, 0, 298, 0, 0, 0, 0, 298, 298, 298, 298, 298, 298
ICE_INITIAL_OUTPUTS = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    298.0,
    0.0,
    0.0,
    0.0,
    0.0,
    298.0,
    298.0,
    298.0,
    298.0,
    298.0,
    298.0,
]
# Drivetrain: 0, 0.7
PG_INITIAL_OUTPUTS = [0.0, 0.7]


def main():
    print("Starting RL Loop Simulation...")

    # 1. Load Models and Scalers
    print("Loading ICE Model...")
    ice_model_tuple = load_network(ICE_MODEL_DIR)
    (
        ice_main,
        ice_init,
        ice_in_scaler,
        ice_out_scaler,
        ice_predict_main,
        ice_predict_init,
    ) = ice_model_tuple

    print("Loading PG (Drivetrain) Model...")
    pg_model_tuple = load_network(PG_MODEL_DIR)
    pg_main, pg_init, pg_in_scaler, pg_out_scaler, pg_predict_main, pg_predict_init = (
        pg_model_tuple
    )

    # 2. Load Data
    print(f"Loading data from {INPUT_DATA_PATH}...")
    # Attempt to read with common delimiters
    try:
        df = pd.read_csv(INPUT_DATA_PATH, delimiter=";", encoding="latin1")
        if df.shape[1] <= 1:
            df = pd.read_csv(INPUT_DATA_PATH, delimiter=",", encoding="latin1")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Clean column names (remove units in brackets if present in header string match)
    # The helper in Modular_NN does some cleaning, we do manual mapping to be safe
    # Based on WLTC.csv content viewed: Columns are like 'ICE_Speed_rpm', 'fuel_mg' etc.
    # We will strip whitespace just in case
    df.columns = [col.strip() for col in df.columns]

    # Verify columns exist
    for col in ICE_INPUTS:
        if col not in df.columns:
            # Try finding with partial match if exact match fails (handling 'v (km/h)' etc)
            pass
            # For now assume they exist as per task description and file view

    # 3. Initialize States
    print("Initializing Model States...")

    # ICE Initialization
    # model_init takes scaled initial output values as input to predict the initial hidden states
    ice_init_vals = np.array([ICE_INITIAL_OUTPUTS]).reshape(1, len(ICE_INITIAL_OUTPUTS))
    ice_init_vals_scaled = ice_out_scaler.transform(ice_init_vals).reshape(1, 1, -1)
    ice_states_tensors = ice_predict_init(ice_init_vals_scaled)
    ice_states_dict = dict(zip(ice_init.output_names, ice_states_tensors))
    set_states(ice_main, ice_states_dict)

    # PG Initialization
    pg_init_vals = np.array([PG_INITIAL_OUTPUTS]).reshape(1, len(PG_INITIAL_OUTPUTS))
    pg_init_vals_scaled = pg_out_scaler.transform(pg_init_vals).reshape(1, 1, -1)
    pg_states_tensors = pg_predict_init(pg_init_vals_scaled)
    pg_states_dict = dict(zip(pg_init.output_names, pg_states_tensors))
    set_states(pg_main, pg_states_dict)

    # 4. Simulation Loop
    num_steps = len(df)
    results = {
        "time": [],
        "ice_torque_pred": [],
        "car_speed_pred": [],
        "soc_pred": [],
        "ice_torque_true": (
            df["ICE_Torque_Nm"].values if "ICE_Torque_Nm" in df.columns else []
        ),
        "car_speed_true": (
            df["Car_Speed_kmph"].values if "Car_Speed_kmph" in df.columns else []
        ),
        "soc_true": df["SOC_1"].values if "SOC_1" in df.columns else [],
    }

    start_time = time.time()

    print(f"Running simulation for {num_steps} steps...")

    for t in range(num_steps):
        # --- ICE Prediction ---
        # Construct Input: ICE_Speed_rpm, fuel_mg, T_amb_K, p_amb_bar
        # Extract values for current timestep
        ice_in_values = df.loc[t, ICE_INPUTS].values.astype(np.float32).reshape(1, -1)

        # Scale Input
        ice_in_scaled = ice_in_scaler.transform(ice_in_values)
        ice_in_reshaped = ice_in_scaled.reshape(1, 1, -1)  # (batch, time, features)

        # Predict
        ice_pred_scaled = ice_predict_main(ice_in_reshaped)  # Returns tensor

        # Inverse Scale Output
        ice_pred = ice_out_scaler.inverse_transform(ice_pred_scaled.numpy()[0])

        # Extract ICE Torque (Assume it's the first output based on config.txt)
        # outputs = ICE_Torque_Nm, fuel_tot_gps, ...
        ice_torque_pred = ice_pred[0][0]

        # --- PG Prediction ---
        # Inputs: ICE_Speed_rpm, ICE: ICE_Torque_Nm, EM2_Torque_Nm, Brake_perc

        # Get CSV inputs
        ice_speed = df.loc[t, "ICE_Speed_rpm"]
        em2_torque = df.loc[t, "EM2_Torque_Nm"]
        brake_perc = df.loc[t, "Brake_perc"]

        # Construct PG Input Vector strictly in order
        pg_in_values = np.array(
            [[ice_speed, ice_torque_pred, em2_torque, brake_perc]], dtype=np.float32
        )

        # Scale Input
        pg_in_scaled = pg_in_scaler.transform(pg_in_values)
        pg_in_reshaped = pg_in_scaled.reshape(1, 1, -1)

        # Predict
        pg_pred_scaled = pg_predict_main(pg_in_reshaped)

        # Inverse Scale Output
        pg_pred = pg_out_scaler.inverse_transform(pg_pred_scaled.numpy()[0])

        # Extract Outputs: Car_Speed_kmph, SOC_1
        car_speed_pred = pg_pred[0][0]
        soc_pred = pg_pred[0][1]

        # --- Logging ---
        results["time"].append(df.loc[t, "Time_s"] if "Time_s" in df.columns else t)
        results["ice_torque_pred"].append(ice_torque_pred)
        results["car_speed_pred"].append(car_speed_pred)
        results["soc_pred"].append(soc_pred)

        if t % 100 == 0:
            print(f"Step {t}/{num_steps}", end="\r")

    total_time = time.time() - start_time
    print(
        f"\nSimulation complete in {total_time:.2f} seconds ({total_time/num_steps*1000:.2f} ms/step)"
    )

    # 5. Visualization
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Plot ICE Torque
    plt.figure(figsize=(12, 6))
    plt.plot(results["time"], results["ice_torque_true"], label="True", alpha=0.7)
    plt.plot(results["time"], results["ice_torque_pred"], label="Predicted", alpha=0.7)
    plt.title("ICE Torque Prediction (Closed Loop)")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "ice_torque.png"))
    plt.close()

    # Plot Car Speed
    plt.figure(figsize=(12, 6))
    plt.plot(results["time"], results["car_speed_true"], label="True", alpha=0.7)
    plt.plot(results["time"], results["car_speed_pred"], label="Predicted", alpha=0.7)
    plt.title("Car Speed Prediction (Closed Loop)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (km/h)")
    plt.yticks(np.arange(-40.0, 140.0, step=20.0))  # To align with the other plots
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "car_speed.png"))
    plt.close()

    # Plot SOC
    plt.figure(figsize=(12, 6))
    plt.plot(results["time"], results["soc_true"], label="True", alpha=0.7)
    plt.plot(results["time"], results["soc_pred"], label="Predicted", alpha=0.7)
    plt.title("SOC Prediction (Closed Loop)")
    plt.xlabel("Time (s)")
    plt.ylabel("SOC")
    plt.yticks(np.arange(0, 0.9, step=0.2))  # To align with the other plots
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "soc.png"))
    plt.close()

    print(f"Plots saved to {OUTPUT_DIR}")

    # Save Results to CSV
    results_df = pd.DataFrame(results)
    csv_output_path = os.path.join(OUTPUT_DIR, "simulation_results.csv")
    results_df.to_csv(csv_output_path, index=False)
    print(f"Results saved to {csv_output_path}")


if __name__ == "__main__":
    main()
