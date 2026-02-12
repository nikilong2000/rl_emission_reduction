import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# --- Configuration ---
OUTPUT_DIR = "internal_lstm_models/NN_Application/Output/RL_Loop_Simulation_ONNX"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to the shared folder containing models
BASE_MODEL_DIR = "controller_for_ICE_PG/SHARE/CTTC_models/ONNX"

# Import ONNX_Predict classes
# Since ONNX_Predict is installed in the venv, we can import it directly.
from ONNX_Predict.LSTM_onnx import LSTM_onnx
from ONNX_Predict.Scaler_onnx import Scaler_onnx

# --- Switches ---
# Set to True to mimic "stateless" behavior (resetting memory every step),
# which essentially makes the model forget previous history and rely only on current inputs/feedback.
# This might match the behavior of the old TF models if they were run statelessly.
SIMULATE_LEGACY_STATELESS = False


def main():
    print("Loading data...")
    # Load WLTC data
    csv_path = "internal_lstm_models/NN_Application/Input_data/WLTC.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # User-defined input columns
    input_cols = [
        "ICE_Speed_rpm",
        "fuel_mg",
        "T_amb_K",
        "p_amb_bar",
        "ICE_Speed_soll_rpm",
        "EM2_Torque_Nm",
        "ICE_Torque_Nm",
        "Brake_perc",
    ]

    # Read CSV with correct separator and headers (User correction)
    # The file seems to have a preamble, user used skiprows=1 initially, but in their edit just read_csv(sep=";")
    # The head command output showed data starting immediately after header?
    # Let's trust the user's edit: pd.read_csv(csv_path, sep=";")
    # But wait, original code had skiprows=1 because of units row?
    # Step 369 output: "Time_s ICE_Speed_rpm ..." then row 23.
    # User's edit: df_full = pd.read_csv(csv_path, sep=";")
    # I will stick to the user's edit logic from step 348.

    try:
        df_full = pd.read_csv(csv_path, sep=";")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Handle missing ICE_Speed_soll_rpm (map from ICE_Speed_rpm if needed)
    if (
        "ICE_Speed_soll_rpm" not in df_full.columns
        and "ICE_Speed_rpm" in df_full.columns
    ):
        df_full["ICE_Speed_soll_rpm"] = df_full["ICE_Speed_rpm"]

    # Ensure all required columns exist
    missing_cols = [c for c in input_cols if c not in df_full.columns]
    if missing_cols:
        print(f"Error: Missing columns in CSV: {missing_cols}")
        return

    # Select and reorder columns
    df = df_full[input_cols].copy()

    # Split into ICE and PG inputs
    ICEinput = df.iloc[:, :4].copy()
    PGinput = df.iloc[:, 4:].copy()

    # NOTE: The previous filter "PGinput.loc[PGinput['ICE_Speed_soll_rpm'] < 900] = 0"
    # has been REMOVED because it incorrectly zeroed out EM2 tokens during EV mode,
    # causing the flatline behavior observed by the user.

    print("Loading models...")

    # Define folders
    ice_folder = os.path.join(BASE_MODEL_DIR, "ICE")
    pg_folder = os.path.join(BASE_MODEL_DIR, "PG")

    # LSTM_onnx requires the original .h5 model name to find it in the folder and initialize states.
    # The user confirmed .h5 files are available.
    tf_model_name = "model.h5"

    # Load ICE Components
    ice_scaler_in = Scaler_onnx("scaler_input.onnx", ice_folder)
    ice_scaler_out = Scaler_onnx("scaler_output.onnx", ice_folder)
    ice_scaler_inv_out = Scaler_onnx("scaler_inverse_output.onnx", ice_folder)

    try:
        ice_model = LSTM_onnx("ICE_onnx.onnx", ice_folder, tf_model_name)
    except Exception as e:
        print(f"Error loading ICE model: {e}")
        return

    # Load PG Components
    pg_scaler_in = Scaler_onnx("scaler_input.onnx", pg_folder)
    pg_scaler_out = Scaler_onnx("scaler_output.onnx", pg_folder)
    pg_scaler_inv_out = Scaler_onnx("scaler_inverse_output.onnx", pg_folder)

    try:
        pg_model = LSTM_onnx("PG_onnx.onnx", pg_folder, tf_model_name)
    except Exception as e:
        print(f"Error loading PG model: {e}")
        return

    # Reset states (handled by class, but good practice to call)
    ice_model.reset_states()
    pg_model.reset_states()

    # Lists to store results
    ice_predictions = []
    pg_predictions = []

    # Initial auxiliary inputs
    # Match Legacy Simulation: Initialize to Zeros in SCALED domain directly
    # Legacy: ice_aux = np.zeros((1, 1, 5), dtype=np.float32)
    # ice_aux = np.zeros((1, 1, 5), dtype=np.float32)
    # ice_aux_scaled = ice_scaler_out.transform(ice_aux)
    ice_aux = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    ice_aux_scaled = ice_scaler_out.transform(ice_aux)[0].reshape(1, 1, 5)

    # PG Aux:
    # Legacy: pg_aux = np.zeros((1, 1, 2), dtype=np.float32)
    # Define ground truth columns if available in DF
    # Based on notebook inspection, columns are present in df_full.
    # The columns are: 'ICE_Torque_Nm', 'Car_Speed_kmph' (or similar?), 'SOC_1'
    # Wait, the user specifically mentioned comparing to "true ones from WLTC.csv".
    # Let's inspect df_full columns again conceptually based on previous head output.
    # The head output didn't show full header, but `rl_loop_simulation.py` (our reference) uses:
    # "ICE_Torque_Nm", "Car_Speed_kmph", "SOC_1"
    # I will assume these column names exist in the cleaned df_full.

    # Extract ground truth arrays before looping
    # Note: df is a subset with reordered columns input_cols.
    # We need to access the full dataframe for potential target columns if they are not in input_cols.
    # input_cols includes "ICE_Torque_Nm" but not "Car_Speed_kmph" or "SOC_1" based on the list in main().

    # Check if target columns exist in df_full
    true_torque = (
        df_full["ICE_Torque_Nm"].values
        if "ICE_Torque_Nm" in df_full.columns
        else np.zeros(n_steps)
    )
    true_speed = (
        df_full["Car_Speed_kmph"].values
        if "Car_Speed_kmph" in df_full.columns
        else np.zeros(n_steps)
    )
    # Note: Column might be 'SOC_1' or similar. `rl_loop_simulation.py` uses 'SOC_1'.
    true_soc = (
        df_full["SOC_1"].values if "SOC_1" in df_full.columns else np.zeros(n_steps)
    )
    time_steps = (
        df_full["Time_s"].values
        if "Time_s" in df_full.columns
        else np.arange(n_steps) * 0.5
    )  # Default 0.5s step

    # pg_aux = np.zeros((1, 1, 2), dtype=np.float32)
    # pg_aux_scaled = pg_scaler_out.transform(pg_aux)
    pg_aux = np.array([[0.0, 0.7]], dtype=np.float32)
    pg_aux_scaled = pg_scaler_out.transform(pg_aux)[0].reshape(1, 1, 2)

    print("Starting simulation loop...")
    start_time = time.time()
    n_steps = len(df)

    for i in range(n_steps):
        if i % 100 == 0:
            print(f"Step {i}/{n_steps}", end="\r")

        # --- Legacy Stateless Mode ---
        if SIMULATE_LEGACY_STATELESS:
            ice_model.reset_states()
            pg_model.reset_states()

        # --- ICE Step ---
        x_ice = ICEinput.iloc[i].values.reshape(1, 4).astype(np.float32)
        x_ice_scaled = ice_scaler_in.transform(x_ice)[0].reshape(1, 1, 4)

        # Predict
        # Input format: [x_scaled, aux_scaled]
        # Returns list containing output tensor: [tensor(1, 1, 5)]
        ice_pred_scaled_list = ice_model([x_ice_scaled, ice_aux_scaled])

        # Extract tensor
        ice_pred_scaled = ice_pred_scaled_list[0]  # (1, 1, 5)

        # Inverse encode for storage
        ice_pred_flat = ice_pred_scaled.reshape(1, 5)
        # Scaler expects array
        ice_pred = ice_scaler_inv_out.transform(ice_pred_flat)[0]  # [5]
        # ice_pred is shape (1, 5). We want to store the flat vector (5,)
        ice_predictions.append(ice_pred[0])

        # Update feedback for next step
        # Pass the SCALED output tensor directly as next step's aux input
        ice_aux_scaled = ice_pred_scaled

        # Get predicted torque for PG input (index 0)
        # ice_pred is (1, 5), so torque is at [0, 0]
        pred_torque = ice_pred[0, 0]

        # --- PG Step ---
        x_pg = PGinput.iloc[i].values.copy()
        x_pg[2] = pred_torque
        x_pg = x_pg.reshape(1, 4).astype(np.float32)

        x_pg_scaled = pg_scaler_in.transform(x_pg)[0].reshape(1, 1, 4)

        # Predict
        pg_pred_scaled_list = pg_model([x_pg_scaled, pg_aux_scaled])
        pg_pred_scaled = pg_pred_scaled_list[0]  # (1, 1, 2)

        # Inverse
        pg_pred_flat = pg_pred_scaled.reshape(1, 2)
        pg_pred = pg_scaler_inv_out.transform(pg_pred_flat)[0]  # [2]

        pg_predictions.append(pg_pred[0])

        # Update feedback
        pg_aux_scaled = pg_pred_scaled

    print(f"\nSimulation complete in {time.time() - start_time:.2f} seconds")

    # --- Plotting ---
    print("Generating plots...")
    ice_results = np.array(ice_predictions)
    pg_results = np.array(pg_predictions)

    # Plot ICE Torque
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, true_torque, label="True Torque (Nm)", alpha=0.6)
    plt.plot(time_steps, ice_results[:, 0], label="Predicted Torque (Nm)", alpha=0.8)
    plt.title("ICE Torque (ONNX vs True)")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "ice_torque.png"))
    plt.close()

    # Plot Car Speed
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, true_speed, label="True Speed (km/h)", alpha=0.6)
    plt.plot(
        time_steps,
        pg_results[:, 0],
        label="Predicted Speed (km/h)",
        color="orange",
        alpha=0.8,
    )
    plt.title("Car Speed (ONNX vs True)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (km/h)")
    plt.yticks(np.arange(-40.0, 140.0, step=20.0))  # To align with the other plots
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "car_speed.png"))
    plt.close()

    # Plot SOC
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, true_soc, label="True SOC", alpha=0.6)
    plt.plot(
        time_steps, pg_results[:, 1], label="Predicted SOC", color="green", alpha=0.8
    )
    plt.title("State of Charge (ONNX vs True)")
    plt.xlabel("Time (s)")
    plt.ylabel("SOC")
    plt.yticks(np.arange(0, 0.9, step=0.2))  # To align with the other plots
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "soc.png"))
    plt.close()

    # --- Save Results to CSV ---
    # Create DataFrame
    results_df = pd.DataFrame(
        {
            "time": time_steps,
            "ice_torque_pred": ice_results[:, 0],
            "ice_torque_true": true_torque,
            "car_speed_pred": pg_results[:, 0],
            "car_speed_true": true_speed,
            "soc_pred": pg_results[:, 1],
            "soc_true": true_soc,
        }
    )

    csv_output_path = os.path.join(OUTPUT_DIR, "simulation_results.csv")
    results_df.to_csv(csv_output_path, index=False)
    print(f"Results saved to {csv_output_path}")

    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
