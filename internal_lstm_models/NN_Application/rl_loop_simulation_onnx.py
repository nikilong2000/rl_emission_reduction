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
# We need absolute path for the library to work correctly if it uses relative paths internally,
# or we just pass the correct path.
# The notebook used: os.path.join("CTTC_models", "ONNX", "ICE") relative to where it was running.
# I will use absolute paths to be safe or relative to CWD.
BASE_MODEL_DIR = "controller_for_ICE_PG/SHARE/CTTC_models/ONNX"

# Import ONNX_Predict classes
# Since ONNX_Predict is installed in the venv, we can import it directly.
from ONNX_Predict.LSTM_onnx import LSTM_onnx
from ONNX_Predict.Scaler_onnx import Scaler_onnx


def main():
    print("Loading data...")
    # Load WLTC data
    csv_path = "internal_lstm_models/NN_Application/Input_data/WLTC.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Use the same column names as before
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

    df = pd.read_csv(csv_path, names=input_cols, skiprows=1)

    # Split into ICE and PG inputs
    ICEinput = df.iloc[:, :4].copy()
    PGinput = df.iloc[:, 4:].copy()

    # PG Input Preprocessing (from notebook)
    PGinput.loc[PGinput["ICE_Speed_soll_rpm"] < 900] = 0

    print("Loading models...")

    # Define folders
    ice_folder = os.path.join(BASE_MODEL_DIR, "ICE")
    pg_folder = os.path.join(BASE_MODEL_DIR, "PG")

    # The LSTM_onnx class requires a tf_model path as a 3rd argument (based on notebook),
    # but the notebook used "model.h5" which might be a dummy or required for some metadata?
    # Notebook line: ICE = LSTM_onnx("ICE_onnx.onnx", ICE_folder, tf_model)
    # where tf_model = "model.h5"
    # I should check if "model.h5" exists or if I can pass None or if I need to point to a dummy.
    # The notebook defined tf_model = "model.h5".
    # Inspecting list_dir of SHARE previously, didn't see model.h5 in root.
    # Maybe it expects it inside ICE_folder?
    # Let's assume for now we can pass a dummy request or if it's not used for inference.
    # The notebook comment says: "it is not necessary to import tensorflow or keras unless required..."
    # If LSTM_onnx uses it, I might need it.
    # Let's try passing "model.h5" as in the notebook, assuming it might be looked for or handled.
    # Or better yet, check if I really need it. The user said "simply do inference by calling the model instances".

    # The LSTM_onnx class initialization requires a path to the original TensorFlow model (.h5)
    # to determine initial states via 'get_initial_states'.
    # This path must be valid on the target system.
    # In the notebook, it is set to "model.h5".
    tf_model_path = "model.h5"

    # Check if model.h5 exists (warning only, as it might be in a different path on Linux)
    if not os.path.exists(tf_model_path) and not os.path.exists(
        os.path.join(ice_folder, tf_model_path)
    ):
        print(
            f"Warning: '{tf_model_path}' not found. LSTM_onnx might fail to initialize states if not found by the library."
        )

    # Load ICE Components
    # Note: Scaler_onnx(model_name, folder_path)
    ice_scaler_in = Scaler_onnx("scaler_input.onnx", ice_folder)
    ice_scaler_out = Scaler_onnx("scaler_output.onnx", ice_folder)
    ice_scaler_inv_out = Scaler_onnx("scaler_inverse_output.onnx", ice_folder)
    ice_model = LSTM_onnx("ICE_onnx.onnx", ice_folder, tf_model_path)

    # Load PG Components
    pg_scaler_in = Scaler_onnx("scaler_input.onnx", pg_folder)
    pg_scaler_out = Scaler_onnx("scaler_output.onnx", pg_folder)
    pg_scaler_inv_out = Scaler_onnx("scaler_inverse_output.onnx", pg_folder)
    pg_model = LSTM_onnx("PG_onnx.onnx", pg_folder, tf_model_path)

    # Reset states
    ice_model.reset_states()
    pg_model.reset_states()

    # Lists to store results
    ice_predictions = []
    pg_predictions = []

    # Initial auxiliary inputs
    # ICE Aux: Torque(0), NO(0), NO2(0), CO(0), CO2(0)
    ice_aux = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)  # Shape [1, 5]

    # PG Aux: Velocity(0), SOC(0.7)
    # Notebook: y_ini = np.array([[velocity_ini, SOC_ini]]...) where velocity_ini=0, SOC_ini=0.7
    pg_aux = np.array([[0, 0.7]], dtype=np.float32)  # Shape [1, 2]

    # Pre-scale initial feedbacks (as done in notebook loop setup?)
    # Notebook: y_scaled_ini = ICEscaler_out.transform(y_ini)[0].reshape((1, 1, 5))
    # We need to scale the aux input BEFORE passing it to the model loop?
    # In the notebook loop (manual calls):
    # y_predict_scaled = ICE([x_scaled, y_scaled_ini])
    # So yes, the model expects scaled aux.

    # Note: In my previous loop, I was scaling inside the loop.
    # For the INITIAL step, I need to scale the initial 0-values.
    # Subsequent steps use the output of the model (which IS scaled) directly as input for next step?
    # Wait.
    # Notebook output: y_predict_scaled = ICE(...)
    # Next step input: y_scaled_ini (which should be the previous output?)
    # The notebook manual calls REUSE y_scaled_ini because it was testing the SAME input repeatedly.
    # In a closed loop, the output of step T is the input of step T+1.
    # The model output is `y_predict_scaled`.
    # So for the next step, I should use `y_predict_scaled` as the aux input.
    # DO I need to inverse scale it for storage/other uses? Yes.
    # But for the *Model Input*, I can feed the scaled output directly back?
    # Let's verify input/output shapes.
    # Model Output: [1, 1, 5] (scaled)
    # Model Input Aux: [1, 1, 5] (scaled)
    # So yes, I can pass the result directly back.

    # However, for the very first step, I have real-world values (0, 0.7) which need scaling.
    ice_aux_scaled = ice_scaler_out.transform(ice_aux).reshape(1, 1, 5)
    pg_aux_scaled = pg_scaler_out.transform(pg_aux).reshape(1, 1, 2)

    print("Starting simulation loop...")
    start_time = time.time()
    n_steps = len(df)

    for i in range(n_steps):
        if i % 100 == 0:
            print(f"Step {i}/{n_steps}", end="\r")

        # --- ICE Step ---
        x_ice = ICEinput.iloc[i].values.reshape(1, 4).astype(np.float32)
        x_ice_scaled = ice_scaler_in.transform(x_ice).reshape(1, 1, 4)

        # Predict
        # Input format from notebook: ICE([x_scaled, y_scaled_ini])
        # Returns [1, 1, 5]
        ice_pred_scaled = ice_model([x_ice_scaled, ice_aux_scaled])

        # Inverse encode for storage/PG input
        # transform expects 2D? [1, 5]
        ice_pred_flat = ice_pred_scaled.reshape(1, 5)
        ice_pred = ice_scaler_inv_out.transform(ice_pred_flat)  # [1, 5]
        ice_predictions.append(ice_pred[0])

        # Update feedback for next step
        # Pass the SCALED output directly as next step's aux input
        ice_aux_scaled = ice_pred_scaled

        # Get predicted torque for PG input
        pred_torque = ice_pred[0, 0]

        # --- PG Step ---
        x_pg = PGinput.iloc[i].values.copy()
        x_pg[2] = pred_torque
        x_pg = x_pg.reshape(1, 4).astype(np.float32)

        x_pg_scaled = pg_scaler_in.transform(x_pg).reshape(1, 1, 4)

        # Predict
        pg_pred_scaled = pg_model([x_pg_scaled, pg_aux_scaled])

        # Inverse
        pg_pred_flat = pg_pred_scaled.reshape(1, 2)
        pg_pred = pg_scaler_inv_out.transform(pg_pred_flat)
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
    plt.plot(ice_results[:, 0], label="Predicted Torque (Nm)")
    plt.title("ICE Torque (ONNX)")
    plt.xlabel("Time Step")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "ice_torque.png"))
    plt.close()

    # Plot Car Speed
    plt.figure(figsize=(12, 6))
    plt.plot(pg_results[:, 0], label="Predicted Car Speed (km/h)", color="orange")
    plt.title("Car Speed (ONNX)")
    plt.xlabel("Time Step")
    plt.ylabel("Speed (km/h)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "car_speed.png"))
    plt.close()

    # Plot SOC
    plt.figure(figsize=(12, 6))
    plt.plot(pg_results[:, 1], label="SOC", color="green")
    plt.title("State of Charge (ONNX)")
    plt.xlabel("Time Step")
    plt.ylabel("SOC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "soc.png"))
    plt.close()

    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
