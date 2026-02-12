import os
import sys
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import joblib
import h5py
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, model_from_config

# Suppress Version Warnings
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICE_MODEL_DIR = os.path.join(
    BASE_DIR, "../../controller_for_ICE_PG/src/models_markus/ICE_Model_Update_01"
)
PG_MODEL_DIR = os.path.join(
    BASE_DIR,
    "../../controller_for_ICE_PG/src/models_markus/PG_Model_M1.1_without_EM1_Torque",
)
INPUT_DATA_PATH = os.path.join(BASE_DIR, "Input_data", "WLTC.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output", "RL_Loop_Simulation_Old")

# Feature Definitions
# ICE: 4 inputs -> 5 outputs (Torque + 4 Emissions)
ICE_INPUT_COLS = ["ICE_Speed_rpm", "fuel_mg", "T_amb_K", "p_amb_bar"]
# PG: 4 inputs -> 2 outputs (Speed, SOC)
# Correct Order: ICE_Speed, ICE_Torque(predicted), EM2,# Drivetrain: 0, 0.7
PG_INITIAL_OUTPUTS = [0.0, 0.0]


def scale_minmax(x, scale, min_):
    """x_scaled = x * scale + min"""
    return x * scale + min_


def descale_minmax(x_scaled, scale, min_):
    """x = (x_scaled - min) / scale"""
    return (x_scaled - min_) / scale


def predict_ice(
    ice_model,
    ice_in_scaler,
    ice_out_scaler,
    Speed_rpm,
    m_fuel_mg,
    T_amb_K,
    p_amb_bar,
    ice_aux,
):
    """
    Predicts ICE outputs for one step using standard scaling logic.
    Inputs are scalars or arrays of shape (1,).
    ice_aux is the current state (1, 1, 5).
    """
    # 1. Construct Input Vector: [Speed, Fuel, T, P]
    # Ensure inputs are 2D for scaler (1, n_features)
    x = np.array([[Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar]], dtype=np.float32)

    # 2. Scale Input
    x_scaled = scale_minmax(x, ice_in_scaler.scale_, ice_in_scaler.min_)
    x_scaled = x_scaled.reshape(1, 1, 4)  # (batch, time, features)

    # 3. Model Inference
    # Model expects [input_sequence, initial_state]
    # Returns scaled output tensor (1, 1, 5)
    y_scaled = ice_model([x_scaled, ice_aux])

    # 4. Update Aux for next step (feedback loop)
    # The model output becomes the aux input for the next step directly
    ice_aux_new = y_scaled

    # 5. Inverse Scale Output
    # Scaler expects 2D array (n_samples, n_features)
    y_flat = y_scaled.numpy().reshape(1, 5)
    y = descale_minmax(y_flat, ice_out_scaler.scale_, ice_out_scaler.min_)

    # Unpack outputs: Torque, NO, NO2, CO, CO2
    Torque_Nm = y[0, 0]
    NO_out = y[0, 1]
    NO2_out = y[0, 2]
    CO_out = y[0, 3]
    CO2_out = y[0, 4]

    return Torque_Nm, NO_out, NO2_out, CO_out, CO2_out, ice_aux_new


def predict_pg(
    pg_model,
    pg_in_scaler,
    pg_out_scaler,
    ICE_Speed_soll_rpm,
    EM2_Torque_Nm,
    ICE_Torque_Nm,
    Brake_perc,
    pg_aux,
):
    """
    Predicts PG outputs for one step using standard scaling logic.
    Inputs are scalars or arrays of shape (1,).
    pg_aux is the current state (1, 1, 2).
    """
    # 1. Construct Input Vector: [Speed, EM2, ICE_Torque, Brake]
    # ORDER MATTERS: Based on transition_function_model.py
    x = np.array(
        [[ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc]],
        dtype=np.float32,
    )

    # 2. Scale Input
    x_scaled = scale_minmax(x, pg_in_scaler.scale_, pg_in_scaler.min_)
    x_scaled = x_scaled.reshape(1, 1, 4)

    # 3. Model Inference
    y_scaled = pg_model([x_scaled, pg_aux])

    # 4. Update Aux
    pg_aux_new = y_scaled

    # 5. Inverse Scale Output
    y_flat = y_scaled.numpy().reshape(1, 2)
    y = descale_minmax(y_flat, pg_out_scaler.scale_, pg_out_scaler.min_)

    # Unpack: Speed, SOC
    Car_Speed_kmph = y[0, 0]
    SOC_1 = y[0, 1]

    return Car_Speed_kmph, SOC_1, pg_aux_new


PG_CSV_INPUTS = ["ICE_Speed_rpm", "EM2_Torque_Nm", "Brake_perc"]


def load_old_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        import tensorflow_addons as tfa

        custom_objects = {"LayerNormLSTMCell": tfa.rnn.LayerNormLSTMCell}
    except ImportError:
        print(
            "Tensorflow Addons not found. Trying without custom objects (might fail if LayerNorm used)."
        )
        custom_objects = {}

    try:
        model = load_model(model_path, compile=False, custom_objects=custom_objects)
        return model
    except ValueError as ve:
        if "time_major" in str(ve):
            print("Detected 'time_major' issue. Attempting config patch...")
            with h5py.File(model_path, "r") as f:
                config_str = f.attrs.get("model_config")
                if isinstance(config_str, bytes):
                    config_str = config_str.decode("utf-8")
                model_config = json.loads(config_str)

            def clean_config(cfg):
                if isinstance(cfg, dict):
                    if "time_major" in cfg:
                        del cfg["time_major"]
                    for key, value in cfg.items():
                        clean_config(value)
                elif isinstance(cfg, list):
                    for item in cfg:
                        clean_config(item)

            clean_config(model_config)

            try:
                model = model_from_config(model_config, custom_objects=custom_objects)
                model.load_weights(model_path)
                print("Model loaded via config patch.")
                return model
            except Exception as e:
                print(f"Patch failed: {e}")
                return None
        else:
            print(f"Error loading model: {ve}")
            return None
    except Exception as e:
        print(f"General error loading model: {e}")
        return None


def load_scaler(directory, name):
    path = os.path.join(directory, f"{name}.lib")
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error loading scaler {path}: {e}")
        return None


def main():
    print("Starting Old Model RL Loop Simulation...")

    # Check Environment
    print(f"Python: {sys.version}")
    print(f"TensorFlow: {tf.__version__}")

    # 1. Load Models & Scalers
    ice_model = load_old_model(os.path.join(ICE_MODEL_DIR, "model.h5"))
    pg_model = load_old_model(os.path.join(PG_MODEL_DIR, "model.h5"))

    if not ice_model or not pg_model:
        print("Failed to load models.")
        return

    ice_in_scaler = load_scaler(ICE_MODEL_DIR, "input_scaler")
    ice_out_scaler = load_scaler(ICE_MODEL_DIR, "output_scaler")
    pg_in_scaler = load_scaler(PG_MODEL_DIR, "input_scaler")
    pg_out_scaler = load_scaler(PG_MODEL_DIR, "output_scaler")

    if not all([ice_in_scaler, ice_out_scaler, pg_in_scaler, pg_out_scaler]):
        print("Failed to load scalers.")
        return

    # 2. Load Data
    try:
        df = pd.read_csv(INPUT_DATA_PATH, delimiter=";", encoding="latin1")
        if df.shape[1] <= 1:
            df = pd.read_csv(INPUT_DATA_PATH, delimiter=",", encoding="latin1")
        df.columns = [col.strip() for col in df.columns]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Initialize States (Aux Inputs)
    # Shape logic:
    # Models seem to be TimeDistributed wrappers or plain LSTMs processing sequences?
    # Based on diagnosis script `test_statefulness`:
    # Input x: (1, 1, features)
    # Input aux: (1, 1, output_features)

    # Dimensions from scaler inspection:
    # ICE Output: 5 features
    # PG Output: 2 features

    # ice_aux = np.zeros((1, 1, 5), dtype=np.float32)
    # 1. ICE Aux: 5 features (Torque, Emissions...)
    # Create array of physical zeros
    ice_aux_initial = np.zeros((1, 5), dtype=np.float32)
    # Transform and reshape
    ice_aux = ice_out_scaler.transform(ice_aux_initial).reshape(1, 1, 5)

    # However, if aux represents the previous output, it should PROBABLY be scaled.
    # Zero scaled corresponds to min value in standard scaling or 0 in standard?
    # Usually in these recursive RNNs, we feed back the SCALED prediction.
    # Initial state 0.0 usually means mean value (if standardscaler) or min (if minmax).
    # Given we don't have initial values, standard procedure is Zeros.

    # PG Aux: 2 features
    # pg_aux = np.zeros((1, 1, 2), dtype=np.float32)
    # 2. PG Aux: 2 features (Speed, SOC)
    # Create array of physical zeros (or specific initial SOC if needed, but 0 seems standard for aux placeholder)
    # pg_aux_initial = np.zeros((1, 2), dtype=np.float32)
    pg_aux_initial = np.array([[0.0, 0.7]], dtype=np.float32)
    # Transform and reshape
    pg_aux = pg_out_scaler.transform(pg_aux_initial).reshape(1, 1, 2)

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

    # Reset internal states
    ice_model.reset_states()
    pg_model.reset_states()

    start_time = time.time()

    for t in range(num_steps):
        # --- ICE Prediction ---
        # Get inputs for current step
        ice_speed = df.loc[t, "ICE_Speed_rpm"]
        fuel_mg = df.loc[t, "fuel_mg"]
        T_amb = df.loc[t, "T_amb_K"]
        p_amb = df.loc[t, "p_amb_bar"]

        # Predict ICE
        ice_torque_pred, _, _, _, _, ice_aux_new = predict_ice(
            ice_model,
            ice_in_scaler,
            ice_out_scaler,
            ice_speed,
            fuel_mg,
            T_amb,
            p_amb,
            ice_aux,
        )

        # Update ICE State for next step
        ice_aux = ice_aux_new

        # --- PG Prediction ---
        # Get PG inputs
        em2_torque = df.loc[t, "EM2_Torque_Nm"]
        brake_perc = df.loc[t, "Brake_perc"]

        # Predict PG
        car_speed_pred, soc_pred, pg_aux_new = predict_pg(
            pg_model,
            pg_in_scaler,
            pg_out_scaler,
            ice_speed,  # ICE_Speed_soll_rpm (assuming actual speed is used as target/state)
            em2_torque,
            ice_torque_pred,
            brake_perc,
            pg_aux,
        )

        # Update PG State for next step
        pg_aux = pg_aux_new

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
    plt.plot(
        results["time"],
        results["ice_torque_pred"],
        label="Predicted (Old Model)",
        alpha=0.7,
    )
    plt.title("ICE Torque Prediction (Old Model)")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "ice_torque_old.png"))
    plt.close()

    # Plot Car Speed
    plt.figure(figsize=(12, 6))
    plt.plot(results["time"], results["car_speed_true"], label="True", alpha=0.7)
    plt.plot(
        results["time"],
        results["car_speed_pred"],
        label="Predicted (Old Model)",
        alpha=0.7,
    )
    plt.title("Car Speed Prediction (Old Model)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (km/h)")
    plt.yticks(np.arange(-40.0, 140.0, step=20.0))  # To align with the other plots
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "car_speed_old.png"))
    plt.close()

    # Plot SOC
    plt.figure(figsize=(12, 6))
    plt.plot(results["time"], results["soc_true"], label="True", alpha=0.7)
    plt.plot(
        results["time"], results["soc_pred"], label="Predicted (Old Model)", alpha=0.7
    )
    plt.title("SOC Prediction (Old Model)")
    plt.xlabel("Time (s)")
    plt.ylabel("SOC")
    plt.yticks(np.arange(0, 0.9, step=0.2))  # To align with the other plots
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "soc_old.png"))
    plt.close()

    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
