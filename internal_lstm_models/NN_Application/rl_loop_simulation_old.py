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
# Correct Order: ICE_Speed, ICE_Torque(predicted), EM2, Brake
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

    ice_aux = np.zeros((1, 1, 5), dtype=np.float32)
    # However, if aux represents the previous output, it should PROBABLY be scaled.
    # Zero scaled corresponds to min value in standard scaling or 0 in standard?
    # Usually in these recursive RNNs, we feed back the SCALED prediction.
    # Initial state 0.0 usually means mean value (if standardscaler) or min (if minmax).
    # Given we don't have initial values, standard procedure is Zeros.

    # PG Aux: 2 features
    pg_aux = np.zeros((1, 1, 2), dtype=np.float32)

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
        # --- ICE ---
        # Input features
        ice_in_vals = df.loc[t, ICE_INPUT_COLS].values.reshape(1, -1)
        ice_in_scaled = ice_in_scaler.transform(ice_in_vals).reshape(1, 1, -1)

        # Predict
        # Note: training=False is crucial for LayerNormLSTMCell sometimes
        ice_pred_scaled = ice_model([ice_in_scaled, ice_aux], training=False)

        # Update Aux for next step (feedback loop)
        ice_aux = ice_pred_scaled

        # Inverse transform for logging/downstream use
        ice_pred = ice_out_scaler.inverse_transform(ice_pred_scaled.numpy()[0])
        ice_torque_pred = ice_pred[0][0]  # Assuming Torque is first index

        # --- PG ---
        ice_speed = df.loc[t, "ICE_Speed_rpm"]
        em2_torque = df.loc[t, "EM2_Torque_Nm"]
        brake_perc = df.loc[t, "Brake_perc"]

        # Construct Input: Speed, ICE_Torque_Pred, EM2, Brake
        pg_in_vals = np.array([[ice_speed, ice_torque_pred, em2_torque, brake_perc]])
        pg_in_scaled = pg_in_scaler.transform(pg_in_vals).reshape(1, 1, -1)

        # Predict
        pg_pred_scaled = pg_model([pg_in_scaled, pg_aux], training=False)

        # Update Aux
        pg_aux = pg_pred_scaled

        # Unscale
        pg_pred = pg_out_scaler.inverse_transform(pg_pred_scaled.numpy()[0])
        car_speed_pred = pg_pred[0][0]
        soc_pred = pg_pred[0][1]

        # --- Log ---
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
