"""
ONNX Model Inference Comparison Script

This script benchmarks ONNX versions of the ICE and Drivetrain models on the WLTC test cycle.
It uses the ONNX models from controller_for_ICE_PG/SHARE/CTTC_models/ONNX/ and compares
inference performance with the same CSV input used by compare_inference_new_models.py and
compare_inference_old_models.py.

The ONNX models are stateful LSTM networks that require:
- Input/output scalers (ONNX format)
- Manual state management (h and c states for 3 LSTM layers)
- Proper initialization of hidden states

Performance metrics are printed to console, and predictions are saved to onnx_model_predictions.csv.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import warnings

# Suppress Warnings for Cleaner Output
warnings.filterwarnings("ignore")

# Add ONNX_Predict module to path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "controller_for_ICE_PG/SHARE/.venv_tf2.18_onnx/lib/python3.10/site-packages",
    ),
)

from ONNX_Predict.LSTM_onnx import LSTM_onnx
from ONNX_Predict.Scaler_onnx import Scaler_onnx


# --- UTILS ---
class Timer:
    def __init__(self):
        self.start = 0
        self.end = 0
        self.duration = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start


# --- ONNX MODEL WRAPPER ---
class ONNXModelWrapper:
    def __init__(self, model_dir, model_name, model_type="ICE"):
        self.model_type = model_type
        self.model_name = model_name
        print(f"Loading ONNX {model_type} Model from {model_dir}")

        # Load scalers
        self.input_scaler = Scaler_onnx("scaler_input.onnx", model_dir)
        self.output_scaler = Scaler_onnx("scaler_output.onnx", model_dir)
        self.output_scaler_inv = Scaler_onnx("scaler_inverse_output.onnx", model_dir)

        # Load model - no TF model needed
        self.model = LSTM_onnx(model_name, model_dir, tf_model_name=None)

        # Determine Aux dims based on model type
        self.aux_dims = 5 if model_type == "ICE" else 2

        # Initialize aux values
        initial_aux = np.zeros((1, self.aux_dims), dtype=np.float32)
        if model_type != "ICE":
            initial_aux[0, 1] = 0.7  # SOC for Drivetrain

        # Scale and reshape aux
        self.aux_scaled = self.output_scaler.transform(initial_aux)[0]
        self.aux_scaled = self.aux_scaled.reshape((1, 1, self.aux_dims))

        # Determine LSTM hidden size from model inputs
        # Find h_0_0 input to get the hidden size
        for inp in self.model.sess.get_inputs():
            if inp.name == "h_0_0":
                self.lstm_hidden_size = inp.shape[1]
                break

        # Initialize states manually (zeros for 3 LSTM layers)
        # Based on ONNX model: h_0_0, c_0_0, h_1_0, c_1_0, h_2_0, c_2_0
        self.states = [
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # h_0_0
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # c_0_0
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # h_1_0
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # c_1_0
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # h_2_0
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # c_2_0
        ]
        self.model.initial_states = True
        self.model.states = self.states

    def reset_states(self):
        """Reset model states"""
        self.states = [
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # h_0_0
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # c_0_0
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # h_1_0
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # c_1_0
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # h_2_0
            np.zeros((1, self.lstm_hidden_size), dtype=np.float32),  # c_2_0
        ]

    def step(self, input_val):
        """
        Perform one inference step.

        Args:
            input_val: numpy array of shape (1, n_features)

        Returns:
            Descaled output prediction
        """
        # Scale input
        x_scaled = self.input_scaler.transform(input_val)[0]
        x_scaled = x_scaled.reshape((1, 1, -1))

        # Predict (pass states manually to ONNX)
        input_dict = {
            "input_x": x_scaled,
            "input_y": self.aux_scaled,
            "h_0_0": self.states[0],
            "c_0_0": self.states[1],
            "h_1_0": self.states[2],
            "c_1_0": self.states[3],
            "h_2_0": self.states[4],
            "c_2_0": self.states[5],
        }

        # Run ONNX session directly
        pred = self.model.sess.run(self.model.outputs, input_dict)

        # Update states from output (indices 1-6)
        self.states = pred[1:]

        # Get prediction output (index 0)
        y_scaled = pred[0]

        # Descale output
        y_scaled_flat = y_scaled.reshape((1, -1))
        y = self.output_scaler_inv.transform(y_scaled_flat)[0]

        return y


# --- MAIN ---
def main():
    print("--- BENCHMARK STARTED (ONNX MODELS) ---")

    # Paths
    ONNX_ICE_DIR = "controller_for_ICE_PG/SHARE/CTTC_models/ONNX/ICE"
    ONNX_PG_DIR = "controller_for_ICE_PG/SHARE/CTTC_models/ONNX/PG"
    CSV_PATH = "internal_lstm_models/Test_Cycles/WLTC.csv"

    # 1. LOAD DATA
    print(f"Loading data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH, sep=";")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # Helper to find columns loosely (like in compare_inference_new_models.py)
    def get_col(name):
        if name in df.columns:
            return df[name].values
        for c in df.columns:
            if c.split("(")[0].strip() == name:
                return df[c].values
        raise KeyError(f"Column '{name}' not found in CSV")

    # Extract Inputs
    try:
        ice_speed = get_col("ICE_Speed_rpm")
        fuel = get_col("fuel_mg")
        t_amb = get_col("T_amb_K")
        p_amb = get_col("p_amb_bar")
        em2_torque = get_col("EM2_Torque_Nm")
        brake = get_col("Brake_perc")
    except KeyError as e:
        print(f"Error loading inputs: {e}")
        return

    # Stack data: [Speed, Fuel, T_amb, P_amb, EM2, Brake]
    data = np.column_stack([ice_speed, fuel, t_amb, p_amb, em2_torque, brake]).astype(
        np.float32
    )
    print(f"Data loaded: {len(data)} steps.")

    # Define Output Columns
    # ONNX models output 5 features for ICE (Torque, and 4 emissions)
    # and 2 features for Drivetrain (Car Speed, SOC)
    ICE_COLS = [
        "ICE_Torque_Nm",
        "NOx_gps",
        "CO_gps",
        "THC_gps",
        "CO2_gps",
    ]
    DRV_COLS = ["Car_Speed_kmph", "SOC_1"]

    # 2. BENCHMARK ONNX MODELS
    print("\n--- Benchmarking ONNX Models ---")

    # Store results
    ice_results = []
    drv_results = []

    try:
        ice_onnx = ONNXModelWrapper(ONNX_ICE_DIR, "ICE_onnx.onnx", "ICE")
        drv_onnx = ONNXModelWrapper(ONNX_PG_DIR, "PG_onnx.onnx", "Drivetrain")

        # Warmup
        ice_in_warmup = data[0, [0, 1, 2, 3]].reshape(1, -1)
        ice_onnx.step(ice_in_warmup)
        ice_onnx.reset_states()

        with Timer() as t_onnx:
            for i in range(len(data)):
                # 1. Prediction ICE
                # Inputs: [Speed, Fuel, T_amb, P_amb]
                ice_in = data[i, [0, 1, 2, 3]].reshape(1, -1)
                ice_out = ice_onnx.step(ice_in)
                ice_results.append(ice_out.flatten())

                # 2. Extract Torque (Index 0)
                torque = ice_out[0, 0] if ice_out.ndim > 1 else ice_out[0]

                # 3. Prediction Drivetrain
                # Inputs: [Speed, Torque, EM2, Brake]
                drv_input = np.array(
                    [[data[i, 0], torque, data[i, 4], data[i, 5]]], dtype=np.float32
                )
                drv_out = drv_onnx.step(drv_input)
                drv_results.append(drv_out.flatten())

        print(
            f"ONNX Models Total Time: {t_onnx.duration:.4f}s ({t_onnx.duration/len(data)*1000:.4f} ms/step)"
        )

        # Save Results
        print("Saving results to 'onnx_model_predictions.csv'...")
        ice_np = np.array(ice_results)
        drv_np = np.array(drv_results)

        # Construct DataFrame
        if ice_np.shape[1] == len(ICE_COLS):
            ice_cols = [f"ICE_{c}" for c in ICE_COLS]
        else:
            print(
                f"Warning: ICE output shape {ice_np.shape[1]} != expected {len(ICE_COLS)}"
            )
            ice_cols = [f"ICE_Out_{j}" for j in range(ice_np.shape[1])]

        if drv_np.shape[1] == len(DRV_COLS):
            drv_cols = [f"DRV_{c}" for c in DRV_COLS]
        else:
            print(
                f"Warning: Drv output shape {drv_np.shape[1]} != expected {len(DRV_COLS)}"
            )
            drv_cols = [f"DRV_Out_{j}" for j in range(drv_np.shape[1])]

        df_res = pd.DataFrame(np.hstack([ice_np, drv_np]), columns=ice_cols + drv_cols)
        df_res.to_csv(
            os.path.join(os.pardir, "results", "onnx_model_predictions.csv"),
            index=False,
        )
        print("Results saved.")

    except Exception as e:
        print(f"FAILED TO RUN ONNX MODELS: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
