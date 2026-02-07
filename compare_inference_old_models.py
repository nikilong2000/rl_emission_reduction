import os
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings

# Suppress Warnings for Cleaner Output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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


# --- OLD MODEL WRAPPER ---
# Tailored for Python 3.10 and TensorFlow 2.15


class OldModelWrapper:
    def __init__(self, model_dir, model_type="ICE"):
        self.model_type = model_type
        self.model_path = os.path.join(model_dir, "model.h5")
        print(f"Loading Old {model_type} Model from {self.model_path}")

        # Try loading standard first, then try with tensorflow_addons if needed
        # In TF 2.15, standard load_model should work for Keras 2 models
        # provided custom layers (like LayerNormLSTMCell) are available.
        custom_objects = {}

        try:
            import tensorflow_addons as tfa

            custom_objects["LayerNormLSTMCell"] = tfa.rnn.LayerNormLSTMCell
        except ImportError:
            pass  # tfa not available, try loading without specific custom objects

        try:
            self.model = load_model(
                self.model_path, compile=False, custom_objects=custom_objects
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print(
                "Note: If the model uses 'LayerNormLSTMCell', ensure 'tensorflow-addons' is installed."
            )
            raise e

        # Load scalers
        self.input_scaler = joblib.load(os.path.join(model_dir, "input_scaler.lib"))
        self.output_scaler = joblib.load(os.path.join(model_dir, "output_scaler.lib"))

        # Determine Aux dims
        self.aux_dims = 5 if model_type == "ICE" else 2

        # Pre-calc constants
        self._in_scale = tf.constant(self.input_scaler.scale_, dtype=tf.float32)
        self._in_min = tf.constant(self.input_scaler.min_, dtype=tf.float32)
        self._out_scale = tf.constant(self.output_scaler.scale_, dtype=tf.float32)
        self._out_min = tf.constant(self.output_scaler.min_, dtype=tf.float32)

        # Constant Aux (simplified for benchmark)
        initial_aux = np.zeros((1, self.aux_dims))
        if model_type != "ICE":
            initial_aux[0, 1] = 0.7  # SOC

        aux_scaled = self.output_scaler.transform(initial_aux).reshape(
            (1, 1, self.aux_dims)
        )
        self.aux = tf.constant(aux_scaled, dtype=tf.float32)

    @tf.function
    def predict_step(self, inputs):
        # Scale
        x_scaled = (inputs * self._in_scale) + self._in_min
        x_scaled = tf.reshape(x_scaled, (1, 1, -1))

        # Predict: Old model signature is called with list [x, aux]
        y_scaled = self.model([x_scaled, self.aux], training=False)
        y_scaled = tf.reshape(y_scaled, (-1,))

        # Descale
        y = (y_scaled - self._out_min) / self._out_scale
        return y


# --- MAIN ---


def main():
    print("--- BENCHMARK STARTED (OLD MODELS ONLY) ---")
    OLD_ICE_DIR = "controller_for_ICE_PG/src/models_markus/ICE_Model_Update_01"
    OLD_PG_DIR = (
        "controller_for_ICE_PG/src/models_markus/PG_Model_M1.1_without_EM1_Torque"
    )
    CSV_PATH = "internal_lstm_models/Test_Cycles/WLTC.csv"

    # 1. LOAD DATA
    print(f"Loading data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH, sep=";", decimal=",")
        data = df[["ICE_Speed_rpm", "fuel_mg", "T_amb_K", "p_amb_bar"]].values.astype(
            np.float32
        )
        print(f"Data loaded: {len(data)} steps.")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # 2. BENCHMARK OLD MODELS
    print("\n--- Benchmarking OLD Models ---")

    # Store results
    ice_results = []
    drv_results = []

    try:
        ice_old = OldModelWrapper(OLD_ICE_DIR, "ICE")
        drv_old = OldModelWrapper(OLD_PG_DIR, "Drivetrain")

        # Warmup
        ice_old.predict_step(data[0:1])

        with Timer() as t_old:
            for i in range(len(data)):
                # 1. Prediction ICE
                ice_out_tensor = ice_old.predict_step(data[i : i + 1])
                ice_out = ice_out_tensor.numpy()
                ice_results.append(ice_out)

                # 2. Extract Torque (Index 0) for Drivetrain Input
                torque = ice_out[0]

                # 3. Prediction Drivetrain
                # Inputs: [Speed_rpm, Torque_Nm, EM2_Torque_Nm, Brake_perc]
                # Speed from Cycle data, Torque from ICE, EM2/Brake fixed dummies (50, 0)
                drv_input = np.array(
                    [[data[i][0], torque, 50.0, 0.0]], dtype=np.float32
                )
                drv_out_tensor = drv_old.predict_step(drv_input)
                drv_results.append(drv_out_tensor.numpy())

        print(
            f"Old Models Total Time: {t_old.duration:.4f}s ({t_old.duration/len(data)*1000:.4f} ms/step)"
        )

        # Save Results
        print("Saving results to 'old_model_predictions.csv'...")
        ice_np = np.array(ice_results)
        drv_np = np.array(drv_results)

        # Construct DataFrame
        # ICE Columns (Generic or Try to use scaler names if available, but for old joblib they might not be)
        ice_cols = [f"ICE_Out_{j}" for j in range(ice_np.shape[1])]
        drv_cols = [f"DRV_Out_{j}" for j in range(drv_np.shape[1])]

        df_res = pd.DataFrame(np.hstack([ice_np, drv_np]), columns=ice_cols + drv_cols)
        df_res.to_csv("old_model_predictions.csv", index=False)
        print("Results saved.")

    except Exception as e:
        print(f"FAILED TO RUN OLD MODELS: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
