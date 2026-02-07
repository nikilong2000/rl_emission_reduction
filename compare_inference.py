import os
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
import warnings

# Suppress Warnings for Cleaner Output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Allow unsafe deserialization (Required for some custom objects)
tf.keras.config.enable_unsafe_deserialization()

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


# --- OLD MODEL WRAPPER (LEGACY SUPPORT ATTEMPT) ---
# Note: These models were trained with TensorFlow < 2.10 and likely use
# 'tensorflow_addons.rnn.LayerNormLSTMCell' or inconsistent Keras 2 serialization.
# Loading them in stock TF 2.18 is highly experimental and likely to fail.


class CustomLSTM(LSTM):
    """Attempt to patch 'time_major' argument mismatch in Keras 3."""

    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)
        self.input_spec = None


class OldModelWrapper:
    def __init__(self, model_dir, model_type="ICE"):
        self.model_type = model_type
        self.model_path = os.path.join(model_dir, "model.h5")
        print(f"Loading Old {model_type} Model from {self.model_path}")

        # Attempt to load with custom LSTM and compile=False
        try:
            self.model = load_model(
                self.model_path, compile=False, custom_objects={"LSTM": CustomLSTM}
            )
        except Exception as e:
            raise RuntimeError(f"Keras Version Incompatibility: {e}")

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

    @tf.function(jit_compile=False)
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


# --- NEW MODEL WRAPPER ---


class NewModelWrapper:
    def __init__(self, model_dir, model_name="ICE"):
        self.model_name = model_name
        print(f"Loading New {model_name} Model from {model_dir}")
        self.model_main = load_model(
            os.path.join(model_dir, "model_inf.keras"), compile=False
        )
        self.model_init = load_model(
            os.path.join(model_dir, "model_init.keras"), compile=False
        )

        self.input_scaler = joblib.load(os.path.join(model_dir, "input_scaler.lib"))
        self.output_scaler = joblib.load(os.path.join(model_dir, "output_scaler.lib"))

        # Dynamic shape handling for Scalers (Important if model inputs differ from scaler inputs)
        # Note: We trust the scaler for transform, but need to be careful if model input requires subset
        # For this benchmark, we assume standard usage.

        # Setup compiled interactions
        @tf.function(jit_compile=False)
        def predict_main(input_tensor):
            return self.model_main(input_tensor)

        @tf.function(jit_compile=False)
        def predict_init(input_tensor):
            return self.model_init(input_tensor)

        self.predict_main_func = predict_main
        self.predict_init_func = predict_init

    def reset_states(self):
        """Manual state reset for Functional Keras 3 models"""
        for layer in self.model_main.layers:
            if hasattr(layer, "reset_states"):
                layer.reset_states()

    def initialize(self, aux_values):
        """Run init model to prime states."""
        # Ensure aux corresponds to output scaler dimensions
        if aux_values.shape[1] != self.output_scaler.n_features_in_:
            # Pad or slice if necessary? The script assumes we pass correct size.
            # For robustness, we recreate aux based of scaler expectations.
            temp_aux = np.zeros((1, self.output_scaler.n_features_in_))
            # Fill first N cols
            cols = min(aux_values.shape[1], temp_aux.shape[1])
            temp_aux[:, :cols] = aux_values[:, :cols]
            aux_values = temp_aux

        aux_scaled = self.output_scaler.transform(aux_values).reshape((1, 1, -1))
        _ = self.predict_init_func(aux_scaled)
        # Note: Actual state setting is implicit in stateful=True Functional models
        # IF we pass the state. But here separate init model is used.
        # Just running it ensures TF graph tracing, but doesn't set main model weights
        # unless 'set_states' logic is applied.
        # For INFERENCE BENCHMARKING: The overhead of running predict_init is captured.

    def step(self, input_val):
        x_scaled = self.input_scaler.transform(input_val).reshape((1, 1, -1))
        out = self.predict_main_func(x_scaled)
        return self.output_scaler.inverse_transform(out.numpy()[0])


# --- MAIN ---


def main():
    print("--- BENCHMARK STARTED ---")
    OLD_ICE_DIR = "controller_for_ICE_PG/src/models_markus/ICE_Model_Update_01"
    OLD_PG_DIR = (
        "controller_for_ICE_PG/src/models_markus/PG_Model_M1.1_without_EM1_Torque"
    )
    NEW_ICE_DIR = "internal_lstm_models/NN_Application/Nets/ICE"
    NEW_DRV_DIR = "internal_lstm_models/NN_Application/Nets/Drivetrain"
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
    try:
        # Wrap in specific try/except to avoid crashing the whole script
        ice_old = OldModelWrapper(OLD_ICE_DIR, "ICE")
        drv_old = OldModelWrapper(OLD_PG_DIR, "Drivetrain")

        # Warmup
        ice_old.predict_step(data[0:1])

        with Timer() as t_old:
            for i in range(len(data)):
                _ = ice_old.predict_step(data[i : i + 1])
                # Mock drv input
                _ = drv_old.predict_step(np.zeros((1, 4), dtype=np.float32))

        print(
            f"Old Models Total Time: {t_old.duration:.4f}s ({t_old.duration/len(data)*1000:.4f} ms/step)"
        )

    except Exception as e:
        print(f"SKIPPED OLD MODELS: {e}")
        print(
            "Reason: Legacy Keras 2 .h5 models with custom layers (LayerNormLSTMCell) are incompatible with Keras 3."
        )

    # 3. BENCHMARK NEW MODELS
    print("\n--- Benchmarking NEW Models ---")
    try:
        new_ice = NewModelWrapper(NEW_ICE_DIR, "ICE")
        new_drv = NewModelWrapper(NEW_DRV_DIR, "Drivetrain")

        # Init with zeros (just for timing)
        aux_ice = np.zeros((1, new_ice.output_scaler.n_features_in_))
        aux_drv = np.zeros((1, new_drv.output_scaler.n_features_in_))

        new_ice.initialize(aux_ice)
        new_drv.initialize(aux_drv)

        new_ice.reset_states()
        new_drv.reset_states()

        # Warmup
        new_ice.step(data[0:1])

        with Timer() as t_new:
            for i in range(len(data)):
                _ = new_ice.step(data[i : i + 1])

                # Create dummy input for Drivetrain (4 features)
                dummy_drv_in = np.array([[data[i][0], 100.0, 50.0, 0.0]])
                _ = new_drv.step(dummy_drv_in)

        print(
            f"New Models (ICE + Drivetrain) Total Time: {t_new.duration:.4f}s ({t_new.duration/len(data)*1000:.4f} ms/step)"
        )

    except Exception as e:
        print(f"FAILED NEW MODELS: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
