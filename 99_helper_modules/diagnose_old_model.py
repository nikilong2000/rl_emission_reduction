import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Suppress Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def test_statefulness():
    print("--- DIAGNOSIS STARTED ---")

    # Path
    model_dir = "controller_for_ICE_PG/src/models_markus/ICE_Model_Update_01"
    model_path = os.path.join(model_dir, "model.h5")

    # Load Model
    print(f"Loading model from {model_path}...")
    try:
        import tensorflow_addons as tfa

        custom_objects = {"LayerNormLSTMCell": tfa.rnn.LayerNormLSTMCell}
    except ImportError:
        print("Tensorflow Addons not found, trying generic load.")
        custom_objects = {}

    try:
        model = load_model(model_path, compile=False, custom_objects=custom_objects)
    except ValueError as ve:
        if "time_major" in str(ve):
            print(
                "Detected 'time_major' compatibility issue. Attempting manual config patch..."
            )
            import h5py

            # Keras 3 / Legacy split adjustment
            try:
                from tensorflow.keras.models import model_from_config
            except ImportError:
                from keras.src.models.model import model_from_config
            # Fallback for newer keras
            if "model_from_config" not in locals():
                from tensorflow.keras.models import model_from_json

                # We can't use model_from_config easily in Keras 3 if not exposed.
                # Let's try deserialization.
                from tensorflow.keras.saving import deserialize_keras_object

            import json

            with h5py.File(model_path, "r") as f:
                config_str = f.attrs.get("model_config")
                if isinstance(config_str, bytes):
                    config_str = config_str.decode("utf-8")
                model_config = json.loads(config_str)

            # Recursive cleaner function
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
                print("Model loaded successfully via Config Patch.")
            except Exception as e2:
                print(f"Patch failed: {e2}")
                return
        else:
            print(f"Error loading model: {ve}")
            return
    except Exception as e:
        print(f"Error loading old model: {e}")
        return

    # Check Layers
    print("\nLayer Configs:")
    lstm_layers = []
    for layer in model.layers:
        print(f"- {layer.name}: {layer.__class__.__name__}")
        if hasattr(layer, "stateful"):
            print(f"  Stateful: {layer.stateful}")
            if layer.stateful:
                lstm_layers.append(layer)

    if not lstm_layers:
        print("\nWARNING: No stateful layers found!")
    else:
        print(f"\nFound {len(lstm_layers)} stateful layers.")

    # Run Inference Test
    print("\n--- Running Stateful Inference Test (10 steps, constant input) ---")

    # Create Dummy Input
    # ICE Model: 4 features
    x_val = np.random.rand(1, 1, 4).astype(np.float32)

    # Aux Input: 5 features (Torque, Emissions)
    aux_val = np.zeros((1, 1, 5), dtype=np.float32)

    # Run loop
    preds = []

    # Reset first
    model.reset_states()

    for i in range(10):
        # Using the same call signature as the script
        # Note: training=False
        p = model([x_val, aux_val], training=False)
        preds.append(p.numpy().flatten()[0])  # Track first output (Torque)

    print("\nPredictions:")
    print(preds)

    # Check Variance
    variance = np.var(preds)
    print(f"\nVariance: {variance}")

    if variance < 1e-9:
        print("RESULT: MODEL IS STATELESS (Outputs are identical)")
    else:
        print("RESULT: MODEL IS STATEFUL (Outputs change)")

    # Check Scalers
    print("\n--- Checking Scalers ---")
    try:
        input_scaler = joblib.load(os.path.join(model_dir, "input_scaler.lib"))
        print(f"Input Scaler: Min {input_scaler.min_}, Scale {input_scaler.scale_}")
    except Exception as e:
        print(f"Scaler Check Failed: {e}")


if __name__ == "__main__":
    test_statefulness()
