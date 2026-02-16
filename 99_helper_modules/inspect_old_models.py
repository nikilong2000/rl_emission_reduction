import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, model_from_config
import h5py
import json

# Suppress Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_old_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        import tensorflow_addons as tfa

        custom_objects = {"LayerNormLSTMCell": tfa.rnn.LayerNormLSTMCell}
    except ImportError:
        print("Tensorflow Addons not found, using empty custom_objects.")
        custom_objects = {}

    try:
        model = load_model(model_path, compile=False, custom_objects=custom_objects)
        return model
    except ValueError as ve:
        if "time_major" in str(ve):
            print("Detected 'time_major' issue. Patching config...")
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


def inspect_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths
    ice_path = os.path.join(
        base_dir, "controller_for_ICE_PG/src/models_markus/ICE_Model_Update_01/model.h5"
    )
    pg_path = os.path.join(
        base_dir,
        "controller_for_ICE_PG/src/models_markus/PG_Model_M1.1_without_EM1_Torque/model.h5",
    )

    print("\n--- Inspecting ICE Model ---")
    ice_model = load_old_model(ice_path)
    if ice_model:
        print("ICE Model Inputs:", ice_model.inputs)
        # Try to infer input names if possible, though mostly just shapes
        for i, inp in enumerate(ice_model.inputs):
            print(f"Input {i}: shape={inp.shape}, name={inp.name}, dtype={inp.dtype}")
        print("ICE Model Outputs:", ice_model.outputs)

    print("\n--- Inspecting PG Model ---")
    pg_model = load_old_model(pg_path)
    if pg_model:
        print("PG Model Inputs:", pg_model.inputs)
        for i, inp in enumerate(pg_model.inputs):
            print(f"Input {i}: shape={inp.shape}, name={inp.name}, dtype={inp.dtype}")
        print("PG Model Outputs:", pg_model.outputs)


if __name__ == "__main__":
    inspect_models()
