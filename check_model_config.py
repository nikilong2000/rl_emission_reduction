import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def check_old_model():
    print("--- OLD MODEL CHECK ---")
    model_dir = "controller_for_ICE_PG/src/models_markus/ICE_Model_Update_01"
    model_path = os.path.join(model_dir, "model.h5")

    if not os.path.exists(model_path):
        print(f"Old model not found at {model_path}")
        return

    try:
        import tensorflow_addons as tfa

        custom_objects = {"LayerNormLSTMCell": tfa.rnn.LayerNormLSTMCell}
    except ImportError:
        custom_objects = {}

    try:
        model = load_model(model_path, compile=False, custom_objects=custom_objects)
        print("Model loaded successfully.")

        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
            if hasattr(layer, "stateful"):
                print(f"  - stateful: {layer.stateful}")
            else:
                print(f"  - no stateful attr")

    except Exception as e:
        print(f"Error loading old model: {e}")


def check_new_model():
    print("\n--- NEW MODEL CHECK ---")
    model_dir = "internal_lstm_models/NN_Application/Nets/ICE"
    model_path = os.path.join(model_dir, "model_inf.keras")

    if not os.path.exists(model_path):
        print(f"New model not found at {model_path}")
        return

    try:
        # Enable unsafe deserialization for Keras 3/new format if needed
        try:
            tf.keras.config.enable_unsafe_deserialization()
        except:
            pass

        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")

        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
            if hasattr(layer, "stateful"):
                print(f"  - stateful: {layer.stateful}")
            else:
                print(f"  - no stateful attr")

    except Exception as e:
        print(f"Error loading new model: {e}")


if __name__ == "__main__":
    check_old_model()
    check_new_model()
