import os
import joblib
import pickle
import tensorflow as tf
import keras

# Configure Keras to allow unsafe deserialization (needed for some models/scalers)
keras.config.enable_unsafe_deserialization()


def load_scaler(directory, scaler_name):
    """
    Load a scaler (joblib or pickle) from the specified directory.
    """
    scaler_lib_path = os.path.join(directory, f"{scaler_name}.lib")
    scaler_p_path = os.path.join(directory, f"{scaler_name}.p")

    if os.path.exists(scaler_lib_path):
        scaler = joblib.load(scaler_lib_path)
    elif os.path.exists(scaler_p_path):
        with open(scaler_p_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        raise FileNotFoundError(
            f"Scaler file not found: {scaler_lib_path} or {scaler_p_path}"
        )

    return scaler


def set_states(model_main, states):
    """
    Set the hidden/cell states of a stateful LSTM model.
    """
    for layer in model_main.layers:
        if hasattr(layer, "reset_states") and layer.stateful:
            name = layer.name
            # Map model layer names to state dictionary keys
            # Expects keys like 'out_h_lstm_layer' for layer 'm_lstm_layer'
            h_key = name.replace("m_", "out_h_")
            c_key = name.replace("m_", "out_c_")

            if h_key in states and c_key in states:
                layer.states[0].assign(states[h_key])
                layer.states[1].assign(states[c_key])


def load_network(
    directory, input_scaler_name="input_scaler", output_scaler_name="output_scaler"
):
    """
    Load the inference and initialization models + scalers from a directory.
    Returns compiled tf.functions for faster inference.
    """
    model_inf_keras = os.path.join(directory, "model_inf.keras")
    model_init_keras = os.path.join(directory, "model_init.keras")

    if not os.path.exists(model_inf_keras):
        raise FileNotFoundError(f"Inference model not found at {model_inf_keras}")
    if not os.path.exists(model_init_keras):
        raise FileNotFoundError(f"Init model not found at {model_init_keras}")

    model_main = keras.models.load_model(model_inf_keras, compile=False)
    model_init = keras.models.load_model(model_init_keras, compile=False)

    print(f"Loaded Main Model from {directory}. Input Shape: {model_main.input_shape}")
    print(f"Loaded Init Model from {directory}. Input Shape: {model_init.input_shape}")

    input_scaler = load_scaler(directory, input_scaler_name)
    output_scaler = load_scaler(directory, output_scaler_name)

    # Wrap in tf.function for performance
    @tf.function(jit_compile=False)
    def predict_main(input_tensor):
        return model_main(input_tensor, training=False)

    @tf.function(jit_compile=False)
    def predict_init(input_tensor):
        return model_init(input_tensor, training=False)

    return (
        model_main,
        model_init,
        input_scaler,
        output_scaler,
        predict_main,
        predict_init,
    )
