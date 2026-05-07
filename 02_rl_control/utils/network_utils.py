import os
import joblib
import pickle
import tensorflow as tf
import keras

try:
    from .platform_utils import should_force_cpu_for_tf_models
except ImportError:
    from platform_utils import should_force_cpu_for_tf_models

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

            try:
                if h_key in states and c_key in states:
                    layer.states[0].assign(states[h_key])
                    layer.states[1].assign(states[c_key])

            except Exception as e:
                print(f"Error setting states in set_states: {e}")


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

    force_cpu_models = should_force_cpu_for_tf_models()
    if force_cpu_models:
        print("[network_utils] Forcing CPU placement for TF model loading/inference.")

    try:
        model_main = _load_keras_model(model_inf_keras, force_cpu=force_cpu_models)
        model_init = _load_keras_model(model_init_keras, force_cpu=force_cpu_models)
    except Exception as exc:
        if force_cpu_models or not _is_cuda_runtime_failure(exc):
            raise

        print(f"[network_utils] CUDA runtime failure while loading models: {exc}")
        print("[network_utils] Retrying model loading on CPU.")
        os.environ["RL_TF_FORCE_CPU_MODELS"] = "1"
        model_main = _load_keras_model(model_inf_keras, force_cpu=True)
        model_init = _load_keras_model(model_init_keras, force_cpu=True)
        force_cpu_models = True

    print(f"Loaded Main Model. Input Shape: {model_main.input_shape}")
    print(f"Loaded Init Model. Input Shape: {model_init.input_shape}")

    input_scaler = load_scaler(directory, input_scaler_name)
    output_scaler = load_scaler(directory, output_scaler_name)

    # Resolve named input keys so Keras Functional models receive dict inputs
    # (avoids "structure of inputs doesn't match" warnings in Keras 3)
    main_input_name = model_main.layers[0].name
    init_input_name = model_init.layers[0].name

    @tf.function(jit_compile=False)
    def predict_main(input_tensor):
        if force_cpu_models:
            with tf.device("/CPU:0"):
                return model_main({main_input_name: input_tensor}, training=False)
        return model_main({main_input_name: input_tensor}, training=False)

    @tf.function(jit_compile=False)
    def predict_init(input_tensor):
        if force_cpu_models:
            with tf.device("/CPU:0"):
                return model_init({init_input_name: input_tensor}, training=False)
        return model_init({init_input_name: input_tensor}, training=False)

    return (
        model_main,
        model_init,
        input_scaler,
        output_scaler,
        predict_main,
        predict_init,
    )


def _load_keras_model(model_path, force_cpu=False):
    if force_cpu:
        with tf.device("/CPU:0"):
            return keras.models.load_model(model_path, compile=False)
    return keras.models.load_model(model_path, compile=False)


def _is_cuda_runtime_failure(exc):
    msg = str(exc)
    markers = (
        "CUDA_ERROR_INVALID_HANDLE",
        "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
        "cuLaunchKernel",
        "device:GPU",
    )
    return any(marker in msg for marker in markers)
