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


# --- CONFIG PARSER (FROM MODULAR_NN) ---
def parse_config(config_path):
    config = {}
    global_settings = {}
    current_network = None

    def to_bool(val: str) -> bool:
        return str(val).strip().lower() in ("true", "1", "yes", "y")

    with open(config_path, "r") as file:
        lines = file.readlines()
        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            if "=" not in line:
                current_network = line
                config[current_network] = {}
                continue

            key, value = map(str.strip, line.split("=", 1))

            if current_network is None:
                global_settings[key] = value
                continue

            if key == "directory":
                config[current_network]["directory"] = value
            elif key == "inputs":
                config[current_network]["inputs"] = [
                    x.strip() for x in value.split(",")
                ]
            elif key == "outputs":
                config[current_network]["outputs"] = [
                    x.strip() for x in value.split(",")
                ]
            elif key == "stateful":
                config[current_network]["stateful"] = to_bool(value)
            elif key == "initial_outputs":
                config[current_network]["initial_outputs"] = [
                    float(x.strip()) for x in value.split(",")
                ]
    return config, global_settings


def get_initial_values(details, df):
    """
    Replicates 'aux_initial_true_values' logic from Modular_NN.
    Prioritizes 'initial_outputs' in config.
    Falls back to first row of 'outputs' in CSV (df).
    """
    if "initial_outputs" in details and "outputs" in details:
        # Use Config Values
        vals = details["initial_outputs"]
    elif "outputs" in details:
        # Fallback to CSV Truth
        vals = []
        for name in details["outputs"]:
            # Need to find column name in df (might have units/brackets)
            # Modular_NN splits by '(' or ' '. We do simple check.
            match = None
            if name in df.columns:
                match = name
            else:
                # Try matching start
                for col in df.columns:
                    if col.startswith(name):
                        match = col
                        break

            if match:
                vals.append(float(df[match].iloc[0]))
            else:
                vals.append(0.0)
    else:
        # Fallback zeros of generic size (risky, but shouldn't happen with valid config)
        vals = [0.0]

    return np.array([vals], dtype=np.float32)


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
            # Resize logic if needed, but we should pass correct size
            temp_aux = np.zeros((1, self.output_scaler.n_features_in_))
            cols = min(aux_values.shape[1], temp_aux.shape[1])
            temp_aux[:, :cols] = aux_values[:, :cols]
            aux_values = temp_aux

        aux_scaled = self.output_scaler.transform(aux_values).reshape((1, 1, -1))

        # 1. Get Initial States from Init Model
        state_tensors = self.predict_init_func(aux_scaled)

        # 2. Map outputs to layer states
        # Assumes naming convention: Main Layer 'm_XXX' -> Init Outputs 'out_h_XXX', 'out_c_XXX'
        states = dict(zip(self.model_init.output_names, state_tensors))

        for layer in self.model_main.layers:
            if hasattr(layer, "reset_states") and getattr(layer, "stateful", False):
                name = layer.name
                h_key = name.replace("m_", "out_h_")
                c_key = name.replace("m_", "out_c_")

                if h_key in states and c_key in states:
                    # Assign states (h, c)
                    layer.states[0].assign(states[h_key])
                    layer.states[1].assign(states[c_key])
                else:
                    print(f"Warning: No matching init states found for layer {name}")

    def step(self, input_val):
        x_scaled = self.input_scaler.transform(input_val).reshape((1, 1, -1))
        out = self.predict_main_func(x_scaled)
        return self.output_scaler.inverse_transform(out.numpy()[0])


# --- MAIN ---


def main():
    print("--- BENCHMARK STARTED ---")

    # Paths
    CONFIG_PATH = "../internal_lstm_models/NN_Application/config.txt"
    NEW_ICE_DIR = "../internal_lstm_models/NN_Application/Nets/ICE"
    NEW_DRV_DIR = "../internal_lstm_models/NN_Application/Nets/Drivetrain"
    CSV_PATH = "../internal_lstm_models/Test_Cycles/WLTC.csv"

    # 1. PARSE CONFIG
    print(f"Parsing config from {CONFIG_PATH}...")
    config, _ = parse_config(CONFIG_PATH)

    # Extract Columns from Config
    ice_conf = config.get("ICE")
    drv_conf = config.get("Drivetrain")

    ICE_COLS = ice_conf.get("outputs", [])
    DRV_COLS = drv_conf.get("outputs", [])

    print(f"ICE Outputs defined: {len(ICE_COLS)}")
    print(f"Drivetrain Outputs defined: {len(DRV_COLS)}")

    # 2. LOAD DATA
    print(f"Loading data from {CSV_PATH}...")

    # Load all columns to support init fallback
    df = pd.read_csv(CSV_PATH, sep=";")

    # Helper to find columns loosely (like Modular_NN)
    def get_col(name):
        if name in df.columns:
            return df[name].values
        for c in df.columns:
            if c.split("(")[0].strip() == name:
                return df[c].values
        raise KeyError(f"Column '{name}' not found in CSV")

    # Extract Inputs
    # We explicitly grab the columns needed for the loop to avoid magic numbers
    # ICE Inputs: ICE_Speed_rpm, fuel_mg, T_amb_K, p_amb_bar
    # Drv Inputs: ICE_Speed_rpm, (Torque), EM2_Torque_Nm, Brake_perc
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

    # Stack for iteration: [Speed, Fuel, T_amb, P_amb, EM2, Brake]
    data = np.column_stack([ice_speed, fuel, t_amb, p_amb, em2_torque, brake]).astype(
        np.float32
    )
    print(f"Data loaded: {len(data)} steps.")

    # 3. BENCHMARK NEW MODELS
    print("\n--- Benchmarking NEW Models ---")

    # Store results
    ice_results = []
    drv_results = []

    try:
        new_ice = NewModelWrapper(NEW_ICE_DIR, "ICE")
        new_drv = NewModelWrapper(NEW_DRV_DIR, "Drivetrain")

        # Determine Initial Values using Logic from Modular_NN
        print("Determining initial values...")
        ice_init_vals = get_initial_values(ice_conf, df)
        drv_init_vals = get_initial_values(drv_conf, df)

        print(f"ICE Init Vector (Shape {ice_init_vals.shape}):\n {ice_init_vals}")
        print(f"Drv Init Vector (Shape {drv_init_vals.shape}):\n {drv_init_vals}")

        new_ice.initialize(ice_init_vals)
        new_drv.initialize(drv_init_vals)

        # Note: Do NOT call reset_states() here anymore as it wipes the init!

        with Timer() as t_new:
            for i in range(len(data)):
                # Data indices based on stack above:
                # 0: Speed, 1: Fuel, 2: T_amb, 3: P_amb, 4: EM2, 5: Brake

                # 1. Prediction ICE
                # Inputs: [Speed, Fuel, T_amb, P_amb]
                ice_in = data[i, [0, 1, 2, 3]].reshape(1, -1)

                ice_out = new_ice.step(ice_in)
                ice_results.append(ice_out.flatten())

                # 2. Extract Torque
                if "ICE_Torque_Nm" in ICE_COLS:
                    torque_idx = ICE_COLS.index("ICE_Torque_Nm")
                else:
                    torque_idx = 0

                torque = ice_out[0, torque_idx]

                # 3. Prediction Drivetrain
                # Inputs: ICE_Speed_rpm, ICE: ICE_Torque_Nm, EM2_Torque_Nm, Brake_perc
                # Constructed: [Speed, Torque, EM2, Brake]
                drv_input_val = np.array([[data[i, 0], torque, data[i, 4], data[i, 5]]])

                drv_out = new_drv.step(drv_input_val)
                drv_results.append(drv_out.flatten())

        print(
            f"New Models (ICE + Drivetrain) Total Time: {t_new.duration:.4f}s ({t_new.duration/len(data)*1000:.4f} ms/step)"
        )

        # Save Results
        print("Saving results to 'new_model_predictions.csv'...")
        ice_np = np.array(ice_results)
        drv_np = np.array(drv_results)

        # Use Explicit Column Names from Config
        ice_cols_final = [f"ICE_{c}" for c in ICE_COLS]
        drv_cols_final = [f"DRV_{c}" for c in DRV_COLS]

        # Safety check on shapes
        if ice_np.shape[1] != len(ice_cols_final):
            print("Warning: ICE output shape mismatch with config columns.")
            ice_cols_final = [f"ICE_Out_{j}" for j in range(ice_np.shape[1])]

        if drv_np.shape[1] != len(drv_cols_final):
            print("Warning: Drv output shape mismatch with config columns.")
            drv_cols_final = [f"DRV_Out_{j}" for j in range(drv_np.shape[1])]

        df_res = pd.DataFrame(
            np.hstack([ice_np, drv_np]), columns=ice_cols_final + drv_cols_final
        )
        df_res.to_csv(
            "new_model_predictions.csv",
            index=False,
        )
        print("Results saved.")

    except Exception as e:
        print(f"FAILED NEW MODELS: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
