import os
import joblib
import numpy as np
import tensorflow as tf
import keras  # Ensure you have keras installed
import warnings

# Enable unsafe deserialization if using Keras 3 (required for some custom layers/configs)
try:
    keras.config.enable_unsafe_deserialization()
except AttributeError:
    pass  # Older Keras versions might not need this

def get_models(ruta_modelo):
    """
    Loads both init and inference Keras models from a directory.
    Expects 'model_init.keras' and 'model_inf.keras'.
    """
    path_init = os.path.join(ruta_modelo, "model_init.keras")
    path_inf = os.path.join(ruta_modelo, "model_inf.keras")

    print(f"🔍 Loading models from: {ruta_modelo}")
    try:
        # standard loading; compile=False prevents optimizer loading (faster)
        model_init = keras.models.load_model(path_init, compile=False)
        model_inf = keras.models.load_model(path_inf, compile=False)
        return model_init, model_inf
    except Exception as e:
        print(f"❌ Error loading models from '{ruta_modelo}': {e}")
        raise FileNotFoundError(f"Check if model_init.keras/model_inf.keras exist in {ruta_modelo}") from e

def get_scaler(ruta_model, tipus="input"):
    """Loads a joblib-serialized scaler object."""
    filename = "input_scaler.lib" if tipus == "input" else "output_scaler.lib"
    fitxer = os.path.join(ruta_model, filename)
    return joblib.load(fitxer)

def setup_transition_function_model(ruta_ICE, ruta_PG, SOC_ini=0.7):
    """Factory function to setup the transition model."""
    
    ICE_init, ICE_inf = get_models(ruta_ICE)
    PG_init, PG_inf = get_models(ruta_PG)

    ICE_scale_input = get_scaler(ruta_ICE, "input")
    ICE_scale_output = get_scaler(ruta_ICE, "output")

    PG_scale_input = get_scaler(ruta_PG, "input")
    PG_scale_output = get_scaler(ruta_PG, "output")

    return transition_function_model(
        ICE_init, ICE_inf,
        PG_init, PG_inf,
        ICE_scale_input, ICE_scale_output,
        PG_scale_input, PG_scale_output,
        SOC_ini=SOC_ini,
    )

class transition_function_model:
    def __init__(
        self,
        ICE_init, ICE_inf,
        PG_init, PG_inf,
        ICE_input_scaler, ICE_output_scaler,
        PG_input_scaler, PG_output_scaler,
        SOC_ini=0.7,
        show_progress=False,
    ):
        # Store Models
        self.ICE_model_init = ICE_init
        self.ICE_model_inf = ICE_inf
        self.PG_model_init = PG_init
        self.PG_model_inf = PG_inf
        
        # Ensure models are not trainable (frozen for environment)
        self.ICE_model_init.trainable = False
        self.ICE_model_inf.trainable = False
        self.PG_model_init.trainable = False
        self.PG_model_inf.trainable = False

        # Store Scalers
        self.ICE_input_scaler = ICE_input_scaler
        self.ICE_output_scaler = ICE_output_scaler
        self.PG_input_scaler = PG_input_scaler
        self.PG_output_scaler = PG_output_scaler

        # Tensor Constants for Scaling
        self._ICE_in_scale = tf.constant(ICE_input_scaler.scale_, dtype=tf.float32)
        self._ICE_in_min = tf.constant(ICE_input_scaler.min_, dtype=tf.float32)
        self._ICE_out_scale = tf.constant(ICE_output_scaler.scale_, dtype=tf.float32)
        self._ICE_out_min = tf.constant(ICE_output_scaler.min_, dtype=tf.float32)

        self._PG_in_scale = tf.constant(PG_input_scaler.scale_, dtype=tf.float32)
        self._PG_in_min = tf.constant(PG_input_scaler.min_, dtype=tf.float32)
        self._PG_out_scale = tf.constant(PG_output_scaler.scale_, dtype=tf.float32)
        self._PG_out_min = tf.constant(PG_output_scaler.min_, dtype=tf.float32)

        # Default Initial Outputs (from Config.txt)
        # ICE: 16 outputs [Torque, fuel, NOx_eo, CO_eo, THC_eo, T_gas_eo, NOx_tp, CO_tp, CO2_tp, THC_tp, Temps...]
        self.ICE_initial_vec = np.array([[0, 0, 0, 0, 0, 298, 0, 0, 0, 0, 298, 298, 298, 298, 298, 298]], dtype=np.float32)
        
        # PG: 2 outputs [Vel, SOC]
        self.PG_initial_vec = np.array([[0, SOC_ini]], dtype=np.float32)

        self.show_progress = show_progress
        self.results = []

    @staticmethod
    def _scale_minmax(x, scale, min_):
        return x * scale + min_

    @staticmethod
    def _descale_minmax(x_scaled, scale, min_):
        return (x_scaled - min_) / scale

    def _init_model_states(self, model_init, model_inf, initial_output_vec, scaler, reset_state=True):
        """
        Runs model_init to get states and sets them to model_inf.
        """
        # 1. Scale initial outputs
        # Suppress warnings about feature names mismatch (sklearn vs numpy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            init_val_scaled = scaler.transform(initial_output_vec) # (1, Features)

        init_val_scaled = init_val_scaled.reshape(1, 1, -1)
        
        # 2. Run Init Model to get internal states
        # The model_init returns a list of state tensors
        # Pass input as a list to avoid Keras warning about input structure
        states = model_init([init_val_scaled])
        
        # 3. Reset and Set states in Inference Model
        # Iterate over layers to find stateful LSTM/RNN layers
        # Note: model_init output order matches model_inf layers order usually
        
        state_idx = 0
        # Map output names to states if possible, but assuming standard Keras Functional API order mapping:
        # We need to flatten the states list if it's nested (though usually it's a list of tensors)
        if not isinstance(states, (list, tuple)):
            states = [states]

        # Reset first
        if reset_state:
            for layer in model_inf.layers:
                if hasattr(layer, "reset_states") and layer.stateful:
                    layer.reset_states()

        # Manual State Assignment
        # This part assumes model_init outputs match model_inf stateful layers order
        # We assign the computed states to the inference model variables
        current_state_idx = 0
        for layer in model_inf.layers:
            if hasattr(layer, "states") and layer.stateful:
                # LSTM layers usually have 2 states (h, c). GRU has 1.
                num_states = len(layer.states)
                new_states = states[current_state_idx : current_state_idx + num_states]
                
                # Assign values
                for i in range(num_states):
                    layer.states[i].assign(new_states[i])
                
                current_state_idx += num_states

    def reset_models(self):
        """
        Initializes the stateful models using the init networks.
        """
        # Initialize ICE
        self._init_model_states(
            self.ICE_model_init, 
            self.ICE_model_inf, 
            self.ICE_initial_vec, 
            self.ICE_output_scaler,
            reset_state=True
        )
        
        # Initialize PG
        self._init_model_states(
            self.PG_model_init, 
            self.PG_model_inf, 
            self.PG_initial_vec, 
            self.PG_output_scaler,
            reset_state=True
        )
        
        self.results = []

    def predict_ice(self, Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar):
        """
        One-step ICE prediction.
        """
        # 1. Pack input [Speed, Fuel, T_amb, P_amb]
        x = tf.stack([Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar], axis=0)
        x = tf.cast(x, tf.float32)

        # 2. Scale
        x_scaled = self._scale_minmax(x, self._ICE_in_scale, self._ICE_in_min)
        x_scaled = tf.reshape(x_scaled, (1, 1, 4)) # (Batch, 1, Feats)

        # 3. Inference (Stateful)
        # Wrap input in list to match Keras model expectation
        y_scaled = self.ICE_model_inf([x_scaled], training=False)
        y_scaled = tf.reshape(y_scaled, (16,)) # Flat 16 outputs

        # 4. De-scale
        y = self._descale_minmax(y_scaled, self._ICE_out_scale, self._ICE_out_min)
        
        # 5. Extract relevant outputs (Mapping based on Config)
        # 0: Torque, 6: NOx_tp, 7: CO_tp, 8: CO2_tp
        # step.py expects: Torque, NO, NO2, CO, CO2
        Torque_Nm = y[0]
        NO_out    = y[6]  # NOx Tailpipe
        NO2_out   = 0.0   # Not available separately/Used in loss
        CO_out    = y[7]  # CO Tailpipe
        CO2_out   = y[8]  # CO2 Tailpipe

        return Torque_Nm, NO_out, NO2_out, CO_out, CO2_out

    def predict_PG(self, ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc):
        """
        One-step PG prediction.
        """
        # 1. Pack input - ORDER CHANGED TO: [Speed, ICE_Torque, EM2, Brake]
        # Config: ICE_Speed_rpm, ICE: ICE_Torque_Nm, EM2_Torque_Nm, Brake_perc
        x = tf.stack(
            [ICE_Speed_soll_rpm, ICE_Torque_Nm, EM2_Torque_Nm, Brake_perc], axis=0
        )
        x = tf.cast(x, tf.float32)

        # 2. Scale
        x_scaled = self._scale_minmax(x, self._PG_in_scale, self._PG_in_min)
        x_scaled = tf.reshape(x_scaled, (1, 1, 4))

        # 3. Inference (Stateful)
        # Wrap input in list to match Keras model expectation
        y_scaled = self.PG_model_inf([x_scaled], training=False)
        y_scaled = tf.reshape(y_scaled, (2,))

        # 4. De-scale
        y = self._descale_minmax(y_scaled, self._PG_out_scale, self._PG_out_min)
        Car_Speed_kmph, SOC_1 = tf.unstack(y, num=2)

        return Car_Speed_kmph, SOC_1