import numpy as np
import pandas as pd  # ← keeping it because the rest of the file uses it
import tensorflow as tf  # NEW
from tensorflow.keras.models import clone_model, load_model
import joblib
import os

import numpy as np
import pandas as pd
from tensorflow.keras.models import clone_model, load_model
import joblib
import os

import numpy as np
import pandas as pd  # ← keeping it because the rest of the file uses it
import tensorflow as tf  # NEW
from tensorflow.keras.models import clone_model, load_model
import joblib
import os


def get_model(ruta_modelo):
    """
    Loads a Keras model from the given path (must contain 'model.h5')
    and displays relevant information about the model on the screen.

    Parameters:
    ruta_modelo (str): Path to the directory containing the 'model.h5' file
    """

    ruta_completa = os.path.join(ruta_modelo, "model.h5")
    # normalize path separators
    ruta_completa = os.path.normpath(ruta_completa)

    print(f"🔍 Attempting to load model from: {ruta_completa}")  # ← key

    try:
        return load_model(ruta_completa)
    except Exception as e:
        print(f"❌ Error loading model from '{ruta_completa}': {e}")
        raise FileNotFoundError(f"Model not found at '{ruta_completa}': {e}") from e


def setup_transition_function_model(ruta_ICE, ruta_PG, SOC_ini=0.7):
    """
    Configures the transition model by loading the necessary models and scalers.

    Parameters:
    ruta_ICE (str): Path to the ICE model.
    ruta_PG (str): Path to the PG model.
    """

    ICE_model = get_model(ruta_ICE)
    PG_model = get_model(ruta_PG)

    ICE_scale_input = get_scaler(ruta_ICE, tipus="input")
    ICE_scale_output = get_scaler(ruta_ICE, tipus="output")

    PG_scale_input = get_scaler(ruta_PG, tipus="input")
    PG_scale_output = get_scaler(ruta_PG, tipus="output")

    return transition_function_model(
        ICE_model,
        PG_model,
        ICE_scale_input,
        ICE_scale_output,
        PG_scale_input,
        PG_scale_output,
        SOC_ini=SOC_ini,
    )


def get_scaler(ruta_model, tipus="input"):
    """
    Loads a scaler (input or output) from a path and displays relevant information.

    param ruta_model (str): Path where the model is located
    param nom_model (str): Name of the model (for printing)
    param tipus (str): 'input' or 'output' depending on which scaler you want to load

    return: Returns the loaded scaler object
    """

    if tipus == "input":
        fitxer = os.path.join(ruta_model, "input_scaler.lib")
    elif tipus == "output":
        fitxer = os.path.join(ruta_model, "output_scaler.lib")
    else:
        raise ValueError("The 'tipus' parameter must be 'input' or 'output'")

    scaler = joblib.load(fitxer)
    return scaler


class transition_function_model:
    def __init__(
        self,
        ICE_model,
        Outlook_model,
        ICE_input_scaler,
        ICE_ouput_scaler,
        OUTLOOK_input_scaler,
        OUTLOOK_ouput_scaler,
        Torque=0,
        NO_ini=0,
        NO2_ini=0,
        CO_ini=0,
        CO2_ini=0,
        velocity_ini=0,
        SOC_ini=0.7,
        show_progress=False,
    ):

        # ===== Frozen models =====
        self.ICE_model = clone_model(ICE_model)
        self.ICE_model.set_weights(ICE_model.get_weights())
        self.Outlook_model = clone_model(Outlook_model)
        self.Outlook_model.set_weights(Outlook_model.get_weights())
        self.ICE_model.trainable = False
        self.Outlook_model.trainable = False

        # ===== Original scalers =====
        self.ICE_input_scaler = ICE_input_scaler
        self.ICE_ouput_scaler = ICE_ouput_scaler
        self.OUTLOOK_input_scaler = OUTLOOK_input_scaler
        self.OUTLOOK_ouput_scaler = OUTLOOK_ouput_scaler

        # ======= NEW: MinMaxScaler parameters as tensors =======
        self._ICE_in_scale = tf.constant(ICE_input_scaler.scale_, dtype=tf.float32)
        self._ICE_in_min = tf.constant(ICE_input_scaler.min_, dtype=tf.float32)
        self._ICE_out_scale = tf.constant(ICE_ouput_scaler.scale_, dtype=tf.float32)
        self._ICE_out_min = tf.constant(ICE_ouput_scaler.min_, dtype=tf.float32)

        self._PG_in_scale = tf.constant(OUTLOOK_input_scaler.scale_, dtype=tf.float32)
        self._PG_in_min = tf.constant(OUTLOOK_input_scaler.min_, dtype=tf.float32)
        self._PG_out_scale = tf.constant(OUTLOOK_ouput_scaler.scale_, dtype=tf.float32)
        self._PG_out_min = tf.constant(OUTLOOK_ouput_scaler.min_, dtype=tf.float32)

        # ===== Auxiliary variables =====
        self.initial_ice_aux = ICE_ouput_scaler.transform(
            [[Torque, NO_ini, NO2_ini, CO_ini, CO2_ini]]
        ).reshape((1, 1, 5))
        self.initial_outlook_aux = OUTLOOK_ouput_scaler.transform(
            [[velocity_ini, SOC_ini]]
        ).reshape((1, 1, 2))
        self.ice_aux = tf.constant(self.initial_ice_aux, dtype=tf.float32)
        self.outlook_aux = tf.constant(self.initial_outlook_aux, dtype=tf.float32)

        self.show_progress = show_progress
        self.results = []

    # ---------- Internal utilities (differentiable) ----------
    @staticmethod
    def _scale_minmax(x, scale, min_):
        """x_scaled = x * scale + min"""
        return x * scale + min_

    @staticmethod
    def _descale_minmax(x_scaled, scale, min_):
        """x = (x_scaled - min) / scale"""
        return (x_scaled - min_) / scale

    def reset_models(self):
        """
        Resets the internal states of ICE and PG (if they have reset_states),
        and returns to the initial auxiliary states as TF tensors.
        """
        # 1) Reset internal states of each model
        if hasattr(self.ICE_model, "reset_states"):
            self.ICE_model.reset_states()
        if hasattr(self.Outlook_model, "reset_states"):
            self.Outlook_model.reset_states()

        # 2) Reset auxiliary variables: return to constant tensors
        self.ice_aux = tf.constant(self.initial_ice_aux, dtype=tf.float32)
        self.outlook_aux = tf.constant(self.initial_outlook_aux, dtype=tf.float32)

        # 3) Clear previous results
        self.results = []

    # ---------- ICE prediction ----------
    def predict_ice(self, Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar):
        # 1. pack input
        x = tf.stack([Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar], axis=0)
        x = tf.cast(x, tf.float32)

        # 2. scale with MinMax in TF
        x_scaled = self._scale_minmax(x, self._ICE_in_scale, self._ICE_in_min)
        x_scaled = tf.reshape(x_scaled, (1, 1, 4))

        # 3. execute ICE model (without .predict)
        y_scaled = self.ICE_model([x_scaled, self.ice_aux], training=False)
        # (3,)
        y_scaled = tf.reshape(y_scaled, (5,))

        # 4. de-scale to physical magnitudes
        y = self._descale_minmax(y_scaled, self._ICE_out_scale, self._ICE_out_min)
        Torque_Nm, NO_out, NO2_out, CO_out, CO2_out = tf.unstack(y, num=5)

        return Torque_Nm, NO_out, NO2_out, CO_out, CO2_out

    # ---------- PG prediction ----------
    def predict_PG(self, ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc):
        # 1. pack input
        x = tf.stack(
            [ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc], axis=0
        )
        x = tf.cast(x, tf.float32)

        # 2. scale
        x_scaled = self._scale_minmax(x, self._PG_in_scale, self._PG_in_min)
        x_scaled = tf.reshape(x_scaled, (1, 1, 4))

        # 3. PG model
        y_scaled = self.Outlook_model([x_scaled, self.outlook_aux], training=False)
        y_scaled = tf.reshape(y_scaled, (2,))

        # 4. de-scale
        y = self._descale_minmax(y_scaled, self._PG_out_scale, self._PG_out_min)
        Car_Speed_kmph, SOC_1 = tf.unstack(y, num=2)

        return Car_Speed_kmph, SOC_1

    # ---------- complete flow (logic unchanged) ----------
    def predict(
        self,
        Speed_rpm,
        m_fuel_mg,
        T_amb_K,
        p_amb_bar,
        ICE_Speed_soll_rpm,
        Brake_perc,
        EM2_Torque_Nm,
    ):

        ICE_Torque_Nm, NO_out_1, NO2_out_1, CO_out_1, CO2_out_1 = self.predict_ice(
            Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar
        )

        Car_Vel_kmph, SOC_1 = self.predict_PG(
            ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc
        )

        self.results.append(
            [
                ICE_Torque_Nm,
                NO_out_1,
                NO2_out_1,
                CO_out_1,
                CO2_out_1,
                Car_Vel_kmph,
                SOC_1,
            ]
        )

        return (
            ICE_Torque_Nm,
            NO_out_1,
            NO2_out_1,
            CO_out_1,
            CO2_out_1,
            Car_Vel_kmph,
            SOC_1,
        )
