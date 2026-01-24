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
    """Loads a Keras model from a given directory path.

    This function expects the directory to contain a model file named 'model.h5'.
    It prints the full path before attempting to load the model.

    Parameters
    ----------
    model_path : str
        Path to the directory containing the 'model.h5' file.

    Returns
    -------
    tf.keras.Model
        The loaded Keras model.

    Raises
    ------
    FileNotFoundError
        If the model file cannot be found or loaded from the specified path.
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
    """Sets up the complete transition model by loading all necessary components.

    This factory function loads the ICE (Internal Combustion Engine) and PG (Powertrain)
    Keras models, along with their corresponding input and output scalers, and uses
    them to instantiate and return a `transition_function_model` object.

    Parameters
    ----------
    path_ICE : str
        The directory path for the ICE model and its scalers.
    path_PG : str
        The directory path for the PG model and its scalers.
    initial_SOC : float, optional
        The initial State of Charge for the battery, by default 0.7.

    Returns
    -------
    transition_function_model
        An initialized instance of the transition function model simulator.
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
    """Loads a joblib-serialized scaler object from a model's directory.

    It constructs the path to either 'input_scaler.lib' or 'output_scaler.lib'
    based on the `scaler_type`.

    Parameters
    ----------
    model_path : str
        The directory path where the scaler file is located.
    scaler_type : str, optional
        The type of scaler to load, either 'input' or 'output'. Defaults to "input".

    Returns
    -------
    object
        The loaded scaler object (e.g., a scikit-learn MinMaxScaler).

    Raises
    ------
    ValueError
        If `scaler_type` is not 'input' or 'output'.
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
        PG_model,
        ICE_input_scaler,
        ICE_ouput_scaler,
        PG_input_scaler,
        PG_ouput_scaler,
        Torque=0,
        NO_ini=0,
        NO2_ini=0,
        CO_ini=0,
        CO2_ini=0,
        velocity_ini=0,
        SOC_ini=0.7,
        show_progress=False,
    ):
        """Initializes the vehicle's propulsion system simulator.

        This object encapsulates the prediction models for the internal combustion
        engine (ICE) and the powertrain (PG), along with their respective
        data scalers. It clones the models to avoid modifying the originals
        and converts the scaler parameters into TensorFlow tensors to allow for
        differentiable operations within the computational graph.

        Parameters
        ----------
        ICE_model : tf.keras.Model
            The pre-trained Keras model for the internal combustion engine.
        PG_model : tf.keras.Model
            The pre-trained Keras model for the powertrain and battery.
        ICE_input_scaler : sklearn.preprocessing.MinMaxScaler
            The fitted scaler for the ICE model's inputs.
        ICE_output_scaler : sklearn.preprocessing.MinMaxScaler
            The fitted scaler for the ICE model's outputs.
        PG_input_scaler : sklearn.preprocessing.MinMaxScaler
            The fitted scaler for the PG model's inputs.
        PG_output_scaler : sklearn.preprocessing.MinMaxScaler
            The fitted scaler for the PG model's outputs.
        Torque, NO_ini, ..., SOC_ini : float, optional
            Initial values for the state variables, used to create the
            initial auxiliary tensors. By default, 0 for most and 0.7 for SOC_ini.
        show_progress : bool, optional
            A flag to control progress display (not implemented in the
            provided code). By default, False.
        """

        # ===== Frozen models =====
        self.ICE_model = clone_model(ICE_model)
        self.ICE_model.set_weights(ICE_model.get_weights())
        self.PG_model = clone_model(PG_model)
        self.PG_model.set_weights(PG_model.get_weights())
        self.ICE_model.trainable = False
        self.PG_model.trainable = False

        # ===== Original scalers =====
        self.ICE_input_scaler = ICE_input_scaler
        self.ICE_ouput_scaler = ICE_ouput_scaler
        self.PG_input_scaler = PG_input_scaler
        self.PG_ouput_scaler = PG_ouput_scaler

        # ======= NEW: MinMaxScaler parameters as tensors =======
        self._ICE_in_scale = tf.constant(ICE_input_scaler.scale_, dtype=tf.float32)
        self._ICE_in_min = tf.constant(ICE_input_scaler.min_, dtype=tf.float32)
        self._ICE_out_scale = tf.constant(ICE_ouput_scaler.scale_, dtype=tf.float32)
        self._ICE_out_min = tf.constant(ICE_ouput_scaler.min_, dtype=tf.float32)

        self._PG_in_scale = tf.constant(PG_input_scaler.scale_, dtype=tf.float32)
        self._PG_in_min = tf.constant(PG_input_scaler.min_, dtype=tf.float32)
        self._PG_out_scale = tf.constant(PG_ouput_scaler.scale_, dtype=tf.float32)
        self._PG_out_min = tf.constant(PG_ouput_scaler.min_, dtype=tf.float32)

        # ===== Auxiliary variables =====
        self.initial_ice_aux = ICE_ouput_scaler.transform(
            [[Torque, NO_ini, NO2_ini, CO_ini, CO2_ini]]
        ).reshape((1, 1, 5))
        self.initial_PG_aux = PG_ouput_scaler.transform(
            [[velocity_ini, SOC_ini]]
        ).reshape((1, 1, 2))
        self.ice_aux = tf.constant(self.initial_ice_aux, dtype=tf.float32)
        self.PG_aux = tf.constant(self.initial_PG_aux, dtype=tf.float32)

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
        and returns the auxiliary states to their initial TF tensors.
        """
        # 1) Reset internal states of each model
        if hasattr(self.ICE_model, "reset_states"):
            self.ICE_model.reset_states()
        if hasattr(self.PG_model, "reset_states"):
            self.PG_model.reset_states()

        # 2) Reset auxiliary variables: return to constant tensors
        self.ice_aux = tf.constant(self.initial_ice_aux, dtype=tf.float32)
        self.PG_aux = tf.constant(self.initial_PG_aux, dtype=tf.float32)

        # 3) Clear previous results
        self.results = []

    # ---------- ICE prediction ----------
    def predict_ice(self, Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar):
        """Performs a one-step prediction for the internal combustion engine (ICE) model.

        It takes inputs in physical units, scales them, runs inference for one
        time step, and descales the outputs to return them in their corresponding
        physical units. All operations are performed using TensorFlow tensors.

        Parameters
        ----------
        Speed_rpm : tf.Tensor
            Engine speed in RPM.
        m_fuel_mg : tf.Tensor
            Mass of fuel injected per cycle in milligrams.
        T_amb_K : tf.Tensor
            Ambient temperature in Kelvin.
        p_amb_bar : tf.Tensor
            Ambient pressure in bar.

        Returns
        -------
        tuple[tf.Tensor, ...]
            A tuple with the predicted outputs in physical units:
            (Torque_Nm, NO_out, NO2_out, CO_out, CO2_out).
        """

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
        """Performs a one-step prediction for the powertrain (PG) model.

        It takes inputs in physical units (including the predicted torque from
        the ICE model), scales them, runs the PG model inference, and descales
        the outputs to their physical units. All operations are performed
        using TensorFlow tensors.

        Parameters
        ----------
        ICE_Speed_soll_rpm : tf.Tensor
            The target engine speed in RPM.
        EM2_Torque_Nm : tf.Tensor
            The torque of the electric motor 2 in Nm.
        ICE_Torque_Nm : tf.Tensor
            The torque of the combustion engine (output of `predict_ice`) in Nm.
        Brake_perc : tf.Tensor
            The applied braking percentage (0-100).

        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            A tuple with the predicted outputs in physical units:
            (Car_Speed_kmph, SOC_1).
        """
        # 1. empaquetar entrada
        x = tf.stack(
            [ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc], axis=0
        )
        x = tf.cast(x, tf.float32)

        # 2. escalar
        x_scaled = self._scale_minmax(x, self._PG_in_scale, self._PG_in_min)
        x_scaled = tf.reshape(x_scaled, (1, 1, 4))

        # 3. modelo PG
        y_scaled = self.PG_model([x_scaled, self.PG_aux], training=False)
        y_scaled = tf.reshape(y_scaled, (2,))

        # 4. des-escalar
        y = self._descale_minmax(y_scaled, self._PG_out_scale, self._PG_out_min)
        Car_Speed_kmph, SOC_1 = tf.unstack(y, num=2)

        return Car_Speed_kmph, SOC_1

    # ---------- flujo completo (sin cambios de lógica) ----------
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
        """Executes the complete prediction flow for one simulation step.

        This method orchestrates the simulation: it first calls `predict_ice` to
        obtain the engine torque and emissions, and then uses that torque as an
        input for `predict_PG` to get the vehicle's speed and the battery's
        state of charge.

        Parameters
        ----------
        Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar : tf.Tensor
            Inputs for the ICE model.
        ICE_Speed_soll_rpm, Brake_perc, EM2_Torque_Nm : tf.Tensor
            Inputs for the PG model.

        Returns
        -------
        tuple[tf.Tensor, ...]
            A tuple containing all outputs from both models in physical units:
            (ICE_Torque_Nm, NO_out_1, NO2_out_1, CO_out_1, CO2_out_1,
            Car_Vel_kmph, SOC_1).
        """
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
