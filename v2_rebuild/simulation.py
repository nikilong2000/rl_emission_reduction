"""
Simulation wrapper for ICE and PG models.
Handles LSTM hidden states and provides a clean interface for predictions.

This uses the existing transition_function_model from the original codebase
to ensure consistency with the trained models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
import joblib
import os
from typing import Tuple, Dict


class Simulation:
    """
    Wrapper for ICE and PG models with clean LSTM state management.

    Uses auxiliary tensor approach matching the original model architecture.
    """

    def __init__(
        self,
        ice_model_path: str,
        pg_model_path: str,
        soc_initial: float = 0.7,
        p_amb_bar: float = 1.0,
        t_amb_k: float = 298.0,
    ):
        """
        Initialize the simulation with ICE and PG models.

        Args:
            ice_model_path: Path to ICE model directory (contains model.h5 and scalers)
            pg_model_path: Path to PG model directory (contains model.h5 and scalers)
            soc_initial: Initial State of Charge [0-1]
            p_amb_bar: Ambient pressure [bar]
            t_amb_k: Ambient temperature [K]
        """
        # Load models
        self.ice_model = self._load_model(ice_model_path, "ICE")
        self.pg_model = self._load_model(pg_model_path, "PG")

        # Load scalers
        self.ice_input_scaler = self._load_scaler(ice_model_path, "input")
        self.ice_output_scaler = self._load_scaler(ice_model_path, "output")
        self.pg_input_scaler = self._load_scaler(pg_model_path, "input")
        self.pg_output_scaler = self._load_scaler(pg_model_path, "output")

        # Store scaler parameters as TF tensors for efficient computation
        self._ice_in_scale = tf.constant(self.ice_input_scaler.scale_, dtype=tf.float32)
        self._ice_in_min = tf.constant(self.ice_input_scaler.min_, dtype=tf.float32)
        self._ice_out_scale = tf.constant(
            self.ice_output_scaler.scale_, dtype=tf.float32
        )
        self._ice_out_min = tf.constant(self.ice_output_scaler.min_, dtype=tf.float32)

        self._pg_in_scale = tf.constant(self.pg_input_scaler.scale_, dtype=tf.float32)
        self._pg_in_min = tf.constant(self.pg_input_scaler.min_, dtype=tf.float32)
        self._pg_out_scale = tf.constant(self.pg_output_scaler.scale_, dtype=tf.float32)
        self._pg_out_min = tf.constant(self.pg_output_scaler.min_, dtype=tf.float32)

        # Ambient conditions
        self.p_amb_bar = p_amb_bar
        self.t_amb_k = t_amb_k

        # Initial state variables
        self.soc_initial = soc_initial

        # Initial auxiliary tensors (scaled initial outputs)
        self.initial_ice_aux = self.ice_output_scaler.transform(
            [[0.0, 0.0, 0.0, 0.0, 0.0]]  # Torque, NO, NO2, CO, CO2
        ).reshape((1, 1, 5))
        self.initial_pg_aux = self.pg_output_scaler.transform(
            [[0.0, soc_initial]]  # velocity, SOC
        ).reshape((1, 1, 2))

        # Reset to initial state
        self.reset()

    def _load_model(self, model_path: str, name: str) -> tf.keras.Model:
        """Load a Keras model from path."""
        model_file = os.path.join(model_path, "model.h5")
        try:
            model = load_model(model_file)
            model.trainable = False
            print(f"✓ Loaded {name} model from {model_file}")
            return model
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load {name} model from {model_file}: {e}"
            )

    def _load_scaler(self, model_path: str, scaler_type: str):
        """Load a scaler (input or output) from path."""
        scaler_file = os.path.join(model_path, f"{scaler_type}_scaler.lib")
        try:
            scaler = joblib.load(scaler_file)
            return scaler
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load {scaler_type} scaler from {scaler_file}: {e}"
            )

    @staticmethod
    def _scale_minmax(x, scale, min_):
        """Apply MinMax scaling: x_scaled = x * scale + min"""
        return x * scale + min_

    @staticmethod
    def _descale_minmax(x_scaled, scale, min_):
        """Inverse MinMax scaling: x = (x_scaled - min) / scale"""
        return (x_scaled - min_) / scale

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset simulation to initial state.

        Returns:
            Dictionary containing initial state variables
        """
        # Reset state variables
        self.velocity = 0.0
        self.soc = self.soc_initial
        self.no = 0.0
        self.no2 = 0.0
        self.co = 0.0
        self.co2 = 0.0
        self.torque = 0.0

        # Reset LSTM states
        if hasattr(self.ice_model, "reset_states"):
            self.ice_model.reset_states()
        if hasattr(self.pg_model, "reset_states"):
            self.pg_model.reset_states()

        # Reset auxiliary tensors
        self.ice_aux = tf.constant(self.initial_ice_aux, dtype=tf.float32)
        self.pg_aux = tf.constant(self.initial_pg_aux, dtype=tf.float32)

        return self.get_state()

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current state variables."""
        return {
            "velocity": np.array([self.velocity], dtype=np.float32),
            "soc": np.array([self.soc], dtype=np.float32),
            "no": np.array([self.no], dtype=np.float32),
            "no2": np.array([self.no2], dtype=np.float32),
            "co": np.array([self.co], dtype=np.float32),
            "co2": np.array([self.co2], dtype=np.float32),
            "torque": np.array([self.torque], dtype=np.float32),
        }

    def step(self, mf: float, brk: float, ice_sp: float) -> Dict[str, np.ndarray]:
        """
        Execute one simulation step.

        Args:
            mf: Motor Front torque demand (EM2_Torque) - normalized
            brk: Brake percentage - normalized
            ice_sp: ICE speed setpoint (Speed_rpm) - normalized

        Returns:
            Dictionary containing updated state variables

        Note: Action values should be in [-1, 1] range and will be denormalized
        to physical units internally based on the expected model ranges.
        """
        # Denormalize actions to physical units
        # Typical ranges (adjust based on actual model training data):
        # ice_sp: 0-6000 RPM -> map [-1,1] to [800, 4000] RPM
        # mf (EM2_Torque): -100 to 100 Nm -> map [-1,1] to [-100, 100] Nm
        # brk: 0-100% -> map [-1,1] to [0, 100]%
        speed_rpm = (ice_sp + 1) * 0.5 * 3200 + 800  # [800, 4000] RPM
        em2_torque = mf * 100.0  # [-100, 100] Nm
        brake_perc = (brk + 1) * 50.0  # [0, 100]%
        m_fuel_mg = max(0, mf * 50.0 + 25.0)  # [0, 75] mg fuel

        # ICE prediction
        ice_input = tf.stack(
            [speed_rpm, m_fuel_mg, self.t_amb_k, self.p_amb_bar], axis=0
        )
        ice_input = tf.cast(ice_input, tf.float32)
        ice_input_scaled = self._scale_minmax(
            ice_input, self._ice_in_scale, self._ice_in_min
        )
        ice_input_scaled = tf.reshape(ice_input_scaled, (1, 1, 4))

        # ICE model call with auxiliary input
        ice_output_scaled = self.ice_model(
            [ice_input_scaled, self.ice_aux], training=False
        )
        ice_output_scaled = tf.reshape(ice_output_scaled, (5,))

        # Descale ICE output
        ice_output = self._descale_minmax(
            ice_output_scaled, self._ice_out_scale, self._ice_out_min
        )
        self.torque, self.no, self.no2, self.co, self.co2 = [
            float(x) for x in tf.unstack(ice_output, num=5)
        ]

        # Update ICE auxiliary tensor for next step
        self.ice_aux = tf.reshape(ice_output_scaled, (1, 1, 5))

        # PG prediction
        pg_input = tf.stack([speed_rpm, em2_torque, self.torque, brake_perc], axis=0)
        pg_input = tf.cast(pg_input, tf.float32)
        pg_input_scaled = self._scale_minmax(
            pg_input, self._pg_in_scale, self._pg_in_min
        )
        pg_input_scaled = tf.reshape(pg_input_scaled, (1, 1, 4))

        # PG model call with auxiliary input
        pg_output_scaled = self.pg_model([pg_input_scaled, self.pg_aux], training=False)
        pg_output_scaled = tf.reshape(pg_output_scaled, (2,))

        # Descale PG output
        pg_output = self._descale_minmax(
            pg_output_scaled, self._pg_out_scale, self._pg_out_min
        )
        self.velocity, self.soc = [float(x) for x in tf.unstack(pg_output, num=2)]

        # Update PG auxiliary tensor for next step
        self.pg_aux = tf.reshape(pg_output_scaled, (1, 1, 2))

        return self.get_state()
