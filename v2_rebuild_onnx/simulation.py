"""
Simulation wrapper for ICE and PG models using ONNX.
Handles LSTM hidden states and provides a clean interface for predictions.
"""

import numpy as np
import os
from typing import Dict, Tuple

try:
    from ONNX_Predict.LSTM_onnx import LSTM_onnx
    from ONNX_Predict.Scaler_onnx import Scaler_onnx
except ImportError:
    # Fallback or informative error if not found immediately
    print("Warning: ONNX_Predict not found in path. Ensure it is installed.")


class Simulation:
    """
    Wrapper for ICE and PG ONNX models.
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
        Initialize the simulation with ICE and PG ONNX models.

        Args:
            ice_model_path: Path to ICE model directory (contains .onnx files)
            pg_model_path: Path to PG model directory (contains .onnx files)
            soc_initial: Initial State of Charge [0-1]
            p_amb_bar: Ambient pressure [bar]
            t_amb_k: Ambient temperature [K]
        """
        # Load ICE components
        self.ice_in_scaler = Scaler_onnx('scaler_input.onnx', ice_model_path)
        self.ice_out_scaler = Scaler_onnx('scaler_output.onnx', ice_model_path)
        self.ice_inv_out_scaler = Scaler_onnx('scaler_inverse_output.onnx', ice_model_path)
        self.ice_model = LSTM_onnx('ICE_onnx.onnx', ice_model_path, 'model.h5')

        # Load PG components
        self.pg_in_scaler = Scaler_onnx('scaler_input.onnx', pg_model_path)
        self.pg_out_scaler = Scaler_onnx('scaler_output.onnx', pg_model_path)
        self.pg_inv_out_scaler = Scaler_onnx('scaler_inverse_output.onnx', pg_model_path)
        self.pg_model = LSTM_onnx('PG_onnx.onnx', pg_model_path, 'model.h5')

        # Ambient conditions
        self.p_amb_bar = p_amb_bar
        self.t_amb_k = t_amb_k

        # Initial state variables
        self.soc_initial = soc_initial

        # Initial auxiliary inputs (scaled initial outputs)
        # Using numpy directly
        ice_init_vals = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.initial_ice_aux = self.ice_out_scaler.transform(ice_init_vals).reshape((1, 1, 5))

        pg_init_vals = np.array([[0.0, soc_initial]], dtype=np.float32)
        self.initial_pg_aux = self.pg_out_scaler.transform(pg_init_vals).reshape((1, 1, 2))

        # Reset to initial state
        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset simulation to initial state.

        Returns:
            Dictionary containing initial state variables
        """
        # Reset state variables
        self.velocity = 0.0
        self.soc = self.soc_initial
        self.no, self.no2, self.co, self.co2 = 0.0, 0.0, 0.0, 0.0
        self.torque = 0.0

        # Reset LSTM states
        if hasattr(self.ice_model, "reset_states"):
            self.ice_model.reset_states()
        if hasattr(self.pg_model, "reset_states"):
            self.pg_model.reset_states()

        # Reset auxiliary inputs
        self.ice_aux = self.initial_ice_aux.copy()
        self.pg_aux = self.initial_pg_aux.copy()

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
        """
        # Denormalize actions to physical units
        speed_rpm = (ice_sp + 1) * 0.5 * 3200 + 800  # [800, 4000] RPM
        em2_torque = mf * 100.0  # [-100, 100] Nm
        brake_perc = (brk + 1) * 50.0  # [0, 100]%
        m_fuel_mg = max(0, mf * 50.0 + 25.0)  # [0, 75] mg fuel

        # ICE prediction
        # Input: [Speed_rpm, fuel_mg, T_amb, p_amb]
        ice_input = np.array([[speed_rpm, m_fuel_mg, self.t_amb_k, self.p_amb_bar]], dtype=np.float32)
        
        # Scale input
        ice_input_scaled = self.ice_in_scaler.transform(ice_input)
        ice_input_scaled = np.reshape(ice_input_scaled, (1, 1, 4))
        
        # Model inference
        # The LSTM_onnx __call__ method likely returns the scaled output
        ice_pred_scaled = self.ice_model([ice_input_scaled, self.ice_aux])
        
        # Update aux for next step (the scaled output becomes the aux input)
        self.ice_aux = ice_pred_scaled
        
        # Descale output
        # Output: [Torque, NO, NO2, CO, CO2]
        ice_pred_flat = ice_pred_scaled.reshape(1, -1)
        ice_output = self.ice_inv_out_scaler.transform(ice_pred_flat)[0]
        
        self.torque, self.no, self.no2, self.co, self.co2 = ice_output

        # PG prediction
        # Input: [Speed_rpm, EM2_Torque, ICE_Torque, Brake_perc]
        pg_input = np.array([[speed_rpm, em2_torque, self.torque, brake_perc]], dtype=np.float32)
        
        # Scale input
        pg_input_scaled = self.pg_in_scaler.transform(pg_input)
        pg_input_scaled = np.reshape(pg_input_scaled, (1, 1, 4))
        
        # Model inference
        pg_pred_scaled = self.pg_model([pg_input_scaled, self.pg_aux])
        
        # Update aux for next step
        self.pg_aux = pg_pred_scaled
        
        # Descale output
        # Output: [Velocity, SOC]
        pg_pred_flat = pg_pred_scaled.reshape(1, -1)
        pg_output = self.pg_inv_out_scaler.transform(pg_pred_flat)[0]
        
        self.velocity, self.soc = pg_output

        return self.get_state()
