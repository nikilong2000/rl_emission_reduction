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
        self.initial_ice_aux = np.array(self.ice_out_scaler.transform(ice_init_vals)).reshape((1, 1, 5))

        pg_init_vals = np.array([[0.0, soc_initial]], dtype=np.float32)
        self.initial_pg_aux = np.array(self.pg_out_scaler.transform(pg_init_vals)).reshape((1, 1, 2))

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
        self.soc = self.soc_initial # TODO: check if makes sense to not reset to 0
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

    def step(self, mf: float, brk: float, ice_sp: float, em2_torque_cmd: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Execute one simulation step.
        """
        # Denormalize actions to physical units
        # ice_sp: [-1, 1] -> [0, 4500] RPM
        speed_rpm = (ice_sp + 1) * 0.5 * 4500.0

        # Turn off ICE if rotational speed is low (gradient-based model logic)
        # In GBM: mask_ice = ice_sp < 900.0
        # If masked: torque=0, mf=0 (which implies fuel=0 or special handling)
        
        # em2_torque: [-1, 1] -> [-421, 421] Nm
        # Decoupled from mf: em2_torque_cmd controls the electric motor
        em2_torque = em2_torque_cmd * 421.0
        
        # m_fuel_mg: [-1, 1] -> [3, 70] mg 
        # Decoupled from em2_torque: mf now strictly controls fuel
        m_fuel_mg = (mf + 1) * 0.5 * (70.0 - 3.0) + 3.0

        brake_perc = (brk + 1) * 50.0  # [0, 100]%

        if speed_rpm < 900.0:
            m_fuel_mg = 3.0 # or 0? GBM sets mf=0 where mf is likely clipped or raw. 
            # GBM logic: torque = tf.where(mask_ice, tf.zeros_like(torque), torque)
            # You requested GBM behavior.
            # However, GBM uses 'mf' as input to ICE. If speed < 900, it zeroes out torque.
            # I will apply the torque zeroing AFTER prediction to match GBM exactly, 
            # rather than modifying inputs here which might disturb the LSTM state continuity.
            pass


        # ICE prediction
        # Input: [Speed_rpm, fuel_mg, T_amb, p_amb]
        ice_input = np.array([[speed_rpm, m_fuel_mg, self.t_amb_k, self.p_amb_bar]], dtype=np.float32)
        
        # Scale input
        ice_input_scaled = self.ice_in_scaler.transform(ice_input)
        ice_input_scaled = np.reshape(ice_input_scaled, (1, 1, 4))
        
        # Model inference
        # The LSTM_onnx __call__ method likely returns the scaled output
        # Returns a list of numpy arrays, we want the first one.
        ice_pred_scaled = self.ice_model([ice_input_scaled, self.ice_aux])[0]
        
        # Update aux for next step (the scaled output becomes the aux input)
        self.ice_aux = ice_pred_scaled
        
        # Descale output
        # Output: [Torque, NO, NO2, CO, CO2]
        ice_pred_flat = ice_pred_scaled.reshape(1, -1)
        ice_output = self.ice_inv_out_scaler.transform(ice_pred_flat)[0][0]
        
        self.torque, self.no, self.no2, self.co, self.co2 = ice_output

        # GBM Logic: Clipping and Low Speed Masking
        self.torque = float(np.clip(self.torque, -50.0, 300.0))
        
        if speed_rpm < 900.0:
            self.torque = 0.0

        # PG prediction
        # Input: [Speed_rpm, EM2_Torque, ICE_Torque, Brake_perc]
        pg_input = np.array([[speed_rpm, em2_torque, self.torque, brake_perc]], dtype=np.float32)
        
        # Scale input
        pg_input_scaled = self.pg_in_scaler.transform(pg_input)
        pg_input_scaled = np.reshape(pg_input_scaled, (1, 1, 4))
        
        # Model inference
        pg_pred_scaled = self.pg_model([pg_input_scaled, self.pg_aux])[0]
        
        # Update aux for next step
        self.pg_aux = pg_pred_scaled
        
        # Descale output
        # Output: [Velocity, SOC]
        pg_pred_flat = pg_pred_scaled.reshape(1, -1)
        pg_output = self.pg_inv_out_scaler.transform(pg_pred_flat)[0][0]
        
        self.velocity, self.soc = pg_output

        return self.get_state()
