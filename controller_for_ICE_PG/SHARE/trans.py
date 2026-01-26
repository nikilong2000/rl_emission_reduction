import os
import numpy as np
import pandas as pd

# Import classes from NEW CODE
from ONNX_Predict.LSTM_onnx import LSTM_onnx
from ONNX_Predict.Scaler_onnx import Scaler_onnx


def setup_transition_function_model(ruta_ICE, ruta_PG, SOC_ini=0.7):
    """
    Configures the transition environment by loading ONNX models and their scalers.
    """
    print(f"⚙️ Configuring ONNX environment...")
    print(f"   📍 ICE: {ruta_ICE}")
    print(f"   📍 PG:  {ruta_PG}")

    # --- ICE Model and Scalers Loading ---
    # I assume the model file is named 'ICE_onnx.onnx' based on your example
    # and that it requires a reference to a dummy or real 'model.h5' if the library requests it.
    tf_model_ref = "model.h5"

    ice_model = LSTM_onnx("ICE_onnx.onnx", ruta_ICE, tf_model_ref)

    ice_scalers = {
        "in": Scaler_onnx("scaler_input.onnx", ruta_ICE),
        "out": Scaler_onnx("scaler_output.onnx", ruta_ICE),  # To scale initial state
        "inv_out": Scaler_onnx(
            "scaler_inverse_output.onnx", ruta_ICE
        ),  # To de-scale prediction
    }

    # --- PG Model and Scalers Loading ---
    pg_model = LSTM_onnx("PG_onnx.onnx", ruta_PG, tf_model_ref)

    pg_scalers = {
        "in": Scaler_onnx("scaler_input.onnx", ruta_PG),
        "out": Scaler_onnx("scaler_output.onnx", ruta_PG),
        "inv_out": Scaler_onnx("scaler_inverse_output.onnx", ruta_PG),
    }

    return transition_function_model(
        ice_model, pg_model, ice_scalers, pg_scalers, SOC_ini=SOC_ini
    )


class transition_function_model:
    def __init__(
        self,
        ICE_model,
        PG_model,
        ICE_scalers,
        PG_scalers,
        Torque=0,
        NO_ini=0,
        NO2_ini=0,
        CO_ini=0,
        CO2_ini=0,
        velocity_ini=0,
        SOC_ini=0.7,
        show_progress=False,
    ):
        """
        Initializes the propulsion system simulator using ONNX models.
        """
        # ONNX Models
        self.ICE_model = ICE_model
        self.PG_model = PG_model

        # Scaler dictionaries {'in', 'out', 'inv_out'}
        self.ICE_scalers = ICE_scalers
        self.PG_scalers = PG_scalers

        # --- Initial State Configuration ---
        # We save initial physical values to be able to reset later
        self.init_vals_ICE = np.array(
            [[Torque, NO_ini, NO2_ini, CO_ini, CO2_ini]], dtype="float32"
        )
        self.init_vals_PG = np.array([[velocity_ini, SOC_ini]], dtype="float32")

        # Transform initial values to normalized scale (0-1) to feed the LSTM
        # Note: We use [0] because transform returns (1, features) and we want the row.
        # Then we reshape to (1, 1, features) to enter the model.

        self.initial_ice_aux = (
            self.ICE_scalers["out"].transform(self.init_vals_ICE)[0].reshape((1, 1, 5))
        )
        self.initial_pg_aux = (
            self.PG_scalers["out"].transform(self.init_vals_PG)[0].reshape((1, 1, 2))
        )

        # Current state (will be updated step by step)
        self.ice_aux = self.initial_ice_aux.copy()
        self.pg_aux = self.initial_pg_aux.copy()

        self.show_progress = show_progress
        self.results = []

    def reset_models(self):
        """
        Resets LSTM internal states and auxiliary variables.
        """
        # 1. Reset internal states of LSTM_onnx class
        if hasattr(self.ICE_model, "reset_states"):
            self.ICE_model.reset_states()
        if hasattr(self.PG_model, "reset_states"):
            self.PG_model.reset_states()

        # 2. Reset feedback variables (explicit recurrent state)
        self.ice_aux = self.initial_ice_aux.copy()
        self.pg_aux = self.initial_pg_aux.copy()

        # 3. Clear history
        self.results = []
        print("🔄 Models reset to initial state.")

    # ---------- ICE Prediction ----------
    def predict_ice(self, Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar):
        """
        Executes ICE inference step by step.
        Inputs: Scalar values or numpy floats.
        Output: Physical values (de-scaled).
        """
        # 1. Pack input into numpy array (batch=1, time=1, features=4)
        # Ensure float32
        x = np.array([[Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar]], dtype="float32")

        # 2. Scale input
        x_scaled = self.ICE_scalers["in"].transform(x)
        x_scaled = np.reshape(x_scaled, (1, 1, 4))

        # 3. Execute ONNX Model
        y_predict_scaled = self.ICE_model([x_scaled, self.ice_aux])

        # --- CORRECTION: Extract array from list with [0] ---
        curr_output_scaled = y_predict_scaled[0]

        # Ensure dimensions (1, 1, 5) for next iteration (Rank 3)
        # If it comes as (1, 5) or (5,), force it to (1, 1, 5)
        curr_output_scaled = curr_output_scaled.reshape((1, 1, 5))

        # Save state for cycle t+1 (ensuring float32)
        self.ice_aux = curr_output_scaled.astype("float32")

        # 4. De-scale to physical magnitudes
        # ... (rest of code stays the same with .flatten() added earlier)
        y_phys = self.ICE_scalers["inv_out"].transform(curr_output_scaled.reshape(1, 5))
        vals = np.array(y_phys).flatten()
        Torque_Nm, NO_out, NO2_out, CO_out, CO2_out = vals

        return Torque_Nm, NO_out, NO2_out, CO_out, CO2_out

    # ---------- PG Prediction ----------
    def predict_PG(self, ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc):
        """
        Executes PG inference step by step.
        """
        # 1. Pack input
        x = np.array(
            [[ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc]],
            dtype="float32",
        )

        # 2. Scale input
        x_scaled = self.PG_scalers["in"].transform(x)
        x_scaled = np.reshape(x_scaled, (1, 1, 4))

        # 3. Execute model
        y_predict_scaled = self.PG_model([x_scaled, self.pg_aux])

        curr_output_scaled = y_predict_scaled[0]

        # Ensure dimensions (1, 1, 2) for next iteration
        curr_output_scaled = curr_output_scaled.reshape((1, 1, 2))

        self.pg_aux = curr_output_scaled.astype("float32")  # Update state

        # 4. De-scale
        y_phys = self.PG_scalers["inv_out"].transform(curr_output_scaled.reshape(1, 2))
        vals = np.array(y_phys).flatten()
        Car_Speed_kmph, SOC_1 = vals

        return Car_Speed_kmph, SOC_1

    # ---------- Full Flow ----------
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
        """
        Orchestrates simulation: First predicts thermal engine (ICE),
        then uses that Torque to predict vehicle dynamics (PG).
        """
        # 1. Predict ICE
        ICE_Torque_Nm, NO_out, NO2_out, CO_out, CO2_out = self.predict_ice(
            Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar
        )

        # 2. Predict PG (using the Torque predicted above)
        Car_Vel_kmph, SOC_1 = self.predict_PG(
            ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc
        )

        # Save history if necessary
        self.results.append(
            [ICE_Torque_Nm, NO_out, NO2_out, CO_out, CO2_out, Car_Vel_kmph, SOC_1]
        )

        return (ICE_Torque_Nm, NO_out, NO2_out, CO_out, CO2_out, Car_Vel_kmph, SOC_1)
