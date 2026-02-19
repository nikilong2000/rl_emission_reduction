import os
import tensorflow as tf
import numpy as np

# Try importing the ONNX helper classes. 
# Make sure ONNX_Predict is in your python path.
try:
    from ONNX_Predict.LSTM_onnx import LSTM_onnx
    from ONNX_Predict.Scaler_onnx import Scaler_onnx
except ImportError:
    print("Warning: Could not import ONNX_Predict. Make sure it is installed or in PYTHONPATH.")
    # Dummy classes for structure if import fails (will crash at runtime if used)
    LSTM_onnx = None
    Scaler_onnx = None

def setup_transition_function_model(ruta_ICE, ruta_PG, SOC_ini=0.7):
    """
    Sets up the ONNX transition model.
    ruta_ICE/PG should be the folder containing .onnx files and model.h5 (for aux info).
    """
    return transition_function_model(ruta_ICE, ruta_PG, SOC_ini)

class transition_function_model:
    def __init__(self, ice_folder, pg_folder, SOC_ini=0.7, tf_model_name='model.h5'):
        
        self.ice_folder = ice_folder
        self.pg_folder = pg_folder
        
        # Load ICE components
        self.ICEscaler_in = Scaler_onnx('scaler_input.onnx', ice_folder)
        self.ICEscaler_out = Scaler_onnx('scaler_output.onnx', ice_folder) # Used for aux state gen
        self.ICEscaler_inv_out = Scaler_onnx('scaler_inverse_output.onnx', ice_folder) # Used for transforming output back
        self.ICE_model = LSTM_onnx('ICE_onnx.onnx', ice_folder, tf_model_name)
        
        # Load PG components
        self.PGscaler_in = Scaler_onnx('scaler_input.onnx', pg_folder)
        self.PGscaler_out = Scaler_onnx('scaler_output.onnx', pg_folder)
        self.PGscaler_inv_out = Scaler_onnx('scaler_inverse_output.onnx', pg_folder)
        self.PG_model = LSTM_onnx('PG_onnx.onnx', pg_folder, tf_model_name)
        
        # Initialize Aux variables
        # Default initial values from original code
        self.Torque = 0
        self.NO_ini = 0
        self.NO2_ini = 0
        self.CO_ini = 0
        self.CO2_ini = 0
        self.velocity_ini = 0
        self.SOC_ini = SOC_ini
        
        self.reset_models()
        self.results = []

    def reset_models(self):
        # Reset ONNX internal states
        self.ICE_model.reset_states()
        self.PG_model.reset_states()
        
        # Reset Aux values (scaled)
        # ICE Aux: [Torque, NO, NO2, CO, CO2]
        y_ini_ice = np.array([[self.Torque, self.NO_ini, self.NO2_ini, self.CO_ini, self.CO2_ini]], dtype='float32')
        self.ice_aux = self.ICEscaler_out.transform(y_ini_ice)[0].reshape((1,1,5))
        
        # PG Aux: [velocity, SOC]
        y_ini_pg = np.array([[self.velocity_ini, self.SOC_ini]], dtype='float32')
        self.pg_aux = self.PGscaler_out.transform(y_ini_pg)[0].reshape((1,1,2))
        
        self.results = []

    def predict_ice(self, Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar):
        # Determine if inputs are tensors strings or numpy
        # If tensors, convert to numpy
        if tf.is_tensor(Speed_rpm):
            Speed_rpm = Speed_rpm.numpy()
            m_fuel_mg = m_fuel_mg.numpy()
            T_amb_K = T_amb_K.numpy()
            p_amb_bar = p_amb_bar.numpy()
            
        # 1. Pack input [Speed, Fuel, T, p]
        # Ensure shape matches (1, 4) or similar? 
        # Notebook: x = ICEinput.iloc[:1].copy().to_numpy() -> shape (1, 4)
        # Inputs here might be scalars.
        x = np.array([Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar], dtype='float32')
        if x.ndim == 1:
            x = x.reshape(1, 4)
        
        # 2. Scale
        x_scaled = self.ICEscaler_in.transform(x)
        # Verify if x_scaled is array, if not convert. 
        # Scaler_onnx might return a list or different type depending on implementation.
        x_scaled = np.array(x_scaled) 
        x_scaled = np.reshape(x_scaled, (1, 1, 4))
        
        # 3. Predict
        # Input: [x_scaled, y_scaled_ini]
        # Note: In the notebook, y_scaled_ini is passed every time. 
        # Does the model update internally? Yes, reset_states() is used.
        y_scaled = self.ICE_model([x_scaled, self.ice_aux])
        # y_scaled shape: (1, 1, 5) ? Notebook says: y_predict_scaled[0][0][0]
        
        # 4. Inverse Scale
        # Notebook: ICEscaler_inv_out.transform(y_predict_scaled_list[pred].reshape(1, -1))
        y_flat = y_scaled.reshape(1, -1)
        y = self.ICEscaler_inv_out.transform(y_flat)
        
        # y is [[Torque, NO, NO2, CO, CO2]]
        outputs = y[0]
        
        # Return as Tensors (to maintain compatibility with step.py if needed, 
        # though gradients won't flow)
        return (tf.convert_to_tensor(outputs[0], dtype=tf.float32), 
                tf.convert_to_tensor(outputs[1], dtype=tf.float32), 
                tf.convert_to_tensor(outputs[2], dtype=tf.float32), 
                tf.convert_to_tensor(outputs[3], dtype=tf.float32), 
                tf.convert_to_tensor(outputs[4], dtype=tf.float32))

    def predict_PG(self, ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc):
        if tf.is_tensor(ICE_Speed_soll_rpm):
            ICE_Speed_soll_rpm = ICE_Speed_soll_rpm.numpy()
            EM2_Torque_Nm = EM2_Torque_Nm.numpy()
            ICE_Torque_Nm = ICE_Torque_Nm.numpy()
            Brake_perc = Brake_perc.numpy()
            
        # [Speed_soll, EM2_T, ICE_T, Brake]
        x = np.array([ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc], dtype='float32')
        if x.ndim == 1:
            x = x.reshape(1, 4)
            
        x_scaled = self.PGscaler_in.transform(x)
        x_scaled = x_scaled.reshape(1, 1, 4)
        
        y_scaled = self.PG_model([x_scaled, self.pg_aux])
        
        y_flat = y_scaled.reshape(1, -1)
        y = self.PGscaler_inv_out.transform(y_flat)
        
        # y is [[Car_Speed, SOC]]
        outputs = y[0]
        
        return (tf.convert_to_tensor(outputs[0], dtype=tf.float32), 
                tf.convert_to_tensor(outputs[1], dtype=tf.float32))

    def predict(self, Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar, ICE_Speed_soll_rpm, Brake_perc, EM2_Torque_Nm):
        
        ICE_Torque_Nm, NO, NO2, CO, CO2 = self.predict_ice(Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar)
        
        Car_Speed, SOC = self.predict_PG(ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc)
        
        # Store results for history (optional)
        # Convert tensors to float if needed
        self.results.append([
            float(ICE_Torque_Nm), float(NO), float(NO2), float(CO), float(CO2),
            float(Car_Speed), float(SOC)
        ])
        
        return ICE_Torque_Nm, NO, NO2, CO, CO2, Car_Speed, SOC
