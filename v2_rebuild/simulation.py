"""
Simulation wrapper for ICE and PG models.

This module provides a clean wrapper around the transition_function_model
from the existing codebase, handling LSTM hidden states cleanly.
"""
import sys
import os
from typing import Tuple, Dict

# Add the RL model path to system path
# Note: This should be made configurable in production
RL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'controller_for_ICE_PG',
    'reinforcement_learning_model'
)
sys.path.insert(0, RL_MODEL_PATH)

from transition_function_model import setup_transition_function_model


class SimulationModel:
    """
    Wrapper class for ICE and PG models with clean state management.
    
    This class wraps the existing transition_function_model and provides
    a clean interface for predictions while managing LSTM hidden states.
    """
    
    def __init__(self, ice_dir: str, pg_dir: str, initial_soc: float = 0.7):
        """
        Initialize the simulation models.
        
        Args:
            ice_dir: Directory containing ICE model.h5 and scalers
            pg_dir: Directory containing PG model.h5 and scalers
            initial_soc: Initial state of charge (default 0.7)
        """
        self.ice_dir = ice_dir
        self.pg_dir = pg_dir
        self.initial_soc = initial_soc
        
        # Create the transition function model
        self.model = setup_transition_function_model(
            ruta_ICE=ice_dir,
            ruta_PG=pg_dir,
            SOC_ini=initial_soc
        )
    
    def reset_states(self):
        """Reset LSTM hidden states to initial values."""
        self.model.reset_models()
    
    def predict_ice(
        self,
        speed_rpm: float,
        fuel_mg: float,
        temp_K: float = 298.0,
        pressure_bar: float = 1.0
    ) -> Tuple[float, float, float, float, float]:
        """
        Predict ICE outputs for one timestep.
        
        Args:
            speed_rpm: Engine speed in RPM
            fuel_mg: Fuel mass in mg
            temp_K: Ambient temperature in Kelvin
            pressure_bar: Ambient pressure in bar
            
        Returns:
            Tuple of (torque_Nm, NO, NO2, CO, CO2)
        """
        torque, no, no2, co, co2 = self.model.predict_ice(
            speed_rpm, fuel_mg, temp_K, pressure_bar
        )
        
        return (
            float(torque.numpy()),
            float(no.numpy()),
            float(no2.numpy()),
            float(co.numpy()),
            float(co2.numpy())
        )
    
    def predict_pg(
        self,
        ice_speed_rpm: float,
        em2_torque_Nm: float,
        ice_torque_Nm: float,
        brake_perc: float
    ) -> Tuple[float, float]:
        """
        Predict PG outputs for one timestep.
        
        Args:
            ice_speed_rpm: ICE speed in RPM
            em2_torque_Nm: EM2 torque in Nm
            ice_torque_Nm: ICE torque in Nm (from predict_ice)
            brake_perc: Brake percentage
            
        Returns:
            Tuple of (car_speed_kmph, soc)
        """
        car_speed, soc = self.model.predict_PG(
            ice_speed_rpm, em2_torque_Nm, ice_torque_Nm, brake_perc
        )
        
        return float(car_speed.numpy()), float(soc.numpy())
    
    def step(
        self,
        ice_speed_rpm: float,
        fuel_mg: float,
        em2_torque_Nm: float = 0.0,
        brake_perc: float = 0.0,
        temp_K: float = 298.0,
        pressure_bar: float = 1.0
    ) -> Dict[str, float]:
        """
        Execute one complete simulation step (ICE + PG).
        
        Args:
            ice_speed_rpm: ICE speed in RPM
            fuel_mg: Fuel mass in mg
            em2_torque_Nm: EM2 torque in Nm
            brake_perc: Brake percentage
            temp_K: Ambient temperature in Kelvin
            pressure_bar: Ambient pressure in bar
            
        Returns:
            Dictionary containing all outputs
        """
        # Predict ICE
        torque, no, no2, co, co2 = self.predict_ice(
            ice_speed_rpm, fuel_mg, temp_K, pressure_bar
        )
        
        # Predict PG
        car_speed, soc = self.predict_pg(
            ice_speed_rpm, em2_torque_Nm, torque, brake_perc
        )
        
        return {
            'ice_torque': torque,
            'no': no,
            'no2': no2,
            'co': co,
            'co2': co2,
            'car_speed': car_speed,
            'soc': soc
        }


def create_simulation_from_directory(
    ice_dir: str,
    pg_dir: str,
    initial_soc: float = 0.7
) -> SimulationModel:
    """
    Create a SimulationModel from model directories.
    
    Args:
        ice_dir: Directory containing ICE model.h5 and scalers
        pg_dir: Directory containing PG model.h5 and scalers
        initial_soc: Initial state of charge
        
    Returns:
        SimulationModel instance
    """
    return SimulationModel(ice_dir=ice_dir, pg_dir=pg_dir, initial_soc=initial_soc)
