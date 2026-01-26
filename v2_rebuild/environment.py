"""
Environment implementation for RL controller.

This module implements a Gymnasium-compatible environment for the vehicle control task.
The key fix is to include the Error in the observation space (6 dimensions total).
"""
import numpy as np
import sys
import os

# Add the old RL model path to system path for importing transition model
sys.path.insert(0, '/home/runner/work/rl_emission_reduction/rl_emission_reduction/controller_for_ICE_PG/reinforcement_learning_model')

from transition_function_model import setup_transition_function_model
from typing import Tuple, Dict, Optional
import random
import csv


class VehicleControlEnvironment:
    """
    Gymnasium-style environment for vehicle control.
    
    **Key Fix**: Observation space now includes Error (6 dimensions):
    - vel_target: Target velocity
    - vel: Current velocity  
    - mf: Fuel mass
    - brk: Brake percentage
    - ice_sp: ICE speed
    - error: vel_target - vel (ADDED TO FIX INPUT MISMATCH)
    
    Action space (3 dimensions):
    - delta_mf: Change in fuel mass
    - delta_brk: Change in brake
    - delta_ice_sp: Change in ICE speed
    """
    
    def __init__(
        self,
        ice_model_dir: str,
        pg_model_dir: str,
        data_dir: str = None,
        max_steps: int = 1200,
        alpha: float = 1.0,
        stable_error_threshold_kmh: float = 0.5,
        n_stable_steps: int = 50,
        initial_soc: float = 0.7
    ):
        """
        Initialize the environment.
        
        Args:
            ice_model_dir: Directory containing ICE model and scalers
            pg_model_dir: Directory containing PG model and scalers
            data_dir: Directory containing CSV files for initial state sampling
            max_steps: Maximum steps per episode
            alpha: Reward coefficient for velocity error
            stable_error_threshold_kmh: Error threshold for considering velocity stable (km/h)
            n_stable_steps: Number of consecutive stable steps to terminate episode
            initial_soc: Initial state of charge
        """
        # Create transition function model
        self.transition_model = setup_transition_function_model(
            ruta_ICE=ice_model_dir,
            ruta_PG=pg_model_dir,
            SOC_ini=initial_soc
        )
        
        # Environment parameters
        self.max_steps = max_steps
        self.alpha = alpha
        self.n_stable_steps = n_stable_steps
        self.data_dir = data_dir
        
        # Maximum possible error for normalization
        self.MAX_ERROR = 80.0  # km/h
        
        # Convert stable error threshold to reward threshold
        norm_stable_error = stable_error_threshold_kmh / self.MAX_ERROR
        self.stable_reward_threshold = 1.0 - (norm_stable_error ** 2)
        
        # Ambient conditions
        self.T_amb_K = 298.0
        self.p_amb_bar = 1.0
        
        # Episode state
        self.step_count = 0
        self.stable_counter = 0
        
        # Current state variables
        self.vel = 0.0
        self.mf = 0.0
        self.brk = 0.0
        self.ice_sp = 0.0
        self.vel_target = 70.0
        
    def _compute_error(self, vel_target: float) -> float:
        """Compute velocity error."""
        return vel_target - self.vel
    
    def _compute_reward(self, vel_target: float) -> float:
        """
        Compute reward based on normalized squared error.
        
        Reward is in range [0, 1]:
        - 1.0 = zero error (perfect)
        - 0.0 = maximum error
        """
        error = abs(self._compute_error(vel_target))
        normalized_error = min(error / self.MAX_ERROR, 1.0)
        reward = 1.0 - (normalized_error ** 2)
        return reward
    
    def _get_observation(self, vel_target: float) -> np.ndarray:
        """
        Get current observation.
        
        **KEY FIX**: Observation now includes error as 6th dimension.
        
        Returns:
            numpy array of shape (6,) with [vel_target, vel, mf, brk, ice_sp, error]
        """
        error = self._compute_error(vel_target)
        return np.array([
            vel_target,
            self.vel,
            self.mf,
            self.brk,
            self.ice_sp,
            error  # <-- ADDED: 6th dimension
        ], dtype=np.float32)
    
    def reset(self, vel_target: float = 70.0) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Args:
            vel_target: Target velocity for the episode
            
        Returns:
            Initial observation (6 dimensions including error)
        """
        # Reset transition model
        self.transition_model.reset_models()
        
        # Sample initial state
        self._sample_initial_state()
        
        # Compute initial velocity using PG model
        vel_tf, _ = self.transition_model.predict_PG(
            self.ice_sp, 0.0, self.torque_ICE, self.brk
        )
        self.vel = float(vel_tf.numpy())
        
        # Reset transition model again (since we did a PG prediction)
        self.transition_model.reset_models()
        
        # Reset counters
        self.step_count = 0
        self.stable_counter = 0
        self.vel_target = vel_target
        
        return self._get_observation(vel_target)
    
    def _sample_initial_state(self):
        """
        Sample an initial state from CSV files or use defaults.
        
        Uses reservoir sampling to randomly select a row from a random CSV file.
        """
        if self.data_dir is None or not os.path.exists(self.data_dir):
            # Use default initial state
            self.mf = 10.0
            self.brk = 0.0
            self.ice_sp = 1500.0
            self.torque_ICE = 0.0
            return
        
        # List CSV files
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        if not csv_files:
            # Use default if no CSV files found
            self.mf = 10.0
            self.brk = 0.0
            self.ice_sp = 1500.0
            self.torque_ICE = 0.0
            return
        
        # Choose random CSV file
        chosen_file = os.path.join(self.data_dir, random.choice(csv_files))
        
        # Reservoir sampling: choose 1 random row
        chosen_row = None
        with open(chosen_file, newline='') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                if random.random() < 1 / i:
                    chosen_row = row
        
        if chosen_row is None:
            # Use default if file is empty
            self.mf = 10.0
            self.brk = 0.0
            self.ice_sp = 1500.0
            self.torque_ICE = 0.0
            return
        
        # Set state from chosen row
        self.mf = float(chosen_row.get('fuel', 10.0))
        self.brk = float(chosen_row.get('Brake', 0.0))
        self.ice_sp = float(chosen_row.get('ICE_Speed_soll', 1500.0))
        self.torque_ICE = float(chosen_row.get('ICE_Torque_pred', 0.0))
    
    def step(
        self,
        action: np.ndarray,
        vel_target: Optional[float] = None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action array [delta_mf, delta_brk, delta_ice_sp]
            vel_target: Target velocity (uses self.vel_target if None)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if vel_target is None:
            vel_target = self.vel_target
        
        # Apply action
        delta_mf, delta_brk, delta_ice_sp = action
        self.mf = float(delta_mf)
        self.brk = float(delta_brk)
        self.ice_sp = float(delta_ice_sp)
        
        # Predict ICE output
        torque_tf, nox_tf, _, co_tf, _ = self.transition_model.predict_ice(
            self.ice_sp, self.mf, self.T_amb_K, self.p_amb_bar
        )
        
        torque_ICE = float(torque_tf.numpy())
        nox = float(nox_tf.numpy())
        co = float(co_tf.numpy())
        
        # Clip ICE torque
        torque_ICE = np.clip(torque_ICE, -50.0, 300.0)
        
        # Turn off ICE if speed is too low
        if self.ice_sp < 900.0:
            torque_ICE = 0.0
            self.mf = 3.0
        
        # Predict PG output
        vel_tf, soc_tf = self.transition_model.predict_PG(
            self.ice_sp, 0.0, torque_ICE, self.brk
        )
        
        self.vel = float(vel_tf.numpy())
        soc = float(soc_tf.numpy())
        
        # Compute reward
        reward = self._compute_reward(vel_target)
        
        # Check stability
        if reward >= self.stable_reward_threshold:
            self.stable_counter += 1
        else:
            self.stable_counter = 0
        
        terminated = self.stable_counter >= self.n_stable_steps
        
        # Check truncation
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        
        # Get new observation (6 dimensions including error)
        observation = self._get_observation(vel_target)
        
        # Info dict
        info = {
            'vel': self.vel,
            'torque_ICE': torque_ICE,
            'nox': nox,
            'co': co,
            'soc': soc,
            'error': self._compute_error(vel_target)
        }
        
        return observation, reward, terminated, truncated, info
    
    @property
    def observation_space_dim(self) -> int:
        """Dimension of observation space (6)."""
        return 6  # vel_target, vel, mf, brk, ice_sp, error
    
    @property
    def action_space_dim(self) -> int:
        """Dimension of action space (3)."""
        return 3  # delta_mf, delta_brk, delta_ice_sp
