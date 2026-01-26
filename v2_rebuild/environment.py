"""
Gymnasium-compatible environment for vehicle control.
FIXED: Includes Error (6D) in observation space to match Agent expectations.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces


class VehicleEnvironment(gym.Env):
    """
    Gymnasium-compatible environment for hybrid vehicle control.

    Observation space includes:
    - vel_target (1D)
    - velocity (1D)
    - mf, brk, ice_sp (3D) - previous actions
    - error (1D) - normalized velocity error
    Total: 6 dimensions

    Action space: [mf, brk, ice_sp] ∈ [-1, 1]³
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulation,
        max_steps: int = 1200,
        vel_target: float = 70.0,
        alpha: float = 1.0,
        stable_error_threshold: float = 0.5,
        n_stable: int = 50,
        max_error: float = 80.0,
    ):
        """
        Initialize the environment.

        Args:
            simulation: Simulation instance (ICE/PG models)
            max_steps: Maximum steps per episode
            vel_target: Target velocity [km/h]
            alpha: Reward coefficient for velocity error
            stable_error_threshold: Error threshold for early termination [km/h]
            n_stable: Number of stable steps required for early termination
            max_error: Maximum possible velocity error [km/h]
        """
        super().__init__()

        self.simulation = simulation
        self.max_steps = max_steps
        self.vel_target = vel_target
        self.alpha = alpha
        self.max_error = max_error

        # Stability criteria
        self.stable_error_threshold = stable_error_threshold
        self.n_stable = n_stable

        # Calculate reward threshold for stability
        norm_stable_error = stable_error_threshold / max_error
        self.stable_reward_threshold = 1.0 - (norm_stable_error**2)

        # Define action space: [mf, brk, ice_sp] ∈ [-1, 1]³
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Define observation space: [vel_target, velocity, mf, brk, ice_sp, error] (6D)
        # All components normalized to approximately [-1, 1] or [0, 1]
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(6,), dtype=np.float32  # Conservative bounds
        )

        # Episode state
        self.step_count = 0
        self.stable_counter = 0
        self.prev_action = np.zeros(3, dtype=np.float32)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Returns:
            observation: Initial observation (6D)
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Reset simulation
        sim_state = self.simulation.reset()

        # Reset episode variables
        self.step_count = 0
        self.stable_counter = 0
        self.prev_action = np.zeros(3, dtype=np.float32)

        # Get initial observation
        observation = self._get_observation(sim_state)

        info = {
            "velocity": float(sim_state["velocity"][0]),
            "soc": float(sim_state["soc"][0]),
            "error": 0.0,
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: Action array [mf, brk, ice_sp] ∈ [-1, 1]³

        Returns:
            observation: New observation (6D)
            reward: Reward value
            terminated: Whether episode ended naturally (stability reached)
            truncated: Whether episode was cut off (max steps)
            info: Additional information
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Execute simulation step
        sim_state = self.simulation.step(
            mf=float(action[0]), brk=float(action[1]), ice_sp=float(action[2])
        )

        # Extract state variables
        velocity = float(sim_state["velocity"][0])
        soc = float(sim_state["soc"][0])

        # Calculate error
        error_raw = self.vel_target - velocity
        error_norm = error_raw / self.max_error

        # Calculate reward (negative squared normalized error)
        reward = self.alpha * (1.0 - error_norm**2)

        # Check stability
        if reward >= self.stable_reward_threshold:
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        # Update step count
        self.step_count += 1

        # Check termination conditions
        terminated = self.stable_counter >= self.n_stable
        truncated = self.step_count >= self.max_steps

        # Get new observation
        observation = self._get_observation(sim_state, action)

        # Store action for next step
        self.prev_action = action.copy()

        # Info dictionary
        info = {
            "velocity": velocity,
            "soc": soc,
            "error": error_raw,
            "error_norm": error_norm,
            "reward": reward,
            "stable_counter": self.stable_counter,
            "step": self.step_count,
            "torque": float(sim_state["torque"][0]),
            "emissions": {
                "no": float(sim_state["no"][0]),
                "no2": float(sim_state["no2"][0]),
                "co": float(sim_state["co"][0]),
                "co2": float(sim_state["co2"][0]),
            },
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(
        self, sim_state: Dict[str, np.ndarray], action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Construct observation from simulation state.

        Observation components (6D):
        - vel_target: normalized [0, 1] (assuming 0-200 km/h range)
        - velocity: normalized [0, 1]
        - mf, brk, ice_sp: previous actions [-1, 1]
        - error: normalized error [-1, 1]

        Args:
            sim_state: Simulation state dictionary
            action: Current action (if None, use previous action)

        Returns:
            observation: 6D observation array
        """
        if action is None:
            action = self.prev_action

        velocity = float(sim_state["velocity"][0])

        # Normalize velocity and target to [0, 1] range (assuming 0-200 km/h)
        vel_target_norm = self.vel_target / 200.0
        velocity_norm = velocity / 200.0

        # Calculate normalized error
        error_raw = self.vel_target - velocity
        error_norm = error_raw / self.max_error  # [-1, 1]

        # Construct observation [vel_target, velocity, mf, brk, ice_sp, error]
        observation = np.array(
            [
                vel_target_norm,
                velocity_norm,
                action[0],  # mf
                action[1],  # brk
                action[2],  # ice_sp
                error_norm,
            ],
            dtype=np.float32,
        )

        return observation
