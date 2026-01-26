"""
Replay buffer for sequence-based RL training.

This buffer stores sequences (windows) of transitions for Burn-in + Unroll training.
"""
import numpy as np
import random
from typing import Tuple, List


class SequenceReplayBuffer:
    """
    Replay buffer that stores windows of trajectories for recurrent RL.
    
    Each window contains:
    - observations: (seq_len, obs_dim)
    - actions: (seq_len, action_dim)
    - rewards: (seq_len,)
    - next_observations: (seq_len, obs_dim)
    - terminated: (seq_len,)
    - truncated: (seq_len,)
    """
    
    def __init__(
        self,
        capacity: int,
        seq_len: int,
        obs_dim: int,
        action_dim: int
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of windows to store
            seq_len: Length of each sequence window
            obs_dim: Observation dimension
            action_dim: Action dimension
        """
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.buffer: List[Tuple] = []
        self.position = 0
        
        # Define expected shapes for validation
        self._expected_shapes = (
            (seq_len, obs_dim),      # observations
            (seq_len, action_dim),   # actions
            (seq_len,),              # rewards
            (seq_len, obs_dim),      # next_observations
            (seq_len,),              # terminated
            (seq_len,)               # truncated
        )
    
    def _validate_window_shapes(self, window: Tuple) -> None:
        """
        Validate that a window has the expected shapes.
        
        Args:
            window: Tuple of (obs, actions, rewards, next_obs, terminated, truncated)
            
        Raises:
            ValueError: If shapes don't match expected
        """
        current_shapes = tuple(np.asarray(item).shape for item in window)
        
        if current_shapes != self._expected_shapes:
            raise ValueError(
                f"Window shape mismatch!\n"
                f"  Expected: {self._expected_shapes}\n"
                f"  Received: {current_shapes}"
            )
    
    def add(self, window: Tuple) -> None:
        """
        Add a sequence window to the buffer.
        
        Args:
            window: Tuple of (observations, actions, rewards, next_obs, terminated, truncated)
                Each element should be a numpy array with shape matching _expected_shapes
        """
        # Validate shapes
        self._validate_window_shapes(window)
        
        # Add to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = window
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a batch of sequence windows.
        
        Args:
            batch_size: Number of windows to sample
            
        Returns:
            Tuple of stacked arrays:
            - observations: (batch_size, seq_len, obs_dim)
            - actions: (batch_size, seq_len, action_dim)
            - rewards: (batch_size, seq_len)
            - next_observations: (batch_size, seq_len, obs_dim)
            - terminated: (batch_size, seq_len)
            - truncated: (batch_size, seq_len)
        """
        # Sample random windows
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        
        # Transpose and stack
        # batch is list of windows, each window is tuple of arrays
        # zip(*batch) groups all observations together, all actions together, etc.
        columns = list(zip(*batch))
        
        # Stack each column
        stacked = tuple(np.stack(col) for col in columns)
        
        return stacked
    
    def __len__(self) -> int:
        """Return number of windows in buffer."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough samples for training.
        
        Args:
            min_size: Minimum number of windows required
            
        Returns:
            True if buffer has at least min_size windows
        """
        return len(self.buffer) >= min_size


class EpisodeBuffer:
    """
    Temporary buffer to collect a full episode before extracting windows.
    """
    
    def __init__(self, obs_dim: int, action_dim: int):
        """
        Initialize episode buffer.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reset()
    
    def reset(self):
        """Reset the episode buffer."""
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_observations: List[np.ndarray] = []
        self.terminated: List[bool] = []
        self.truncated: List[bool] = []
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool
    ):
        """
        Add a single transition to the episode buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
    
    def extract_windows(self, seq_len: int, stride: int = 1) -> List[Tuple]:
        """
        Extract overlapping sequence windows from the episode.
        
        Args:
            seq_len: Length of each sequence window
            stride: Step size between windows (default 1 for maximum overlap)
            
        Returns:
            List of windows, each a tuple of numpy arrays
        """
        episode_len = len(self.observations)
        
        if episode_len < seq_len:
            # Episode too short, return empty list
            return []
        
        windows = []
        
        # Extract windows with stride
        for start_idx in range(0, episode_len - seq_len + 1, stride):
            end_idx = start_idx + seq_len
            
            window = (
                np.array(self.observations[start_idx:end_idx]),
                np.array(self.actions[start_idx:end_idx]),
                np.array(self.rewards[start_idx:end_idx]),
                np.array(self.next_observations[start_idx:end_idx]),
                np.array(self.terminated[start_idx:end_idx]),
                np.array(self.truncated[start_idx:end_idx])
            )
            
            windows.append(window)
        
        return windows
    
    def __len__(self) -> int:
        """Return number of transitions in episode."""
        return len(self.observations)
