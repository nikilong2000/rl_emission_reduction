"""
Neural network architectures for TD3 with recurrent critic.

This module implements:
1. Actor network: LSTM-based policy network
2. RecurrentCritic network: LSTM-based Q-value network

The networks are designed for sequences to support Burn-in + Unroll training.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


# Constants for observation space structure
OBS_DIM_WITHOUT_ERROR = 5  # First 5 dimensions: vel_target, vel, mf, brk, ice_sp
ERROR_DIM_INDEX = 5  # Error is at index 5 (6th dimension)


class Actor(nn.Module):
    """
    Actor network with LSTM for sequential decision making.
    
    Input: Observation (6 dims: vel_target, vel, mf, brk, ice_sp, error)
    Output: Action (3 dims: delta_mf, delta_brk, delta_ice_sp) in range [-1, 1]
    """
    
    def __init__(
        self,
        obs_dim: int = 6,
        action_dim: int = 3,
        hidden_dim: int = 128,
        scaler_params: Optional[dict] = None
    ):
        """
        Initialize Actor network.
        
        Args:
            obs_dim: Observation dimension (default 6)
            action_dim: Action dimension (default 3)
            hidden_dim: LSTM hidden dimension
            scaler_params: Optional scaler parameters for input normalization
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input normalization (if scaler params provided)
        if scaler_params is not None:
            self.register_buffer(
                "_scale",
                torch.tensor(scaler_params["scale"], dtype=torch.float32)
            )
            self.register_buffer(
                "_min",
                torch.tensor(scaler_params["min"], dtype=torch.float32)
            )
            self.use_scaler = True
        else:
            self.use_scaler = False
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output layers
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Conservative initialization to prevent saturation
        nn.init.uniform_(self.action_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.action_head.bias, -3e-3, 3e-3)
        
        # Hidden state for inference
        self.hidden_state = None
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input using scaler parameters.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        if not self.use_scaler:
            return x
        
        # Normalize first 5 dimensions, keep error (6th) as is
        x_first5 = x[..., :OBS_DIM_WITHOUT_ERROR] * self._scale + self._min
        x_error = x[..., ERROR_DIM_INDEX:ERROR_DIM_INDEX+1]  # Error already normalized
        return torch.cat([x_first5, x_error], dim=-1)
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state."""
        self.hidden_state = None
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through Actor.
        
        Args:
            obs: Observations
                - For training: shape (batch, seq_len, obs_dim)
                - For inference: shape (batch, obs_dim)
            hidden_state: Optional LSTM hidden state
            
        Returns:
            Tuple of (actions, new_hidden_state)
            - actions: shape (..., action_dim) in range [-1, 1]
            - new_hidden_state: Updated hidden state (or None)
        """
        # Normalize input
        if self.use_scaler:
            obs = self._normalize_input(obs)
        
        # Check if this is inference (2D) or training (3D)
        is_inference = obs.dim() == 2
        
        if is_inference:
            # Inference mode: process single timestep
            obs_step = obs.unsqueeze(1)  # (batch, 1, obs_dim)
            lstm_out, self.hidden_state = self.lstm(obs_step, self.hidden_state)
            lstm_out = lstm_out.squeeze(1)  # (batch, hidden_dim)
            new_hidden_state = self.hidden_state
        else:
            # Training mode: process sequence
            lstm_out, new_hidden_state = self.lstm(obs, hidden_state)
            # lstm_out: (batch, seq_len, hidden_dim)
        
        # Process LSTM output
        h = self.layer_norm(lstm_out)
        h = self.activation(self.fc(h))
        
        # Generate actions
        actions = torch.tanh(self.action_head(h))  # Range [-1, 1]
        
        return actions, new_hidden_state


class RecurrentCritic(nn.Module):
    """
    Recurrent Critic network for TD3.
    
    Input: Concatenated (observation, action)
    Output: Q-value estimate
    """
    
    def __init__(
        self,
        obs_dim: int = 6,
        action_dim: int = 3,
        hidden_dim: int = 128,
        scaler_params: Optional[dict] = None
    ):
        """
        Initialize Recurrent Critic.
        
        Args:
            obs_dim: Observation dimension (default 6)
            action_dim: Action dimension (default 3)
            hidden_dim: LSTM hidden dimension
            scaler_params: Optional scaler parameters for input normalization
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input normalization (if scaler params provided)
        if scaler_params is not None:
            self.register_buffer(
                "_scale",
                torch.tensor(scaler_params["scale"], dtype=torch.float32)
            )
            self.register_buffer(
                "_min",
                torch.tensor(scaler_params["min"], dtype=torch.float32)
            )
            self.use_scaler = True
        else:
            self.use_scaler = False
        
        # LSTM processes concatenated obs + action
        self.lstm = nn.LSTM(
            input_size=obs_dim + action_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Q-network head
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Hidden state for inference
        self.hidden_state = None
    
    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Normalize observations using scaler parameters.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Normalized observations
        """
        if not self.use_scaler:
            return obs
        
        # Normalize first 5 dimensions, keep error (6th) as is
        obs_first5 = obs[..., :OBS_DIM_WITHOUT_ERROR] * self._scale + self._min
        obs_error = obs[..., ERROR_DIM_INDEX:ERROR_DIM_INDEX+1]  # Error already normalized
        return torch.cat([obs_first5, obs_error], dim=-1)
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state."""
        self.hidden_state = None
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through Critic.
        
        Args:
            obs: Observations
                - For training: shape (batch, seq_len, obs_dim)
                - For inference: shape (batch, obs_dim)
            action: Actions
                - For training: shape (batch, seq_len, action_dim)
                - For inference: shape (batch, action_dim)
            hidden_state: Optional LSTM hidden state
            
        Returns:
            Tuple of (q_values, new_hidden_state)
            - q_values: Q-value estimates
            - new_hidden_state: Updated hidden state (or None)
        """
        # Normalize observations
        if self.use_scaler:
            obs = self._normalize_obs(obs)
        
        # Concatenate obs and action before LSTM
        lstm_input = torch.cat([obs, action], dim=-1)
        
        # Check if this is inference (2D) or training (3D)
        is_inference = obs.dim() == 2
        
        if is_inference:
            # Inference mode: process single timestep
            lstm_input_step = lstm_input.unsqueeze(1)  # (batch, 1, obs_dim + action_dim)
            lstm_out, self.hidden_state = self.lstm(lstm_input_step, self.hidden_state)
            lstm_out = lstm_out.squeeze(1)  # (batch, hidden_dim)
            new_hidden_state = self.hidden_state
        else:
            # Training mode: process sequence
            lstm_out, new_hidden_state = self.lstm(lstm_input, hidden_state)
            # lstm_out: (batch, seq_len, hidden_dim)
        
        # Process LSTM output
        h_norm = self.layer_norm(lstm_out)
        
        # Compute Q-value
        q_value = self.q_network(h_norm)
        
        return q_value, new_hidden_state


def copy_network_parameters(target: nn.Module, source: nn.Module):
    """
    Copy parameters from source network to target network.
    
    Args:
        target: Target network
        source: Source network
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


def soft_update_network(target: nn.Module, source: nn.Module, tau: float):
    """
    Soft update target network parameters.
    
    target = tau * source + (1 - tau) * target
    
    Args:
        target: Target network
        source: Source network
        tau: Soft update coefficient
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )
