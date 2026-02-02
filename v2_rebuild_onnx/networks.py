"""
Neural network architectures for TD3 agent.
Actor and RecurrentCritic networks with LSTM for temporal dependencies.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


class ActorNetwork(nn.Module):
    """
    LSTM-based Actor network for continuous control.

    Input: 7D observation [vel_target, velocity, mf, brk, ice_sp, em2_torque, error]
    Output: 4D action [mf, brk, ice_sp, em2_torque] ∈ [-1, 1]
    """

    def __init__(
        self,
        scaler_params: Dict,
        obs_dim: int = 7,
        action_dim: int = 4,
        hidden_size: int = 128,
    ):
        """
        Initialize the Actor network.

        Args:
            scaler_params: Dictionary with normalization parameters (data_min, data_max, scale, min)
            obs_dim: Observation dimension (7)
            action_dim: Action dimension (4)
            hidden_size: LSTM hidden size
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # Register normalization buffers
        self._register_scaler_buffers(scaler_params)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=obs_dim, hidden_size=hidden_size, batch_first=True
        )

        # Layer normalization for stability
        self.layernorm = nn.LayerNorm(hidden_size)

        # Output layers
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, action_dim)

        # Conservative initialization to prevent saturation
        nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.output_layer.bias, -3e-3, 3e-3)

        # Hidden state for inference
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _register_scaler_buffers(self, scaler_params: Dict):
        """Register scaler parameters as buffers."""
        scale = torch.tensor(scaler_params.get("scale", [1.0] * 5), dtype=torch.float32)
        min_ = torch.tensor(scaler_params.get("min", [0.0] * 5), dtype=torch.float32)
        self.register_buffer("_scale", scale)
        self.register_buffer("_min", min_)

    def _normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize observation.
        First 5 dims scaled, next dims (e.g. em2_torque, error) passed through.
        
        Note on Dimensions for 7D observation:
        Indices 0-4 (5 dims): [vel_target, velocity, mf, brk, ice_sp] - handled by scaler params?
        Wait, scaler params are from PG model input scaler which has 4 dims?
        
        Actually, the observation construction in Environment.py is already normalized to roughly [-1,1] or [0,1].
        If we trust the environment's normalization, we might skip complex re-normalization here OR
        we need to ensure scaler_params aligns with the observation structure.
        
        The original code used `scaler_params` derived from `pg_in_scaler` which likely expects 4 inputs:
        [Speed_rpm, EM2_Torque, ICE_Torque, Brake_perc]. 
        But the observation is: [vel_target, velocity, mf, brk, ice_sp, em2_torque, error].
        These don't match index-for-index.
        
        Given the environment already normalizes Inputs:
        - vel_target_norm (0-1)
        - velocity_norm (0-1)
        - actions (-1 to 1)
        - error (-1 to 1)
        
        It is safer to modify this to assume inputs are already normalized enough for the neural net,
        OR apply a simple identity if we aren't sure about the scaler mapping.
        """
        # For now, pass through as the environment explicitly normalizes.
        return x


    @torch.no_grad()
    def reset_states(self):
        """Reset LSTM hidden states for new episode."""
        self.hidden_state = None

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Observation tensor [batch, obs_dim] or [batch, seq_len, obs_dim]
            hidden_state: Optional LSTM hidden state tuple (h, c)

        Returns:
            actions: Action tensor [-1, 1]
            new_hidden_state: Updated LSTM hidden state
        """
        # Normalize input
        x_norm = self._normalize_obs(x)

        # Handle inference (single step) vs training (sequence)
        is_inference = x.dim() == 2

        if is_inference:
            # Add sequence dimension for LSTM
            x_step = x_norm.unsqueeze(1)
            lstm_out, self.hidden_state = self.lstm(x_step, self.hidden_state)
            lstm_out = lstm_out.squeeze(1)
            new_hidden_state = self.hidden_state
        else:
            # Training: process full sequence
            lstm_out, new_hidden_state = self.lstm(x_norm, hidden_state)

        # Process through layers
        h = self.layernorm(lstm_out)
        h = self.activation(self.dense(h))

        # Output with tanh activation for [-1, 1] range
        actions = torch.tanh(self.output_layer(h))

        return actions, new_hidden_state


class CriticNetwork(nn.Module):
    """
    LSTM-based Critic network for Q-value estimation.

    Input: 6D observation + 3D action concatenated
    Output: Q-value scalar
    """

    def __init__(
        self,
        scaler_params: Dict,
        obs_dim: int = 7,
        action_dim: int = 4,
        hidden_size: int = 128,
    ):
        """
        Initialize the Critic network.

        Args:
            scaler_params: Dictionary with normalization parameters
            obs_dim: Observation dimension (6)
            action_dim: Action dimension (3)
            hidden_size: LSTM hidden size
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        # Register normalization buffers
        self._register_scaler_buffers(scaler_params)

        # LSTM processes observation + action
        self.lstm = nn.LSTM(
            input_size=obs_dim + action_dim, hidden_size=hidden_size, batch_first=True
        )

        self.layernorm = nn.LayerNorm(hidden_size)

        # Q-value head
        self.q_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Hidden state
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _register_scaler_buffers(self, scaler_params: Dict):
        """Register scaler parameters as buffers."""
        scale = torch.tensor(scaler_params.get("scale", [1.0] * 5), dtype=torch.float32)
        min_ = torch.tensor(scaler_params.get("min", [0.0] * 5), dtype=torch.float32)
        self.register_buffer("_scale", scale)
        self.register_buffer("_min", min_)

    def _normalize_obs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize observation.
        
        The environment already handles normalization (velocity/target in [0,1], 
        actions in [-1,1], error in [-1,1]).
        
        However, the original architecture allowed for an additional scaler layer based on 
        simulation input statistics. 
        
        In the ONNX rebuild v2, we preserve the input 'x' structure (7D) and return it.
        We skip valid usage of the legacy scaler buffers for now to prevent dimension mismatch,
        since the environment is already normalized.
        """
        return x

    @torch.no_grad()
    def reset_states(self):
        """Reset LSTM hidden states."""
        self.hidden_state = None

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            obs: Observation tensor [batch, obs_dim] or [batch, seq_len, obs_dim]
            action: Action tensor [batch, action_dim] or [batch, seq_len, action_dim]
            hidden_state: Optional LSTM hidden state

        Returns:
            q_value: Q-value estimate
            new_hidden_state: Updated LSTM hidden state
        """
        # Normalize observation
        obs_norm = self._normalize_obs(obs)

        # Concatenate observation and action
        x = torch.cat([obs_norm, action], dim=-1)

        # Handle inference vs training
        is_inference = x.dim() == 2

        if is_inference:
            x_step = x.unsqueeze(1)
            lstm_out, self.hidden_state = self.lstm(x_step, self.hidden_state)
            lstm_out = lstm_out.squeeze(1)
            new_hidden_state = self.hidden_state
        else:
            lstm_out, new_hidden_state = self.lstm(x, hidden_state)

        # Q-value estimation
        h = self.layernorm(lstm_out)
        q_value = self.q_network(h)

        return q_value, new_hidden_state


def copy_target(target: nn.Module, source: nn.Module):
    """Hard copy parameters from source to target network."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Soft update target network: θ_target = τ*θ_source + (1-τ)*θ_target"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )
