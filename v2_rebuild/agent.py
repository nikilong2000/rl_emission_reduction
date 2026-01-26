"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) agent with recurrent networks.

This module implements the TD3 algorithm with LSTM-based Actor and Critic networks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
from networks import Actor, RecurrentCritic, copy_network_parameters, soft_update_network


class TD3Agent:
    """
    TD3 Agent with recurrent networks.
    
    Key features:
    - Twin critics to reduce overestimation
    - Delayed policy updates
    - Target policy smoothing
    - LSTM-based networks for sequential data
    """
    
    def __init__(
        self,
        obs_dim: int = 6,
        action_dim: int = 3,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scaler_params: Optional[dict] = None
    ):
        """
        Initialize TD3 Agent.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden dimension for LSTM
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critics
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            policy_delay: Delay between policy updates
            target_noise: Noise added to target policy
            target_noise_clip: Clip range for target noise
            exploration_noise: Noise added during exploration
            device: Device to run on (cpu/cuda)
            scaler_params: Optional scaler parameters for normalization
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.exploration_noise = exploration_noise
        self.device = torch.device(device)
        
        # Create networks
        self.actor = Actor(obs_dim, action_dim, hidden_dim, scaler_params).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, hidden_dim, scaler_params).to(self.device)
        copy_network_parameters(self.actor_target, self.actor)
        
        # Twin critics
        self.critic1 = RecurrentCritic(obs_dim, action_dim, hidden_dim, scaler_params).to(self.device)
        self.critic2 = RecurrentCritic(obs_dim, action_dim, hidden_dim, scaler_params).to(self.device)
        
        self.critic1_target = RecurrentCritic(obs_dim, action_dim, hidden_dim, scaler_params).to(self.device)
        self.critic2_target = RecurrentCritic(obs_dim, action_dim, hidden_dim, scaler_params).to(self.device)
        
        copy_network_parameters(self.critic1_target, self.critic1)
        copy_network_parameters(self.critic2_target, self.critic2)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Training step counter
        self.total_it = 0
    
    def select_action(
        self,
        obs: np.ndarray,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            obs: Observation (obs_dim,)
            add_noise: Whether to add exploration noise
            
        Returns:
            Action (action_dim,) in range [-1, 1]
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, obs_dim)
            action, _ = self.actor(obs_tensor)
            action = action.cpu().numpy()[0]
            
            if add_noise:
                noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
                action = np.clip(action + noise, -1.0, 1.0)
            
            return action
    
    def reset_hidden_states(self):
        """Reset hidden states for all networks."""
        self.actor.reset_hidden_state()
        self.actor_target.reset_hidden_state()
        self.critic1.reset_hidden_state()
        self.critic2.reset_hidden_state()
        self.critic1_target.reset_hidden_state()
        self.critic2_target.reset_hidden_state()
    
    def train_step(
        self,
        obs_batch: np.ndarray,
        action_batch: np.ndarray,
        reward_batch: np.ndarray,
        next_obs_batch: np.ndarray,
        terminated_batch: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform one training step.
        
        Args:
            obs_batch: Observations (batch_size, seq_len, obs_dim)
            action_batch: Actions (batch_size, seq_len, action_dim)
            reward_batch: Rewards (batch_size, seq_len)
            next_obs_batch: Next observations (batch_size, seq_len, obs_dim)
            terminated_batch: Terminated flags (batch_size, seq_len)
            
        Returns:
            Tuple of (critic_loss, actor_loss)
            actor_loss is None if policy not updated this step
        """
        self.total_it += 1
        
        # Convert to tensors
        obs = torch.FloatTensor(obs_batch).to(self.device)
        action = torch.FloatTensor(action_batch).to(self.device)
        reward = torch.FloatTensor(reward_batch).to(self.device)
        next_obs = torch.FloatTensor(next_obs_batch).to(self.device)
        terminated = torch.FloatTensor(terminated_batch).to(self.device)
        
        # ===== Update Critics =====
        with torch.no_grad():
            # Compute target actions with noise
            next_action, _ = self.actor_target(next_obs)
            
            # Add target policy smoothing noise
            noise = torch.randn_like(next_action) * self.target_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            
            # Compute target Q-values (minimum of two critics)
            target_q1, _ = self.critic1_target(next_obs, next_action)
            target_q2, _ = self.critic2_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # Compute TD target
            # target_q and reward should have compatible shapes
            target_q = target_q.squeeze(-1)  # (batch_size, seq_len)
            target_value = reward + self.gamma * (1 - terminated) * target_q
        
        # Compute current Q-values
        current_q1, _ = self.critic1(obs, action)
        current_q2, _ = self.critic2(obs, action)
        current_q1 = current_q1.squeeze(-1)  # (batch_size, seq_len)
        current_q2 = current_q2.squeeze(-1)  # (batch_size, seq_len)
        
        # Compute critic losses
        critic1_loss = nn.MSELoss()(current_q1, target_value)
        critic2_loss = nn.MSELoss()(current_q2, target_value)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        actor_loss = None
        
        # ===== Delayed Policy Update =====
        if self.total_it % self.policy_delay == 0:
            # Compute actor loss
            predicted_action, _ = self.actor(obs)
            actor_q, _ = self.critic1(obs, predicted_action)
            actor_loss = -actor_q.mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            soft_update_network(self.actor_target, self.actor, self.tau)
            soft_update_network(self.critic1_target, self.critic1, self.tau)
            soft_update_network(self.critic2_target, self.critic2, self.tau)
            
            actor_loss = actor_loss.item()
        
        return critic_loss.item(), actor_loss
    
    def save(self, filepath: str):
        """
        Save agent state.
        
        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'total_it': self.total_it
        }, filepath)
    
    def load(self, filepath: str):
        """
        Load agent state.
        
        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        self.total_it = checkpoint['total_it']
