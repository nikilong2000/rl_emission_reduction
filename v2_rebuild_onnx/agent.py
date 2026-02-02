"""
TD3 Agent implementation with recurrent networks.
Includes noise exploration, target networks, and delayed policy updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from networks import ActorNetwork, CriticNetwork, copy_target, soft_update


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
    ):
        """
        Initialize OU noise.

        Args:
            action_dim: Dimension of action space
            mu: Mean of noise
            theta: Rate of mean reversion
            sigma: Volatility of noise
            dt: Time step
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        """Reset noise state."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        """Sample noise value."""
        dx = self.theta * (self.mu - self.state) * self.dt
        dx += self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state += dx
        return self.state.astype(np.float32)


class GaussianNoise:
    """Simple Gaussian noise for exploration."""

    def __init__(self, action_dim: int, sigma: float = 0.1):
        self.action_dim = action_dim
        self.sigma = sigma

    def reset(self):
        pass

    def sample(self) -> np.ndarray:
        return np.random.randn(self.action_dim).astype(np.float32) * self.sigma


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent.

    Features:
    - Twin critics to reduce overestimation
    - Delayed policy updates
    - Target policy smoothing
    - Recurrent networks (LSTM)
    """

    def __init__(
        self,
        scaler_params: Dict,
        obs_dim: int = 6,
        action_dim: int = 3,
        hidden_size: int = 128,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize TD3 agent.

        Args:
            scaler_params: Normalization parameters for networks
            obs_dim: Observation dimension (6)
            action_dim: Action dimension (3)
            hidden_size: LSTM hidden size
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            policy_delay: Steps between policy updates
            policy_noise: Noise added to target actions
            noise_clip: Clipping range for target noise
            device: Computation device ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        # Actor networks
        self.actor = ActorNetwork(scaler_params, obs_dim, action_dim, hidden_size).to(
            self.device
        )
        self.actor_target = ActorNetwork(
            scaler_params, obs_dim, action_dim, hidden_size
        ).to(self.device)
        copy_target(self.actor_target, self.actor)

        # Twin Critics
        self.critic_1 = CriticNetwork(
            scaler_params, obs_dim, action_dim, hidden_size
        ).to(self.device)
        self.critic_2 = CriticNetwork(
            scaler_params, obs_dim, action_dim, hidden_size
        ).to(self.device)
        self.critic_1_target = CriticNetwork(
            scaler_params, obs_dim, action_dim, hidden_size
        ).to(self.device)
        self.critic_2_target = CriticNetwork(
            scaler_params, obs_dim, action_dim, hidden_size
        ).to(self.device)
        copy_target(self.critic_1_target, self.critic_1)
        copy_target(self.critic_2_target, self.critic_2)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=critic_lr,
        )

        # Exploration noise
        self.exploration_noise = OrnsteinUhlenbeckNoise(action_dim)

        # Training step counter
        self.total_steps = 0

    def reset_episode(self):
        """Reset agent for new episode."""
        self.actor.reset_states()
        self.exploration_noise.reset()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action for given observation.

        Args:
            obs: Observation array (6D)
            add_noise: Whether to add exploration noise

        Returns:
            action: Action array (3D) in [-1, 1]
        """
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        # Get action from actor
        action, _ = self.actor(obs_tensor)
        action = action.squeeze(0).cpu().numpy()

        # Add exploration noise
        if add_noise:
            noise = self.exploration_noise.sample()
            action = action + noise

        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)

        return action.astype(np.float32)

    def update(
        self, batch: Dict[str, np.ndarray], burn_in_length: int = 20
    ) -> Dict[str, float]:
        """
        Update agent with a batch of sequences.

        Args:
            batch: Dictionary with obs, action, reward, next_obs, done, mask
            burn_in_length: Number of steps for LSTM warm-up

        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        action = torch.FloatTensor(batch["action"]).to(self.device)
        reward = torch.FloatTensor(batch["reward"]).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        done = torch.FloatTensor(batch["done"]).to(self.device)
        mask = torch.FloatTensor(batch["mask"]).to(self.device)

        batch_size, seq_len = obs.shape[:2]

        # ========== Critic Update ==========
        with torch.no_grad():
            # Get target actions with noise
            target_actions, _ = self.actor_target(next_obs)
            noise = torch.randn_like(target_actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            target_actions = (target_actions + noise).clamp(-1.0, 1.0)

            # Compute target Q-values (minimum of twin critics)
            target_q1, _ = self.critic_1_target(next_obs, target_actions)
            target_q2, _ = self.critic_2_target(next_obs, target_actions)
            target_q = torch.min(target_q1, target_q2)

            # TD target: r + γ * (1 - done) * Q_target
            target = reward + self.gamma * (1 - done) * target_q

        # Current Q-values
        current_q1, _ = self.critic_1(obs, action)
        current_q2, _ = self.critic_2(obs, action)

        # Critic loss (masked to ignore burn-in period)
        critic_loss_1 = (mask * (current_q1 - target) ** 2).mean()
        critic_loss_2 = (mask * (current_q2 - target) ** 2).mean()
        critic_loss = critic_loss_1 + critic_loss_2

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            max_norm=1.0,
        )
        self.critic_optimizer.step()

        # ========== Actor Update (Delayed) ==========
        actor_loss = torch.tensor(0.0)
        self.total_steps += 1

        if self.total_steps % self.policy_delay == 0:
            # Compute actor loss
            new_actions, _ = self.actor(obs)
            actor_q1, _ = self.critic_1(obs, new_actions)

            # Actor loss: maximize Q-value (minimize negative Q)
            # Apply mask to ignore burn-in
            actor_loss = -(mask * actor_q1).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Soft update target networks
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": current_q1.mean().item(),
            "q2_mean": current_q2.mean().item(),
        }

    def save(self, path: str):
        """Save agent state."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "critic_1_target": self.critic_1_target.state_dict(),
                "critic_2_target": self.critic_2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.critic_1_target.load_state_dict(checkpoint["critic_1_target"])
        self.critic_2_target.load_state_dict(checkpoint["critic_2_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_steps = checkpoint["total_steps"]
