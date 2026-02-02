"""
Replay Buffer for recurrent TD3 with burn-in support.
Stores complete episodes and samples sequences for training.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class SequenceReplayBuffer:
    """
    Replay buffer designed for recurrent networks.

    Stores complete episodes and samples sequences with:
    - Burn-in period: Initial steps to warm up LSTM hidden states
    - Unroll period: Steps used for actual learning

    This approach prevents bootstrap from stale hidden states.
    """

    def __init__(
        self,
        capacity: int = 100000,
        burn_in_length: int = 20,
        unroll_length: int = 40,
        obs_dim: int = 6,
        action_dim: int = 3,
    ):
        """
        Initialize the buffer.

        Args:
            capacity: Maximum number of transitions to store
            burn_in_length: Number of steps for LSTM warm-up (not used for loss)
            unroll_length: Number of steps for actual learning
            obs_dim: Observation dimension
            action_dim: Action dimension
        """
        self.capacity = capacity
        self.burn_in_length = burn_in_length
        self.unroll_length = unroll_length
        self.sequence_length = burn_in_length + unroll_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Storage for complete episodes
        self.episodes: deque = deque()
        self.total_transitions = 0

        # Current episode being collected
        self.current_episode: List[Dict] = []

    def add_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """
        Add a single transition to the current episode.

        Args:
            obs: Observation (6D)
            action: Action (3D)
            reward: Reward value
            next_obs: Next observation (6D)
            done: Whether episode ended
        """
        transition = {
            "obs": np.array(obs, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "reward": float(reward),
            "next_obs": np.array(next_obs, dtype=np.float32),
            "done": bool(done),
        }

        self.current_episode.append(transition)

        if done:
            self._store_episode()

    def _store_episode(self):
        """Store the current episode and manage capacity."""
        if len(self.current_episode) >= self.sequence_length:
            episode_length = len(self.current_episode)
            self.episodes.append(list(self.current_episode))
            self.total_transitions += episode_length

            # Remove old episodes if over capacity
            while self.total_transitions > self.capacity and len(self.episodes) > 1:
                removed_episode = self.episodes.popleft()
                self.total_transitions -= len(removed_episode)

        self.current_episode = []

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of sequences.

        Returns sequences of length (burn_in + unroll) from random episodes.

        Args:
            batch_size: Number of sequences to sample

        Returns:
            Dictionary containing:
            - obs: [batch, seq_len, obs_dim]
            - action: [batch, seq_len, action_dim]
            - reward: [batch, seq_len, 1]
            - next_obs: [batch, seq_len, obs_dim]
            - done: [batch, seq_len, 1]
            - mask: [batch, seq_len, 1] - mask for burn-in period
        """
        # Filter episodes that are long enough
        valid_episodes = [ep for ep in self.episodes if len(ep) >= self.sequence_length]

        if len(valid_episodes) < batch_size:
            # Sample with replacement if not enough episodes
            sampled_episodes = random.choices(valid_episodes, k=batch_size)
        else:
            sampled_episodes = random.sample(valid_episodes, batch_size)

        # Initialize batch arrays
        batch_obs = np.zeros(
            (batch_size, self.sequence_length, self.obs_dim), dtype=np.float32
        )
        batch_action = np.zeros(
            (batch_size, self.sequence_length, self.action_dim), dtype=np.float32
        )
        batch_reward = np.zeros((batch_size, self.sequence_length, 1), dtype=np.float32)
        batch_next_obs = np.zeros(
            (batch_size, self.sequence_length, self.obs_dim), dtype=np.float32
        )
        batch_done = np.zeros((batch_size, self.sequence_length, 1), dtype=np.float32)

        # Mask: 0 for burn-in, 1 for unroll period
        batch_mask = np.zeros((batch_size, self.sequence_length, 1), dtype=np.float32)
        batch_mask[:, self.burn_in_length :, :] = 1.0

        for i, episode in enumerate(sampled_episodes):
            # Random starting point within episode
            max_start = len(episode) - self.sequence_length
            start_idx = random.randint(0, max_start)

            # Extract sequence
            for j in range(self.sequence_length):
                t = episode[start_idx + j]
                batch_obs[i, j] = t["obs"]
                batch_action[i, j] = t["action"]
                batch_reward[i, j, 0] = t["reward"]
                batch_next_obs[i, j] = t["next_obs"]
                batch_done[i, j, 0] = float(t["done"])

        return {
            "obs": batch_obs,
            "action": batch_action,
            "reward": batch_reward,
            "next_obs": batch_next_obs,
            "done": batch_done,
            "mask": batch_mask,
        }

    def __len__(self) -> int:
        """Return total number of transitions stored."""
        return self.total_transitions

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough data to sample."""
        valid_episodes = [ep for ep in self.episodes if len(ep) >= self.sequence_length]
        return len(valid_episodes) >= 1 and self.total_transitions >= batch_size

    @property
    def num_episodes(self) -> int:
        """Return number of complete episodes stored."""
        return len(self.episodes)


class SimpleReplayBuffer:
    """
    Simple flat replay buffer for non-recurrent training comparison.
    Stores individual transitions without episode structure.
    """

    def __init__(self, capacity: int = 100000, obs_dim: int = 6, action_dim: int = 3):
        """
        Initialize the buffer.

        Args:
            capacity: Maximum number of transitions
            obs_dim: Observation dimension
            action_dim: Action dimension
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Pre-allocate storage
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Add a transition to the buffer."""
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr, 0] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr, 0] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs": self.obs[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "next_obs": self.next_obs[indices],
            "done": self.done[indices],
        }

    def __len__(self) -> int:
        return self.size

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size
