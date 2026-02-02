"""
Distributed training with Ray.
Separates data collection (RolloutWorker) from learning (Learner).
"""

import ray
import numpy as np
import torch
import os
import time
from typing import Dict, List, Tuple, Optional
import copy

from simulation import Simulation
from environment import VehicleEnvironment
from agent import TD3Agent
from buffer import SequenceReplayBuffer
from networks import ActorNetwork


@ray.remote
class RolloutWorker:
    """
    Worker that collects rollouts from the environment.
    Runs on CPU, collects experience and sends to learner.
    """

    def __init__(
        self,
        worker_id: int,
        ice_model_path: str,
        pg_model_path: str,
        scaler_params: Dict,
        max_steps: int = 1200,
        vel_target: float = 70.0,
    ):
        """
        Initialize rollout worker.

        Args:
            worker_id: Unique worker identifier
            ice_model_path: Path to ICE model
            pg_model_path: Path to PG model
            scaler_params: Normalization parameters
            max_steps: Maximum steps per episode
            vel_target: Target velocity
        """
        self.worker_id = worker_id

        # Initialize local simulation and environment
        self.sim = Simulation(
            ice_model_path=ice_model_path, pg_model_path=pg_model_path, soc_initial=0.7
        )

        self.env = VehicleEnvironment(
            simulation=self.sim, max_steps=max_steps, vel_target=vel_target
        )

        # Local actor for action selection (no gradients needed)
        self.actor = ActorNetwork(
            scaler_params=scaler_params, obs_dim=7, action_dim=4, hidden_size=128
        )
        self.actor.eval()

        # Exploration noise parameters
        self.noise_scale = 0.2

    def set_weights(self, weights: Dict):
        """Update local actor weights from learner."""
        self.actor.load_state_dict(weights)

    def collect_episode(self, add_noise: bool = True) -> Dict:
        """
        Collect one episode of experience.

        Returns:
            Dictionary containing episode data and metrics
        """
        obs, info = self.env.reset()
        self.actor.reset_states()

        episode_data = {
            "obs": [],
            "action": [],
            "reward": [],
            "next_obs": [],
            "done": [],
        }

        episode_reward = 0.0
        episode_velocities = []
        done = False

        while not done:
            # Get action from local actor
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _ = self.actor(obs_tensor)
                action = action.squeeze(0).numpy()

            # Add exploration noise
            if add_noise:
                noise = np.random.randn(4).astype(np.float32) * self.noise_scale
                action = np.clip(action + noise, -1.0, 1.0)

            # Environment step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            episode_data["obs"].append(obs.copy())
            episode_data["action"].append(action.copy())
            episode_data["reward"].append(reward)
            episode_data["next_obs"].append(next_obs.copy())
            episode_data["done"].append(done)

            episode_reward += reward
            episode_velocities.append(info["velocity"])
            obs = next_obs

        # Convert to numpy arrays
        episode_data["obs"] = np.array(episode_data["obs"], dtype=np.float32)
        episode_data["action"] = np.array(episode_data["action"], dtype=np.float32)
        episode_data["reward"] = np.array(episode_data["reward"], dtype=np.float32)
        episode_data["next_obs"] = np.array(episode_data["next_obs"], dtype=np.float32)
        episode_data["done"] = np.array(episode_data["done"], dtype=bool)

        return {
            "episode_data": episode_data,
            "episode_reward": episode_reward,
            "episode_length": len(episode_data["reward"]),
            "mean_velocity": np.mean(episode_velocities),
            "worker_id": self.worker_id,
        }


@ray.remote(num_gpus=0.5 if torch.cuda.is_available() else 0)
class Learner:
    """
    Central learner that performs gradient updates.
    Runs on GPU if available.
    """

    def __init__(
        self,
        scaler_params: Dict,
        buffer_capacity: int = 100000,
        burn_in_length: int = 20,
        unroll_length: int = 40,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize learner.

        Args:
            scaler_params: Normalization parameters
            buffer_capacity: Replay buffer capacity
            burn_in_length: LSTM burn-in steps
            unroll_length: LSTM unroll steps
            batch_size: Training batch size
            device: Computation device
        """
        self.device = device
        self.batch_size = batch_size
        self.burn_in_length = burn_in_length

        # Initialize agent
        self.agent = TD3Agent(
            scaler_params=scaler_params,
            obs_dim=7,
            action_dim=4,
            hidden_size=128,
            device=device,
        )

        # Initialize buffer
        self.buffer = SequenceReplayBuffer(
            capacity=buffer_capacity,
            burn_in_length=burn_in_length,
            unroll_length=unroll_length,
            obs_dim=7,
            action_dim=4,
        )

        # Metrics
        self.total_updates = 0
        self.critic_losses = []
        self.actor_losses = []

    def get_weights(self) -> Dict:
        """Get actor weights for workers."""
        return {k: v.cpu() for k, v in self.agent.actor.state_dict().items()}

    def add_episode(self, episode_data: Dict):
        """Add episode data to replay buffer."""
        n_steps = len(episode_data["reward"])
        for i in range(n_steps):
            self.buffer.add_transition(
                obs=episode_data["obs"][i],
                action=episode_data["action"][i],
                reward=episode_data["reward"][i],
                next_obs=episode_data["next_obs"][i],
                done=episode_data["done"][i],
            )

    def update(self, num_updates: int = 1) -> Dict:
        """
        Perform gradient updates.

        Args:
            num_updates: Number of update steps

        Returns:
            Update metrics
        """
        if not self.buffer.can_sample(self.batch_size):
            return {"status": "buffer_insufficient"}

        metrics = {"critic_loss": 0.0, "actor_loss": 0.0}

        for _ in range(num_updates):
            batch = self.buffer.sample(self.batch_size)
            update_info = self.agent.update(batch, self.burn_in_length)

            metrics["critic_loss"] += update_info["critic_loss"]
            metrics["actor_loss"] += update_info["actor_loss"]

            self.critic_losses.append(update_info["critic_loss"])
            self.actor_losses.append(update_info["actor_loss"])
            self.total_updates += 1

        metrics["critic_loss"] /= num_updates
        metrics["actor_loss"] /= num_updates
        metrics["total_updates"] = self.total_updates
        metrics["buffer_size"] = len(self.buffer)

        return metrics

    def save(self, path: str):
        """Save agent checkpoint."""
        self.agent.save(path)

    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return {
            "total_updates": self.total_updates,
            "buffer_size": len(self.buffer),
            "recent_critic_loss": (
                np.mean(self.critic_losses[-100:]) if self.critic_losses else 0
            ),
            "recent_actor_loss": (
                np.mean(self.actor_losses[-100:]) if self.actor_losses else 0
            ),
        }


def train_distributed(
    ice_model_path: str,
    pg_model_path: str,
    num_workers: int = 4,
    num_episodes: int = 500,
    updates_per_episode: int = 10,
    warmup_episodes: int = 10,
    save_interval: int = 50,
    log_interval: int = 10,
    output_dir: str = "results_distributed",
):
    """
    Distributed training with Ray.

    Args:
        ice_model_path: Path to ICE model
        pg_model_path: Path to PG model
        num_workers: Number of parallel workers
        num_episodes: Total episodes to collect
        updates_per_episode: Gradient updates per episode
        warmup_episodes: Episodes before learning starts
        save_interval: Episodes between checkpoints
        log_interval: Episodes between logging
        output_dir: Output directory
    """
    print("=" * 70)
    print("DISTRIBUTED TD3 TRAINING WITH RAY")
    print("=" * 70)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    os.makedirs(output_dir, exist_ok=True)

    # Load simulation once to get scaler parameters
    print("\n1. Loading models for scaler parameters...")
    temp_sim = Simulation(ice_model_path, pg_model_path)
    scaler = temp_sim.pg_in_scaler
    
    scale_vals = scaler.scale_.tolist()[:5] if hasattr(scaler, "scale_") else [1.0] * 5
    min_vals = scaler.min_.tolist()[:5] if hasattr(scaler, "min_") else [0.0] * 5

    scaler_params = {
        "scale": scale_vals,
        "min": min_vals,
    }
    del temp_sim

    # Create learner
    print("\n2. Creating learner...")
    learner = Learner.remote(scaler_params=scaler_params)

    # Create workers
    print(f"\n3. Creating {num_workers} rollout workers...")
    workers = [
        RolloutWorker.remote(
            worker_id=i,
            ice_model_path=ice_model_path,
            pg_model_path=pg_model_path,
            scaler_params=scaler_params,
        )
        for i in range(num_workers)
    ]

    # Metrics
    episode_rewards = []
    episode_velocities = []

    print("\n4. Starting distributed training...")
    print(f"   Workers: {num_workers}")
    print(f"   Warmup: {warmup_episodes} episodes")
    print(f"   Total: {num_episodes} episodes")
    print("-" * 70)

    start_time = time.time()
    total_episodes = 0

    # Initial weight sync
    initial_weights = ray.get(learner.get_weights.remote())
    ray.get([w.set_weights.remote(initial_weights) for w in workers])

    while total_episodes < num_episodes:
        # Collect episodes from all workers
        episode_futures = [w.collect_episode.remote(add_noise=True) for w in workers]
        results = ray.get(episode_futures)

        for result in results:
            # Add to buffer
            ray.get(learner.add_episode.remote(result["episode_data"]))

            # Track metrics
            episode_rewards.append(result["episode_reward"])
            episode_velocities.append(result["mean_velocity"])
            total_episodes += 1

            # Logging
            if total_episodes % log_interval == 0:
                elapsed = time.time() - start_time
                avg_reward = np.mean(episode_rewards[-log_interval:])
                avg_velocity = np.mean(episode_velocities[-log_interval:])
                metrics = ray.get(learner.get_metrics.remote())

                print(
                    f"Episode {total_episodes:4d} | "
                    f"Reward: {avg_reward:7.2f} | "
                    f"Vel: {avg_velocity:6.2f} | "
                    f"Buffer: {metrics['buffer_size']:6d} | "
                    f"Updates: {metrics['total_updates']:6d} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Save checkpoint
            if total_episodes % save_interval == 0:
                checkpoint_path = os.path.join(
                    output_dir, f"checkpoint_{total_episodes}.pt"
                )
                ray.get(learner.save.remote(checkpoint_path))
                print(f"   → Saved: {checkpoint_path}")

            if total_episodes >= num_episodes:
                break

        # Learning step (after warmup)
        if total_episodes >= warmup_episodes:
            ray.get(learner.update.remote(num_updates=updates_per_episode))

        # Sync weights to workers
        weights = ray.get(learner.get_weights.remote())
        ray.get([w.set_weights.remote(weights) for w in workers])

    # Final save
    final_path = os.path.join(output_dir, "final_agent.pt")
    ray.get(learner.save.remote(final_path))

    print(f"\n✓ Training complete. Final model: {final_path}")
    print(f"   Total episodes: {total_episodes}")
    print(f"   Total time: {time.time() - start_time:.1f}s")

    ray.shutdown()
    return episode_rewards, episode_velocities


if __name__ == "__main__":
    ICE_PATH = "../controller_for_ICE_PG/SHARE/CTTC_models/ONNX/ICE"
    PG_PATH = "../controller_for_ICE_PG/SHARE/CTTC_models/ONNX/PG"

    train_distributed(
        ice_model_path=ICE_PATH,
        pg_model_path=PG_PATH,
        num_workers=2,  # Adjust based on available CPUs
        num_episodes=100,
        updates_per_episode=5,
        warmup_episodes=5,
        save_interval=20,
        log_interval=5,
        output_dir="results_distributed",
    )
