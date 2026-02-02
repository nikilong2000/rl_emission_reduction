"""
Single-threaded training loop for TD3 agent.
For debugging and validation before moving to distributed training.
"""

import numpy as np
import torch
import os
import time
from typing import Dict, Optional
import matplotlib.pyplot as plt

from simulation import Simulation
from environment import VehicleEnvironment
from agent import TD3Agent
from buffer import SequenceReplayBuffer


def get_scaler_params_from_simulation(sim: Simulation) -> Dict:
    """Extract scaler parameters from simulation for network initialization."""
    # Use PG input scaler as reference for observation normalization
    scaler = sim.pg_in_scaler
    return {
        "scale": scaler.scale_.tolist()[:5] if hasattr(scaler, "scale_") else [1.0] * 5,
        "min": scaler.min_.tolist()[:5] if hasattr(scaler, "min_") else [0.0] * 5,
        "data_min": (
            scaler.data_min_.tolist()[:5] if hasattr(scaler, "data_min_") else [0.0] * 5
        ),
        "data_max": (
            scaler.data_max_.tolist()[:5] if hasattr(scaler, "data_max_") else [1.0] * 5
        ),
    }


def train_serial(
    ice_model_path: str,
    pg_model_path: str,
    num_episodes: int = 500,
    max_steps_per_episode: int = 1200,
    batch_size: int = 32,
    buffer_capacity: int = 100000,
    burn_in_length: int = 20,
    unroll_length: int = 40,
    updates_per_step: int = 1,
    warmup_episodes: int = 5,
    eval_interval: int = 20,
    save_interval: int = 50,
    log_interval: int = 10,
    output_dir: str = "results",
    device: str = "cpu",
):
    """
    Single-threaded training loop.

    Args:
        ice_model_path: Path to ICE model directory
        pg_model_path: Path to PG model directory
        num_episodes: Total number of episodes
        max_steps_per_episode: Maximum steps per episode
        batch_size: Training batch size
        buffer_capacity: Replay buffer capacity
        burn_in_length: LSTM burn-in steps
        unroll_length: LSTM unroll steps for learning
        updates_per_step: Gradient updates per environment step
        warmup_episodes: Episodes for initial exploration (no learning)
        eval_interval: Episodes between evaluations
        save_interval: Episodes between checkpoints
        log_interval: Episodes between logging
        output_dir: Directory for saving results
        device: Computation device
    """
    print("=" * 70)
    print("SERIAL TD3 TRAINING")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize simulation
    print("\n1. Initializing simulation...")
    sim = Simulation(
        ice_model_path=ice_model_path, pg_model_path=pg_model_path, soc_initial=0.7
    )

    # Initialize environment
    print("\n2. Initializing environment...")
    env = VehicleEnvironment(
        simulation=sim, max_steps=max_steps_per_episode, vel_target=70.0
    )

    # Get scaler parameters for networks
    scaler_params = get_scaler_params_from_simulation(sim)

    # Initialize agent
    print("\n3. Initializing TD3 agent...")
    agent = TD3Agent(
        scaler_params=scaler_params,
        obs_dim=7,
        action_dim=4,
        hidden_size=128,
        device=device,
    )

    # Initialize replay buffer
    print("\n4. Initializing replay buffer...")
    buffer = SequenceReplayBuffer(
        capacity=buffer_capacity,
        burn_in_length=burn_in_length,
        unroll_length=unroll_length,
        obs_dim=7,
        action_dim=4,
    )

    # Training metrics
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "episode_velocities": [],
        "critic_losses": [],
        "actor_losses": [],
    }

    print("\n5. Starting training...")
    print(f"   Warmup episodes: {warmup_episodes}")
    print(f"   Total episodes: {num_episodes}")
    print(f"   Sequence length: {burn_in_length} (burn-in) + {unroll_length} (unroll)")
    print("-" * 70)

    start_time = time.time()

    for episode in range(num_episodes):
        # Reset environment and agent
        obs, info = env.reset()
        agent.reset_episode()

        episode_reward = 0.0
        episode_velocities = []
        step = 0
        done = False

        while not done:
            # Select action (add noise during exploration)
            add_noise = episode < warmup_episodes or np.random.random() < 0.9
            action = agent.select_action(obs, add_noise=add_noise)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            buffer.add_transition(obs, action, reward, next_obs, done)

            # Track metrics
            episode_reward += reward
            episode_velocities.append(info["velocity"])

            # Update agent (after warmup)
            if episode >= warmup_episodes and buffer.can_sample(batch_size):
                for _ in range(updates_per_step):
                    batch = buffer.sample(batch_size)
                    update_info = agent.update(batch, burn_in_length)
                    metrics["critic_losses"].append(update_info["critic_loss"])
                    metrics["actor_losses"].append(update_info["actor_loss"])

            obs = next_obs
            step += 1

        # Store episode metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(step)
        metrics["episode_velocities"].append(np.mean(episode_velocities))

        # Logging
        if (episode + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(metrics["episode_rewards"][-log_interval:])
            avg_velocity = np.mean(metrics["episode_velocities"][-log_interval:])
            print(
                f"Episode {episode+1:4d} | "
                f"Reward: {avg_reward:7.2f} | "
                f"Vel: {avg_velocity:6.2f} km/h | "
                f"Steps: {step:4d} | "
                f"Buffer: {len(buffer):6d} | "
                f"Time: {elapsed:.1f}s"
            )

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{episode+1}.pt")
            agent.save(checkpoint_path)
            print(f"   → Saved checkpoint: {checkpoint_path}")

        # Evaluation
        if (episode + 1) % eval_interval == 0:
            eval_reward, eval_velocity = evaluate_agent(env, agent)
            print(
                f"   → Eval: Reward={eval_reward:.2f}, Velocity={eval_velocity:.2f} km/h"
            )

    # Final save
    final_path = os.path.join(output_dir, "final_agent.pt")
    agent.save(final_path)
    print(f"\n✓ Training complete. Final model saved to: {final_path}")

    # Plot training curves
    plot_training_curves(metrics, output_dir)

    return agent, metrics


def evaluate_agent(
    env: VehicleEnvironment, agent: TD3Agent, num_episodes: int = 3
) -> tuple:
    """Evaluate agent without exploration noise."""
    total_reward = 0.0
    total_velocity = 0.0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        agent.reset_episode()
        done = False
        episode_reward = 0.0
        velocities = []

        while not done:
            action = agent.select_action(obs, add_noise=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            velocities.append(info["velocity"])

        total_reward += episode_reward
        total_velocity += np.mean(velocities)

    return total_reward / num_episodes, total_velocity / num_episodes


def plot_training_curves(metrics: Dict, output_dir: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Episode rewards
    axes[0, 0].plot(metrics["episode_rewards"], alpha=0.6)
    if len(metrics["episode_rewards"]) > 10:
        window = min(50, len(metrics["episode_rewards"]) // 4)
        smoothed = np.convolve(
            metrics["episode_rewards"], np.ones(window) / window, mode="valid"
        )
        axes[0, 0].plot(
            range(window - 1, len(metrics["episode_rewards"])),
            smoothed,
            "r-",
            linewidth=2,
        )
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Episode Reward")
    axes[0, 0].set_title("Training Rewards")
    axes[0, 0].grid(True, alpha=0.3)

    # Episode velocities
    axes[0, 1].plot(metrics["episode_velocities"], alpha=0.6)
    axes[0, 1].axhline(y=70, color="r", linestyle="--", label="Target")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Average Velocity [km/h]")
    axes[0, 1].set_title("Average Velocity per Episode")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Critic loss
    if metrics["critic_losses"]:
        axes[1, 0].plot(metrics["critic_losses"], alpha=0.3)
        axes[1, 0].set_xlabel("Update Step")
        axes[1, 0].set_ylabel("Critic Loss")
        axes[1, 0].set_title("Critic Loss")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)

    # Episode lengths
    axes[1, 1].plot(metrics["episode_lengths"], alpha=0.6)
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Steps")
    axes[1, 1].set_title("Episode Length")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   → Saved training curves: {output_path}")
    plt.close()


if __name__ == "__main__":
    # Default paths
    ICE_PATH = "../controller_for_ICE_PG/SHARE/CTTC_models/ONNX/ICE"
    PG_PATH = "../controller_for_ICE_PG/SHARE/CTTC_models/ONNX/PG"

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Run training
    agent, metrics = train_serial(
        ice_model_path=ICE_PATH,
        pg_model_path=PG_PATH,
        num_episodes=100,  # Start small for testing
        max_steps_per_episode=500,
        batch_size=32,
        warmup_episodes=3,
        eval_interval=10,
        save_interval=25,
        log_interval=5,
        output_dir="results",
        device=device,
    )
