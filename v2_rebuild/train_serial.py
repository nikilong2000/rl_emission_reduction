"""
Single-threaded training loop for TD3 agent.

This script implements a simple, single-process training loop to validate
the algorithm before adding Ray complexity.
"""
import numpy as np
import torch
import os
import time
from typing import Optional

# Note: Requires Keras compatibility fix for environment/simulation to work
# This is a template showing the correct structure

def train_serial(
    ice_model_dir: str,
    pg_model_dir: str,
    data_dir: str,
    save_dir: str = "./checkpoints",
    num_episodes: int = 1000,
    max_steps_per_episode: int = 1200,
    seq_len: int = 96,
    buffer_capacity: int = 10000,
    batch_size: int = 32,
    min_buffer_size: int = 1000,
    learning_starts: int = 1000,
    eval_frequency: int = 10,
    save_frequency: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train TD3 agent in a single-threaded loop.
    
    Args:
        ice_model_dir: Path to ICE model directory
        pg_model_dir: Path to PG model directory
        data_dir: Path to data directory for initial states
        save_dir: Directory to save checkpoints
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        seq_len: Length of sequence windows for replay buffer
        buffer_capacity: Capacity of replay buffer
        batch_size: Batch size for training
        min_buffer_size: Minimum buffer size before training starts
        learning_starts: Number of steps before learning starts
        eval_frequency: Evaluate every N episodes
        save_frequency: Save checkpoint every N episodes
        device: Device to use (cpu/cuda)
    """
    from environment import VehicleControlEnvironment
    from agent import TD3Agent
    from buffer import SequenceReplayBuffer, EpisodeBuffer
    
    print("="*70)
    print("Single-Threaded TD3 Training")
    print("="*70)
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print(f"Sequence length: {seq_len}")
    print(f"Buffer capacity: {buffer_capacity}")
    print(f"Batch size: {batch_size}")
    print("="*70)
    print()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = VehicleControlEnvironment(
        ice_model_dir=ice_model_dir,
        pg_model_dir=pg_model_dir,
        data_dir=data_dir,
        max_steps=max_steps_per_episode
    )
    print(f"✓ Environment created (obs_dim={env.observation_space_dim}, action_dim={env.action_space_dim})")
    print()
    
    # Create agent
    print("Creating TD3 agent...")
    agent = TD3Agent(
        obs_dim=env.observation_space_dim,
        action_dim=env.action_space_dim,
        device=device
    )
    print(f"✓ Agent created")
    print()
    
    # Create replay buffer
    print("Creating replay buffer...")
    replay_buffer = SequenceReplayBuffer(
        capacity=buffer_capacity,
        seq_len=seq_len,
        obs_dim=env.observation_space_dim,
        action_dim=env.action_space_dim
    )
    print(f"✓ Replay buffer created")
    print()
    
    # Training loop
    print("Starting training...")
    print()
    
    total_steps = 0
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Reset environment
        obs = env.reset(vel_target=70.0)
        agent.reset_hidden_states()
        
        # Episode buffer
        episode_buffer = EpisodeBuffer(
            obs_dim=env.observation_space_dim,
            action_dim=env.action_space_dim
        )
        
        episode_reward = 0
        episode_steps = 0
        start_time = time.time()
        
        # Episode loop
        for step in range(max_steps_per_episode):
            # Select action
            if total_steps < learning_starts:
                # Random exploration
                action = np.random.uniform(-1.0, 1.0, size=env.action_space_dim)
            else:
                # Policy with exploration noise
                action = agent.select_action(obs, add_noise=True)
            
            # Scale action to environment ranges (simple linear scaling for now)
            # This is a placeholder - adjust based on actual action ranges
            scaled_action = np.array([
                (action[0] + 1) * 10 + 3,      # mf: [3, 23]
                (action[1] + 1) * 50,           # brk: [0, 100]
                (action[2] + 1) * 1100 + 800    # ice_sp: [800, 3000]
            ])
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(scaled_action)
            
            # Store transition
            episode_buffer.add(obs, action, reward, next_obs, terminated, truncated)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Extract windows from episode
        windows = episode_buffer.extract_windows(seq_len, stride=seq_len // 2)
        for window in windows:
            replay_buffer.add(window)
        
        episode_time = time.time() - start_time
        episode_rewards.append(episode_reward)
        
        # Training
        critic_losses = []
        actor_losses = []
        
        if total_steps >= learning_starts and replay_buffer.is_ready(min_buffer_size):
            # Perform multiple training steps
            num_train_steps = max(1, episode_steps // 4)
            
            for _ in range(num_train_steps):
                # Sample batch
                batch = replay_buffer.sample(batch_size)
                obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch, truncated_batch = batch
                
                # Train agent
                critic_loss, actor_loss = agent.train_step(
                    obs_batch, action_batch, reward_batch,
                    next_obs_batch, terminated_batch
                )
                
                critic_losses.append(critic_loss)
                if actor_loss is not None:
                    actor_losses.append(actor_loss)
        
        # Logging
        if episode % eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-eval_frequency:])
            avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
            avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
            
            print(f"Episode {episode:4d} | "
                  f"Steps: {total_steps:6d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Buffer: {len(replay_buffer):5d} | "
                  f"C-Loss: {avg_critic_loss:7.4f} | "
                  f"A-Loss: {avg_actor_loss:7.4f} | "
                  f"Time: {episode_time:5.2f}s")
        
        # Save checkpoint
        if episode > 0 and episode % save_frequency == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_ep{episode}.pt")
            agent.save(checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
    
    # Final save
    final_path = os.path.join(save_dir, "final_model.pt")
    agent.save(final_path)
    print()
    print(f"✓ Training complete! Final model saved: {final_path}")
    print()
    
    return agent, episode_rewards


if __name__ == "__main__":
    # Example usage
    base_path = "/home/runner/work/rl_emission_reduction/rl_emission_reduction/controller_for_ICE_PG"
    
    ice_dir = os.path.join(base_path, "SHARE/CTTC_models/ONNX/ICE")
    pg_dir = os.path.join(base_path, "SHARE/CTTC_models/ONNX/PG")
    data_dir = os.path.join(base_path, "src/data")
    
    # Check for alternative paths
    if not os.path.exists(ice_dir):
        ice_dir = os.path.join(base_path, "src/models_markus/ICE_Model_Update_01")
    if not os.path.exists(pg_dir):
        pg_dir = os.path.join(base_path, "src/models_markus/PG_v3")
    
    print("\nNote: This script requires Keras compatibility fix to run.")
    print("The simulation/environment will fail to load models with current Keras version.")
    print("However, the training loop structure is correct and ready to use.\n")
    
    # Uncomment to run (after Keras compatibility is fixed):
    # agent, rewards = train_serial(
    #     ice_model_dir=ice_dir,
    #     pg_model_dir=pg_dir,
    #     data_dir=data_dir,
    #     num_episodes=100,  # Reduced for testing
    #     eval_frequency=5,
    #     save_frequency=20
    # )
