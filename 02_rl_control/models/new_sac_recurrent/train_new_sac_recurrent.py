import os
import csv
import json
import time
import datetime
import warnings
import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

try:
    from ...env import EmissionControlEnv
    from ...env_thermal import EmissionControlEnvThermal
    from ...plotting import plot_exploration_entropy
    from ...utils import safety_utils
    from ...utils.platform_utils import configure_environment, configure_tf_devices
    from . import config
    from .eval_new_sac_recurrent import evaluate_model
    from .recurrent_sac_core import (
        EpisodeReplayBuffer,
        RecurrentSACAgent,
        RunningMeanStd,
        compute_state_entropy,
        resolve_model_path,
        resolve_torch_device,
    )
except ImportError:
    from env import EmissionControlEnv
    from env_thermal import EmissionControlEnvThermal
    from plotting import plot_exploration_entropy
    from utils import safety_utils
    from utils.platform_utils import configure_environment, configure_tf_devices
    import config
    from eval_new_sac_recurrent import evaluate_model
    from recurrent_sac_core import (
        EpisodeReplayBuffer,
        RecurrentSACAgent,
        RunningMeanStd,
        compute_state_entropy,
        resolve_model_path,
        resolve_torch_device,
    )


class SimpleMonitor:
    def __init__(self, env, monitor_path: str):
        self.env = env
        self.monitor_path = monitor_path
        self.episode_reward = 0.0
        self.episode_length = 0
        self.t_start = time.time()

        os.makedirs(os.path.dirname(monitor_path), exist_ok=True)
        with open(self.monitor_path, "w", newline="") as f:
            f.write(f'#{{"t_start": {self.t_start}, "env_id": "EmissionControlEnv"}}\n')
            writer = csv.writer(f)
            writer.writerow(["r", "l", "t"])

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += float(reward)
        self.episode_length += 1
        done = bool(terminated or truncated)
        if done:
            elapsed = time.time() - self.t_start
            with open(self.monitor_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.episode_reward, self.episode_length, elapsed])
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()


def _update_training_progress_plot(
    log_dir: str,
    plot_timesteps: list,
    plot_rewards: list,
    current_timestep: int,
) -> None:
    monitor_path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_path):
        return

    monitor_df = pd.read_csv(monitor_path, skiprows=1)
    if monitor_df.empty:
        return

    mean_reward = float(np.mean(monitor_df["r"].values[-100:]))
    plot_timesteps.append(current_timestep)
    plot_rewards.append(mean_reward)

    plt.figure(figsize=(10, 5))
    plt.plot(plot_timesteps, plot_rewards, label="Mean Reward (Last 100 Eps)")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, "training_progress.png"))
    plt.close()


def _find_vec_normalize_path(model_path: str) -> str:
    resolved_model_path = resolve_model_path(model_path)
    model_dir = os.path.dirname(os.path.abspath(resolved_model_path))
    model_basename = os.path.splitext(os.path.basename(resolved_model_path))[0]

    candidate_paths = [
        os.path.join(model_dir, f"{model_basename}_vecnormalize.pkl"),
        os.path.join(model_dir, "vec_normalize.pkl"),
        os.path.join(os.path.dirname(model_dir), "vec_normalize.pkl"),
    ]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate
    return ""


def main(args):
    configure_environment()
    configure_tf_devices()

    base_log_dir = os.path.join(
        os.path.dirname(os.path.dirname(current_dir)), "logs", "new_sac_recurrent"
    )
    run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, run_name)
    checkpoints_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Logging to {log_dir}")

    env_cls = EmissionControlEnvThermal if args.use_thermal else EmissionControlEnv
    env = SimpleMonitor(
        env_cls(config_module=config), os.path.join(log_dir, "monitor.csv")
    )

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    train_config = {
        "algorithm": "SAC_Recurrent",
        "env": "thermal" if args.use_thermal else "base",
        "learning_rate": config.LEARNING_RATE,
        "buffer_size": config.BUFFER_SIZE,
        "batch_size": config.BATCH_SIZE,
        "tau": config.TAU,
        "gamma": config.GAMMA,
        "train_freq": config.TRAIN_FREQ,
        "gradient_steps": config.GRADIENT_STEPS,
        "learning_starts": config.LEARNING_STARTS,
        "ent_coef": config.ENT_COEF,
        "target_entropy": config.TARGET_ENTROPY,
        "hidden_size": config.HIDDEN_SIZE,
        "mlp_hidden_size": config.MLP_HIDDEN_SIZE,
        "sequence_length": config.SEQUENCE_LENGTH,
        "clip_obs": config.CLIP_OBS,
        "total_timesteps": config.TOTAL_TIMESTEPS,
        "w_speed": config.W_SPEED,
        "w_emission": config.W_EMISSION,
        "w_fuel": config.W_FUEL,
        "w_brake": config.W_BRAKE,
        "w_soc": config.W_SOC,
        "w_soc_squared": config.W_SOC_SQUARED,
        "w_flicker": config.W_FLICKER,
        "use_onnx": bool(getattr(config, "USE_ONNX", False)),
        "continued_run": args.continue_from is not None,
        "continued_from": args.continue_from,
    }

    if args.continue_from:
        safety_utils.config_check(args.continue_from, train_config)

    agent = RecurrentSACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=config.HIDDEN_SIZE,
        mlp_hidden_size=config.MLP_HIDDEN_SIZE,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        tau=config.TAU,
        target_entropy=config.TARGET_ENTROPY,
        ent_coef=config.ENT_COEF,
        device=resolve_torch_device(args.agent_device),
    )
    normalizer = RunningMeanStd(shape=(obs_dim,))

    if args.continue_from:
        resolved_continue_path = resolve_model_path(args.continue_from)
        print(
            f"Loading existing model from {resolved_continue_path} to continue training..."
        )
        agent.load_from_file(resolved_continue_path, load_optimizers=True)

        vec_norm_path = _find_vec_normalize_path(resolved_continue_path)
        if vec_norm_path:
            normalizer.load(vec_norm_path)
            print(f"Loaded VecNormalize stats from {vec_norm_path}")
        else:
            print(
                "Warning: Could not find vec_normalize.pkl. Starting fresh normalizer."
            )

    replay_buffer = EpisodeReplayBuffer(capacity_steps=config.BUFFER_SIZE)

    training_plot_timesteps = []
    training_plot_rewards = []
    entropy_history = []
    entropy_timestep_history = []
    latest_stats = None

    obs, _ = env.reset()
    normalizer.update(obs)
    normalized_obs = normalizer.normalize(obs, clip_obs=config.CLIP_OBS)
    actor_state = agent.get_initial_actor_state(batch_size=1)
    episode_observations = []

    print("Starting Recurrent SAC Training...")
    training_start_time = time.perf_counter()
    for step in range(1, config.TOTAL_TIMESTEPS + 1):
        if step < config.LEARNING_STARTS:
            _, actor_state = agent.select_action(
                normalized_obs, actor_state, deterministic=False
            )
            action = env.action_space.sample().astype(np.float32)
        else:
            action, actor_state = agent.select_action(
                normalized_obs, actor_state, deterministic=False
            )

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        normalizer.update(next_obs)
        normalized_next_obs = normalizer.normalize(next_obs, clip_obs=config.CLIP_OBS)
        episode_observations.append(normalized_next_obs.copy())

        replay_buffer.add(obs, action, float(reward), next_obs, done)
        obs = next_obs
        normalized_obs = normalized_next_obs

        if (
            step >= config.LEARNING_STARTS
            and step % config.TRAIN_FREQ == 0
            and replay_buffer.can_sample(config.BATCH_SIZE, config.SEQUENCE_LENGTH)
        ):
            for _ in range(config.GRADIENT_STEPS):
                sampled_batch = replay_buffer.sample(
                    batch_size=config.BATCH_SIZE,
                    sequence_length=config.SEQUENCE_LENGTH,
                )
                sampled_batch["obs"] = normalizer.normalize(
                    sampled_batch["obs"], clip_obs=config.CLIP_OBS
                )
                sampled_batch["next_obs"] = normalizer.normalize(
                    sampled_batch["next_obs"], clip_obs=config.CLIP_OBS
                )
                latest_stats = agent.update(sampled_batch)

        if done:
            if len(episode_observations) > 1:
                entropy_history.append(
                    compute_state_entropy(np.stack(episode_observations), bins=20)
                )
                entropy_timestep_history.append(step)
                if len(entropy_history) % config.ENTROPY_PLOT_FREQ == 0:
                    plot_exploration_entropy(
                        entropy_history,
                        entropy_timestep_history,
                        log_dir,
                    )

            episode_observations = []
            replay_buffer.end_episode()
            obs, _ = env.reset()
            normalizer.update(obs)
            normalized_obs = normalizer.normalize(obs, clip_obs=config.CLIP_OBS)
            actor_state = agent.get_initial_actor_state(batch_size=1)

        if step % config.PLOT_FREQ == 0:
            _update_training_progress_plot(
                log_dir=log_dir,
                plot_timesteps=training_plot_timesteps,
                plot_rewards=training_plot_rewards,
                current_timestep=step,
            )
            if latest_stats is not None:
                print(
                    f"Step {step:>8d} | "
                    f"CriticL {latest_stats['critic_loss']:.4f} | "
                    f"ActorL {latest_stats['actor_loss']:.4f} | "
                    f"Alpha {latest_stats['alpha_value']:.4f}"
                )

        if step % config.CHECKPOINT_FREQ == 0:
            checkpoint_prefix = "sac_recurrent_emission_model"
            checkpoint_base = os.path.join(
                checkpoints_dir, f"{checkpoint_prefix}_{step}_steps"
            )
            checkpoint_path = agent.save(checkpoint_base, total_timesteps=step)
            vec_checkpoint_path = os.path.join(
                checkpoints_dir,
                f"{checkpoint_prefix}_{step}_steps_vecnormalize.pkl",
            )
            normalizer.save(vec_checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
            print(f"Saved VecNormalize checkpoint to {vec_checkpoint_path}")

    training_duration_seconds = time.perf_counter() - training_start_time
    training_duration_hms = str(
        datetime.timedelta(seconds=int(training_duration_seconds))
    )
    train_config["training_duration_seconds"] = round(training_duration_seconds, 3)
    train_config["training_duration_hms"] = training_duration_hms

    replay_buffer.end_episode()
    _update_training_progress_plot(
        log_dir=log_dir,
        plot_timesteps=training_plot_timesteps,
        plot_rewards=training_plot_rewards,
        current_timestep=config.TOTAL_TIMESTEPS,
    )
    plot_exploration_entropy(
        entropy_history,
        entropy_timestep_history,
        log_dir,
    )

    final_model_base = os.path.join(log_dir, "sac_recurrent_emission_final")
    final_model_path = agent.save(
        final_model_base, total_timesteps=config.TOTAL_TIMESTEPS
    )
    normalizer.save(os.path.join(log_dir, "vec_normalize.pkl"))
    with open(os.path.join(log_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    print(
        f"Training finished in {training_duration_seconds:.2f}s "
        f"({training_duration_hms})."
    )
    print(f"Final model saved to {final_model_path}")
    print(f"Model and VecNormalize stats saved to {log_dir}")

    print("Evaluating Model...")
    evaluate_model(
        final_model_base,
        eval_log_dir=log_dir,
        train_config=train_config,
        use_thermal=args.use_thermal,
        agent_device=args.agent_device,
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or continue training a Recurrent SAC emission-control model."
    )
    parser.add_argument(
        "--continue_from",
        type=str,
        default=None,
        help="Path to an existing model (.zip) to continue training from",
    )
    parser.add_argument(
        "--agent_device",
        type=str,
        default="auto",
        help=(
            "PyTorch device for recurrent SAC actor/critic networks "
            "(e.g. 'cpu', 'cuda', 'auto')."
        ),
    )
    parser.add_argument(
        "--use_thermal",
        action="store_true",
        default=False,
        help=(
            "Use EmissionControlEnvThermal (10-dim observation space that includes "
            "aftertreatment temperatures T_gas_eo, T_Sub_DPF, T_gas_tp)."
        ),
    )
    args = parser.parse_args()
    main(args)
