import os
import sys
import json
import datetime
import warnings
import argparse
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
_RL_CONTROL_DIR = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Paths captured as constants for pickling into Ray workers
_MODEL_DIR = current_dir
_RL_DIR = _RL_CONTROL_DIR


def env_creator(env_config):
    """Create environment for Ray workers. Called in each worker process."""
    import sys
    import os
    import gymnasium as gym
    import numpy as np

    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    for p in [_MODEL_DIR, _RL_DIR]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import config  # noqa: F811 — our sac_recurrent/config.py, cached for env.py

    # Ensure TF eager execution is enabled (Ray + torch framework may disable it)
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)

    from utils.platform_utils import configure_environment, configure_tf_devices

    configure_environment()
    configure_tf_devices()

    if env_config.get("use_thermal", False):
        from env_thermal import EmissionControlEnvThermal

        base_env = EmissionControlEnvThermal()
    else:
        from env import EmissionControlEnv

        base_env = EmissionControlEnv()

    # Clip observations to declared bounds (env simulation can slightly exceed them)
    return gym.wrappers.TransformObservation(
        base_env,
        lambda obs: np.clip(
            obs,
            base_env.observation_space.low,
            base_env.observation_space.high,
        ),
        observation_space=base_env.observation_space,
    )


def main(args):
    sys.path.insert(0, _MODEL_DIR)
    sys.path.insert(0, _RL_DIR)

    # Ensure TF eager execution before any TF model loading
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)

    import config

    import ray
    from ray.rllib.algorithms.sac import SACConfig
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env

    from recurrent_sac_model import RecurrentSACTorchModel, RecurrentSAC

    # Register custom model
    ModelCatalog.register_custom_model(
        "recurrent_sac_model", RecurrentSACTorchModel
    )

    # Create Log Directory
    base_log_dir = os.path.join(_RL_CONTROL_DIR, "logs", "sac_recurrent")
    run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, run_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Logging to {log_dir}")

    # Initialize Ray — expose the GPU to the learner
    ray.init(num_gpus=1, log_to_driver=True)
    register_env("EmissionControlEnv", env_creator)

    # ------------------------------------------------------------------ #
    #  Build SAC config with custom recurrent model (old API stack)       #
    # ------------------------------------------------------------------ #
    sac_config = (
        SACConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="EmissionControlEnv",
            env_config={"use_thermal": args.use_thermal},
        )
        .framework("torch")
        .env_runners(
            num_env_runners=config.NUM_ENV_RUNNERS,
            num_envs_per_env_runner=1,
            rollout_fragment_length=config.ROLLOUT_FRAGMENT_LENGTH,
        )
        .training(
            lr=config.LEARNING_RATE,
            tau=config.TAU,
            gamma=config.GAMMA,
            train_batch_size=config.BATCH_SIZE,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": config.BUFFER_SIZE,
            },
            num_steps_sampled_before_learning_starts=10_000,
            target_network_update_freq=0,  # 0 = soft update every step
        )
        .resources(
            num_gpus=1,
        )
        .reporting(
            min_sample_timesteps_per_iteration=1000,
        )
    )

    # Custom recurrent model — no use_lstm flag needed (LSTM is built-in)
    sac_config["model"] = {
        "custom_model": "recurrent_sac_model",
        "lstm_cell_size": config.LSTM_CELL_SIZE,
        "max_seq_len": config.MAX_SEQ_LEN,
        "fcnet_hiddens": config.FCNET_HIDDENS,
        "fcnet_activation": "relu",
    }

    # ------------------------------------------------------------------ #
    #  Training configuration snapshot                                    #
    # ------------------------------------------------------------------ #
    train_config = {
        "algorithm": "SAC_Recurrent (Ray RLlib)",
        "framework": "torch",
        "env": "thermal" if args.use_thermal else "base",
        "learning_rate": config.LEARNING_RATE,
        "buffer_size": config.BUFFER_SIZE,
        "batch_size": config.BATCH_SIZE,
        "tau": config.TAU,
        "gamma": config.GAMMA,
        "lstm_cell_size": config.LSTM_CELL_SIZE,
        "max_seq_len": config.MAX_SEQ_LEN,
        "fcnet_hiddens": config.FCNET_HIDDENS,
        "num_env_runners": config.NUM_ENV_RUNNERS,
        "rollout_fragment_length": config.ROLLOUT_FRAGMENT_LENGTH,
        "total_training_iterations": config.TOTAL_TRAINING_ITERATIONS,
        "checkpoint_freq": config.CHECKPOINT_FREQ,
        "w_speed": config.W_SPEED,
        "w_emission": config.W_EMISSION,
        "w_fuel": config.W_FUEL,
        "w_brake": config.W_BRAKE,
        "w_soc": config.W_SOC,
        "w_soc_squared": config.W_SOC_SQUARED,
        "w_flicker": config.W_FLICKER,
    }
    with open(os.path.join(log_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    # ------------------------------------------------------------------ #
    #  Build & train                                                      #
    # ------------------------------------------------------------------ #
    print("Building Recurrent SAC algorithm with LSTM policy...")
    algo = RecurrentSAC(config=sac_config)
    print("Algorithm built successfully.")

    num_iters = config.TOTAL_TRAINING_ITERATIONS
    print(f"Starting training for {num_iters} iterations...")

    best_reward = -np.inf

    for i in range(1, num_iters + 1):
        result = algo.train()

        # Metrics can be at top level or nested under "env_runners"/"sampler_results"
        env_r = result.get("env_runners", result.get("sampler_results", {}))
        ep_rew = env_r.get(
            "episode_reward_mean",
            result.get("episode_reward_mean", float("nan")),
        )
        ep_len = env_r.get(
            "episode_len_mean",
            result.get("episode_len_mean", float("nan")),
        )
        ts = result.get("timesteps_total", 0)

        # SAC-specific learner info
        learner = result.get("info", {}).get("learner", {})
        default_p = learner.get("default_policy", {})
        lstats = default_p.get("learner_stats", default_p)
        critic_loss = lstats.get("critic_loss", float("nan"))
        actor_loss = lstats.get("actor_loss", float("nan"))
        alpha_val = lstats.get("alpha_value", float("nan"))

        print(
            f"Iter {i:>4d}/{num_iters} | "
            f"Timesteps: {ts:>8d} | "
            f"Reward: {ep_rew:>10.2f} | "
            f"EpLen: {ep_len:>7.0f} | "
            f"CriticL: {critic_loss:>8.3f} | "
            f"ActorL: {actor_loss:>8.3f} | "
            f"Alpha: {alpha_val:>6.3f}"
        )

        # Checkpoint
        if i % config.CHECKPOINT_FREQ == 0 or i == num_iters:
            ckpt = algo.save(checkpoint_dir)
            print(f"  Checkpoint saved: {ckpt}")

            if not np.isnan(ep_rew) and ep_rew > best_reward:
                best_reward = ep_rew
                best_ckpt = algo.save(os.path.join(log_dir, "best_checkpoint"))
                print(f"  New best checkpoint: {best_ckpt}")

    print("Training finished.")

    # Final save
    final_path = algo.save(os.path.join(log_dir, "final_checkpoint"))
    print(f"Final checkpoint: {final_path}")

    summary = {
        **train_config,
        "final_timesteps": result.get("timesteps_total", 0),
        "final_reward_mean": float(ep_rew) if not np.isnan(ep_rew) else None,
        "best_reward": float(best_reward) if not np.isnan(best_reward) else None,
        "episodes_completed": len(
            env_r.get("hist_stats", {}).get("episode_reward", [])
        ),
    }
    with open(os.path.join(log_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=4, default=str)

    algo.stop()
    ray.shutdown()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Recurrent SAC (LSTM) emission-control model with Ray RLlib."
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
    parser.add_argument(
        "--num_iters",
        type=int,
        default=None,
        help="Override TOTAL_TRAINING_ITERATIONS from config.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override NUM_ENV_RUNNERS from config (default 0 = local worker only).",
    )
    args = parser.parse_args()

    if args.num_iters is not None or args.num_workers is not None:
        sys.path.insert(0, current_dir)
        import config

        if args.num_iters is not None:
            config.TOTAL_TRAINING_ITERATIONS = args.num_iters
        if args.num_workers is not None:
            config.NUM_ENV_RUNNERS = args.num_workers

    main(args)
