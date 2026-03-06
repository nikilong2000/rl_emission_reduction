import os
import json
import datetime
import numpy as np
import warnings
import argparse

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

try:
    from ...env import EmissionControlEnv
    from ...env_thermal import EmissionControlEnvThermal
    from ...plotting import TrainingLivePlotCallback
    from . import config
    from .eval_sac import evaluate_model
    from ...utils import safety_utils
    from ...utils.platform_utils import configure_environment, configure_tf_devices
    from ...utils.checkpoint_utils import VecNormalizeCheckpointCallback
except ImportError:
    from env import EmissionControlEnv
    from env_thermal import EmissionControlEnvThermal
    from plotting import TrainingLivePlotCallback
    import config
    from eval_sac import evaluate_model
    from utils import safety_utils
    from utils.platform_utils import configure_environment, configure_tf_devices
    from utils.checkpoint_utils import VecNormalizeCheckpointCallback


def main(args):
    # GPU setup for the LSTM environment models (TensorFlow)
    configure_environment()
    configure_tf_devices()

    # Create Log Directory
    base_log_dir = os.path.join(
        os.path.dirname(os.path.dirname(current_dir)), "logs", "sac"
    )
    run_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env_cls = EmissionControlEnvThermal if args.use_thermal else EmissionControlEnv

    def make_env():
        e = env_cls()
        return Monitor(e, os.path.join(log_dir, "monitor.csv"))

    env = DummyVecEnv([make_env])

    # SAC config snapshot (written into evaluation_metrics.json for reproducibility)
    train_config = {
        "algorithm": "SAC",
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
        "use_sde": config.USE_SDE,
        "sde_sample_freq": config.SDE_SAMPLE_FREQ,
        "total_timesteps": config.TOTAL_TIMESTEPS,
        "w_speed": config.W_SPEED,
        "w_emission": config.W_EMISSION,
        "w_fuel": config.W_FUEL,
        "w_brake": config.W_BRAKE,
        "w_soc": config.W_SOC,
        "w_soc_squared": config.W_SOC_SQUARED,
        "w_flicker": config.W_FLICKER,
        "continued_run": args.continue_from is not None,
        "continued_from": args.continue_from,
    }

    # VecNormalize: normalise observations only.
    # SAC is off-policy — raw rewards must be stored in the replay buffer, so
    # reward normalisation is disabled to keep the value scale consistent.
    if args.continue_from:
        safety_utils.config_check(args.continue_from, train_config)

        vec_norm_path = os.path.join(
            os.path.dirname(args.continue_from), "vec_normalize.pkl"
        )
        if os.path.exists(vec_norm_path):
            env = VecNormalize.load(vec_norm_path, env)
            env.training = True
            env.norm_reward = False
            print(f"Loaded VecNormalize stats from {vec_norm_path}")
        else:
            print(
                "Warning: Could not find vec_normalize.pkl. Starting fresh normalizer."
            )
            env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # SAC is inherently stochastic — no explicit action noise object is needed.
    # Exploration is driven by the learned stochastic policy (entropy regularisation).

    if args.continue_from:
        print(
            f"Loading existing model from {args.continue_from} to continue training..."
        )
        model = SAC.load(
            args.continue_from,
            env=env,
            verbose=1,
            learning_rate=config.LEARNING_RATE,
            buffer_size=config.BUFFER_SIZE,
            batch_size=config.BATCH_SIZE,
            tau=config.TAU,
            gamma=config.GAMMA,
            train_freq=config.TRAIN_FREQ,
            gradient_steps=config.GRADIENT_STEPS,
            ent_coef=config.ENT_COEF,
            target_entropy=config.TARGET_ENTROPY,
            use_sde=config.USE_SDE,
            sde_sample_freq=config.SDE_SAMPLE_FREQ,
            tensorboard_log=log_dir,
            device=args.agent_device,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=config.LEARNING_RATE,
            buffer_size=config.BUFFER_SIZE,
            batch_size=config.BATCH_SIZE,
            tau=config.TAU,
            gamma=config.GAMMA,
            train_freq=config.TRAIN_FREQ,
            gradient_steps=config.GRADIENT_STEPS,
            learning_starts=config.LEARNING_STARTS,
            ent_coef=config.ENT_COEF,
            target_entropy=config.TARGET_ENTROPY,
            use_sde=config.USE_SDE,
            sde_sample_freq=config.SDE_SAMPLE_FREQ,
            tensorboard_log=log_dir,
            device=args.agent_device,
        )

    # Callbacks
    CHECKPOINT_FREQ = 100_000
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac_emission_model",
    )
    vec_normalize_checkpoint_callback = VecNormalizeCheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="sac_emission_model",
        vec_normalize=env,
    )
    plot_callback = TrainingLivePlotCallback(check_freq=1_000, log_dir=log_dir)

    # Train
    print("Starting SAC Training...")
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, vec_normalize_checkpoint_callback, plot_callback],
    )
    print("Training finished.")

    # Save
    model.save(os.path.join(log_dir, "sac_emission_final"))
    env.save(os.path.join(log_dir, "vec_normalize.pkl"))
    with open(os.path.join(log_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    print(f"Model and VecNormalize stats saved to {log_dir}")

    # Evaluate
    print("Evaluating Model...")
    evaluate_model(
        os.path.join(log_dir, "sac_emission_final"),
        eval_log_dir=log_dir,
        train_config=train_config,
        use_thermal=args.use_thermal,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or continue training a SAC emission-control model."
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
            "PyTorch device for the SAC actor/critic networks "
            "(e.g. 'cpu', 'cuda', 'auto'). "
            "TensorFlow automatically uses the GPU for LSTM inference regardless."
        ),
    )
    parser.add_argument(
        "--use_thermal",
        action="store_true",
        default=False,
        help=(
            "Use EmissionControlEnvThermal (10-dim observation space that includes "
            "aftertreatment temperatures T_gas_eo, T_Sub_DPF, T_gas_tp). "
            "Recommended when thermodynamic context is needed for global optimality."
        ),
    )
    parser.add_argument(
        "--use_sde",
        action="store_true",
        default=False,
        help=(
            "Enable State-Dependent Exploration (gSDE). "
            "Replaces diagonal Gaussian noise with a state-correlated noise process — "
            "can improve exploration in systems with slow thermodynamic dynamics."
        ),
    )
    args = parser.parse_args()

    # Allow CLI flag to override config value without editing config.py
    if args.use_sde:
        config.USE_SDE = True

    main(args)
