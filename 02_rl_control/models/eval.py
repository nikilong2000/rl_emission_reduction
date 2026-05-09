"""
Generic evaluation script for PPO, SAC, and TD3 emission-control agents.

Usage:
    python eval.py path/to/model.zip --algorithm ppo
    python eval.py path/to/model.zip --target_speed 120
    python eval.py path/to/model.zip --random_target

The algorithm is auto-detected from train_config.json when --algorithm is not
supplied explicitly.
"""

import os
import sys
import argparse
import datetime
import json
import numpy as np
import pandas as pd
import warnings

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(current_dir)

from env import EmissionControlEnv
from env_thermal import EmissionControlEnvThermal
from plotting import (
    plot_evaluation,
    plot_actions,
    plot_state_visitation_1d,
    plot_state_visitation_2d,
    plot_action_distribution,
    plot_state_action_occupancy,
    plot_temporal_state_heatmap,
    plot_drivetrain_plus,
)
from utils.evaluation_utils import calculate_emissions_per_km
from utils.config_utils import load_config

# Suppress sklearn warnings about feature names
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

# ---------------------------------------------------------------------------
# Algorithm dispatch
# ---------------------------------------------------------------------------
ALGO_CLASSES = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


def evaluate_model(
    model_path,
    eval_log_dir=None,
    train_config=None,
    algorithm="ppo",
    use_thermal=False,
    random_target=False,
    target_speed=None,
):
    algo_key = algorithm.lower()
    AlgoClass = ALGO_CLASSES.get(algo_key)
    if AlgoClass is None:
        print(
            f"Error: Unknown algorithm '{algo_key}'. Choose from: {list(ALGO_CLASSES.keys())}"
        )
        sys.exit(1)

    # Validate model path
    if not (os.path.exists(model_path) or os.path.exists(model_path + ".zip")):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    print(f"Loading {algo_key.upper()} model from {model_path}...")
    model = AlgoClass.load(model_path)

    # Load algorithm-specific config for environment construction
    config = load_config(current_dir=current_dir, algo_key=algo_key)

    # Keep TensorFlow device setup aligned with training before env/model loading.
    try:
        from utils.platform_utils import configure_environment, configure_tf_devices

        configure_environment()
        configure_tf_devices()
    except ImportError:
        print(
            "Warning: utils.platform_utils not available. Continuing without TF device setup."
        )

    if eval_log_dir is None:
        base_log_dir = os.path.join(os.path.dirname(current_dir), "logs", algo_key)
        run_name = datetime.datetime.now().strftime("eval_%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = eval_log_dir
    print(f"Logging evaluation results to {log_dir}")

    # Initialize environment
    print(40 * "=", f"Evaluating {algo_key.upper()}...")

    def make_env():
        wltc_path = os.path.join(
            os.path.dirname(current_dir),
            "data_train",
            "WLTC.csv",
        )
        env_cls = EmissionControlEnvThermal if use_thermal else EmissionControlEnv
        if target_speed is not None:
            return env_cls(
                config_module=config, fixed_target_speed=target_speed, eval_mode=True
            )
        elif random_target:
            return env_cls(config_module=config, random_target=True, eval_mode=True)
        else:
            return env_cls(dataset_path=wltc_path, config_module=config, eval_mode=True)

    env = DummyVecEnv([make_env])

    # Get a reference to the underlying env for denormalizing [0,1] obs
    # back to physical units (km/h, Nm, rpm, etc.) for plotting
    underlying_env = env.envs[0].unwrapped

    model_dir = os.path.dirname(os.path.abspath(model_path))
    model_basename = os.path.splitext(os.path.basename(model_path))[0]

    # Keep only a reference to training configuration location in eval metrics.
    train_config_path = None
    for search_dir in [model_dir, os.path.dirname(model_dir)]:
        candidate = os.path.join(search_dir, "train_config.json")
        if os.path.exists(candidate):
            train_config_path = candidate
            break

    # Priority: (1) per-checkpoint pkl, (2) same-dir vec_normalize.pkl,
    # (3) parent-dir vec_normalize.pkl (when model is in a checkpoints/ subdir)
    vec_norm_path = os.path.join(model_dir, f"{model_basename}_vecnormalize.pkl")
    if not os.path.exists(vec_norm_path):
        vec_norm_path = os.path.join(model_dir, "vec_normalize.pkl")
    if not os.path.exists(vec_norm_path):
        vec_norm_path = os.path.join(os.path.dirname(model_dir), "vec_normalize.pkl")

    vec_normalized = False

    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
        vec_normalized = True
        print(f"Loaded VecNormalize stats from {vec_norm_path}")
    else:
        print(
            "Warning: Could not find vec_normalize.pkl. Evaluation might be inaccurate."
        )

    obs = env.reset()
    done = False
    total_reward = 0

    # Data collection for plotting
    eval_results = {
        "speed_actual": [],
        "speed_target": [],
        "soc": [],
        "ice_torque": [],
        "nox": [],
        "fuel": [],
        "engine_on": [],
        "ice_speed_rpm": [],
        "em2_torque_nm": [],
        "brake_perc": [],
        "em1_torque_ist_nm": [],
        "em2_torque_ist_nm": [],
        "sun_speed_rpm": [],
        "ring_speed_rpm": [],
    }

    # Helper to convert [0,1]-normalised env obs back to physical units
    def _to_physical(obs_01):
        return underlying_env._denormalize_obs(obs_01)

    # Store initial state
    raw_obs_01 = env.get_original_obs()[0] if vec_normalized else obs[0]
    raw_obs = _to_physical(raw_obs_01)

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        done = dones[0]
        total_reward += rewards[0]

        i = infos[0]
        # VecEnv auto-resets on done=True, so use terminal_observation to log
        # the true final state of the finished episode instead of the next reset state.
        if done and "terminal_observation" in i:
            terminal_obs = i["terminal_observation"]
            if vec_normalized:
                raw_obs_01 = env.unnormalize_obs(np.array([terminal_obs]))[0]
            else:
                raw_obs_01 = terminal_obs
        else:
            raw_obs_01 = env.get_original_obs()[0] if vec_normalized else obs[0]

        # Denormalize from [0,1] env scale back to physical units
        raw_obs = _to_physical(raw_obs_01)

        eval_results["speed_actual"].append(raw_obs[0])
        eval_results["speed_target"].append(
            i.get("target_speed", raw_obs[0] + raw_obs[1])
        )
        eval_results["soc"].append(raw_obs[2])
        eval_results["ice_torque"].append(raw_obs[3])
        eval_results["nox"].append(raw_obs[4])
        eval_results["fuel"].append(i.get("fuel"))
        eval_results["engine_on"].append(i.get("engine_on"))
        eval_results["ice_speed_rpm"].append(i.get("ice_speed_rpm"))
        eval_results["em2_torque_nm"].append(i.get("em2_torque_nm"))
        eval_results["brake_perc"].append(i.get("brake_perc"))
        eval_results["em1_torque_ist_nm"].append(i.get("em1_torque_ist_nm", np.nan))
        eval_results["em2_torque_ist_nm"].append(i.get("em2_torque_ist_nm", np.nan))
        eval_results["sun_speed_rpm"].append(i.get("sun_speed_rpm", np.nan))
        eval_results["ring_speed_rpm"].append(i.get("ring_speed_rpm", np.nan))

    print(f"Evaluation finished. Total Reward: {total_reward}")

    # --- Calculate Metrics ---
    speed_actual = np.array(eval_results["speed_actual"])
    speed_target = np.array(eval_results["speed_target"])
    fuel_mg = np.array(eval_results["fuel"])
    nox_gs = np.array(eval_results["nox"])
    soc = np.array(eval_results["soc"])

    dt = 0.5  # since measurements have 2hz frequency (2 steps == 1 second)
    total_fuel_g = np.sum(fuel_mg) / 1000.0
    total_nox_g = np.sum(nox_gs) * dt

    speed_error = speed_actual - speed_target
    mae_speed = np.mean(np.abs(speed_error))
    rmse_speed = np.sqrt(np.mean(speed_error**2))

    initial_soc = soc[0]
    final_soc = soc[-1]
    delta_soc = final_soc - initial_soc
    soc_drift = soc - initial_soc
    max_abs_soc_drift = float(np.max(np.abs(soc_drift)))
    rms_soc_drift = float(np.sqrt(np.mean(soc_drift**2)))

    # Keep evaluation metrics focused on evaluation outputs only.
    # Training configuration is already persisted in train_config.json.
    metrics = {
        "model_path": model_path,
        "train_config_path": train_config_path,
    }

    metrics.update(
        {
            "total_reward": float(total_reward),
            "total_fuel_g": float(total_fuel_g),
            "total_nox_g": float(total_nox_g),
            "mae_speed_kmph": float(mae_speed),
            "rmse_speed_kmph": float(rmse_speed),
            "initial_soc": float(initial_soc),
            "final_soc": float(final_soc),
            "delta_soc": float(delta_soc),
            "max_abs_soc_drift": max_abs_soc_drift,
            "rms_soc_drift": rms_soc_drift,
            "custom_notes": "",
        }
    )

    metrics_path = os.path.join(log_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

    df_res = pd.DataFrame(eval_results)
    csv_path = os.path.join(log_dir, "evaluation_data.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nEvaluation data saved to {csv_path}")

    # plot analysis plots
    algo_label = algo_key.upper()
    calculate_emissions_per_km(eval_results, log_dir)
    plot_evaluation(eval_results, log_dir, config)
    plot_actions(eval_results, log_dir, window_start=1, window_size=3600)
    plot_state_visitation_1d([eval_results], [algo_label], log_dir)
    plot_state_visitation_2d(eval_results, log_dir)
    plot_action_distribution([eval_results], [algo_label], log_dir)
    plot_state_action_occupancy(eval_results, log_dir)
    plot_temporal_state_heatmap(eval_results, log_dir)
    plot_drivetrain_plus(eval_results, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO/SAC/TD3 emission-control model."
    )
    parser.add_argument("model_path", type=str, help="Path to the trained model (.zip)")
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["ppo", "sac", "td3"],
        help="RL algorithm (auto-detected from train_config.json if omitted).",
    )
    parser.add_argument(
        "--use_thermal",
        action="store_true",
        default=False,
        help="Use EmissionControlEnvThermal (10-dim obs with aftertreatment temps).",
    )
    parser.add_argument(
        "--random_target",
        action="store_true",
        help="Evaluate with random target speeds.",
    )
    parser.add_argument(
        "--target_speed",
        type=float,
        default=None,
        help="Evaluate with a specific fixed target speed (km/h). Implies --random_target.",
    )
    args = parser.parse_args()

    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    # Try to load train_config.json from the run dir or parent (for checkpoints/ subdir)
    train_config = None
    for search_dir in [model_dir, os.path.dirname(model_dir)]:
        candidate = os.path.join(search_dir, "train_config.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                train_config = json.load(f)
            print(f"Loaded train config from {candidate}")
            break

    # Auto-detect algorithm from train_config if not provided
    algorithm = args.algorithm
    if algorithm is None:
        if train_config is not None and "algorithm" in train_config:
            algorithm = train_config["algorithm"].lower()
            print(f"Auto-detected algorithm: {algorithm.upper()}")
        else:
            print(
                "Error: --algorithm is required when train_config.json is not available or missing 'algorithm' key."
            )
            sys.exit(1)

    use_thermal = args.use_thermal or bool(
        train_config is not None and train_config.get("use_thermal", False)
    )
    # Auto-detect random_target from train_config unless explicitly provided
    random_target = args.random_target or bool(
        train_config is not None and train_config.get("random_target", False)
    )

    evaluate_model(
        args.model_path,
        eval_log_dir=model_dir,
        train_config=train_config,
        algorithm=algorithm,
        use_thermal=use_thermal,
        random_target=random_target,
        target_speed=args.target_speed,
    )
