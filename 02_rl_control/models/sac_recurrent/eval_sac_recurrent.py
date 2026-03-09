import os
import sys
import json
import datetime
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
_RL_CONTROL_DIR = os.path.abspath(os.path.join(current_dir, "..", ".."))

_MODEL_DIR = current_dir
_RL_DIR = _RL_CONTROL_DIR


def env_creator(env_config):
    """Create environment — identical to the one used during training."""
    import sys
    import os
    import numpy as np
    import gymnasium as gym

    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    for p in [_MODEL_DIR, _RL_DIR]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import config  # noqa: F811

    import tensorflow as tf
    tf.config.run_functions_eagerly(True)

    from utils.platform_utils import configure_environment, configure_tf_devices

    configure_environment()
    configure_tf_devices()

    if env_config.get("use_thermal", False):
        from env_thermal import EmissionControlEnvThermal

        base_env = EmissionControlEnvThermal(
            dataset_path=env_config.get("dataset_path")
        )
    else:
        from env import EmissionControlEnv

        base_env = EmissionControlEnv(
            dataset_path=env_config.get("dataset_path")
        )

    return gym.wrappers.TransformObservation(
        base_env,
        lambda obs: np.clip(
            obs,
            base_env.observation_space.low,
            base_env.observation_space.high,
        ),
        observation_space=base_env.observation_space,
    )


def calculate_emissions_per_km(results, log_dir):
    speed_actual = np.array(results["speed_actual"])
    nox_gs = np.array(results["nox"])
    dt = 0.5

    distance_km = 0.0
    accumulated_nox_mg = 0.0
    km_counter = 1

    out_path = os.path.join(log_dir, "emissions_per_km.txt")
    with open(out_path, "w") as f:
        f.write("Kilometer, NOx (mg/km), NOx_Pass (<=80 mg/km)\n")

        for v, nox in zip(speed_actual, nox_gs):
            dist_step = v * dt / 3600.0
            nox_step_mg = nox * dt * 1000.0
            distance_km += dist_step
            accumulated_nox_mg += nox_step_mg

            if distance_km >= 1.0:
                nox_pass = accumulated_nox_mg <= 80.0
                f.write(f"{km_counter}, {accumulated_nox_mg:.2f}, {nox_pass}\n")
                distance_km = 0.0
                accumulated_nox_mg = 0.0
                km_counter += 1

        if distance_km > 0.1:
            nox_per_km = accumulated_nox_mg / distance_km
            nox_pass = nox_per_km <= 80.0
            f.write(
                f"Partial ({distance_km:.2f} km), {nox_per_km:.2f}, {nox_pass}\n"
            )


def evaluate_model(
    checkpoint_path, eval_log_dir=None, train_config=None, use_thermal=False
):
    import tensorflow as tf
    tf.config.run_functions_eagerly(True)

    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.tune.registry import register_env

    wltc_path = os.path.join(
        _RL_CONTROL_DIR,
        "..",
        "internal_lstm_models",
        "NN_Application",
        "Input_data",
        "WLTC.csv",
    )

    ray.init(num_gpus=1, log_to_driver=False)
    register_env("EmissionControlEnv", env_creator)

    # Load algorithm from checkpoint
    print(f"Loading checkpoint from {checkpoint_path} ...")
    algo = Algorithm.from_checkpoint(checkpoint_path)

    # Create a local env for deterministic rollout
    env = env_creator(
        {"use_thermal": use_thermal, "dataset_path": wltc_path}
    )

    if eval_log_dir is None:
        base_log_dir = os.path.join(_RL_CONTROL_DIR, "logs", "sac_recurrent")
        run_name = datetime.datetime.now().strftime("eval_%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = eval_log_dir
    print(f"Logging evaluation results to {log_dir}")

    # Get policy and initial LSTM state
    policy = algo.get_policy()
    state = policy.get_initial_state()

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    eval_results = {
        "speed_actual": [float(obs[0])],
        "speed_target": [float(obs[0] + obs[1])],
        "soc": [float(obs[2])],
        "ice_torque": [float(obs[3])],
        "nox": [float(obs[4])],
        "fuel": [0.0],
        "engine_on": [False],
        "ice_speed_rpm": [0.0],
        "em2_torque_nm": [0.0],
        "brake_perc": [0.0],
    }

    step_count = 0
    while not done:
        action, state, _ = algo.compute_single_action(
            obs, state=state, explore=False
        )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        eval_results["speed_actual"].append(float(obs[0]))
        eval_results["speed_target"].append(
            float(info.get("target_speed", obs[0] + obs[1]))
        )
        eval_results["soc"].append(float(obs[2]))
        eval_results["ice_torque"].append(float(obs[3]))
        eval_results["nox"].append(float(obs[4]))
        eval_results["fuel"].append(float(info.get("fuel", 0.0)))
        eval_results["engine_on"].append(bool(info.get("engine_on", False)))
        eval_results["ice_speed_rpm"].append(
            float(info.get("ice_speed_rpm", 0.0))
        )
        eval_results["em2_torque_nm"].append(
            float(info.get("em2_torque_nm", 0.0))
        )
        eval_results["brake_perc"].append(float(info.get("brake_perc", 0.0)))

    print(
        f"Evaluation finished — {step_count} steps, "
        f"Total Reward: {total_reward:.2f}"
    )

    # --- Metrics ---
    speed_actual = np.array(eval_results["speed_actual"])
    speed_target = np.array(eval_results["speed_target"])
    fuel_mg = np.array(eval_results["fuel"])
    nox_gs = np.array(eval_results["nox"])
    soc = np.array(eval_results["soc"])

    dt = 0.5
    total_fuel_g = np.sum(fuel_mg) / 1000.0
    total_nox_g = np.sum(nox_gs) * dt

    speed_error = speed_actual - speed_target
    mae_speed = np.mean(np.abs(speed_error))
    rmse_speed = np.sqrt(np.mean(speed_error**2))

    initial_soc = soc[0]
    final_soc = soc[-1]
    delta_soc = final_soc - initial_soc

    metrics = {"checkpoint_path": checkpoint_path}
    if train_config is not None:
        metrics["configuration"] = train_config

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
            "custom_notes": "",
        }
    )

    metrics_path = os.path.join(log_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    df_res = pd.DataFrame(eval_results)
    csv_path = os.path.join(log_dir, "evaluation_data.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Evaluation data saved to {csv_path}")

    calculate_emissions_per_km(eval_results, log_dir)

    # Plotting (reuse existing plotting module)
    try:
        sys.path.insert(0, _RL_DIR)
        from plotting import (
            plot_evaluation,
            plot_actions,
            plot_state_visitation_1d,
            plot_state_visitation_2d,
            plot_action_distribution,
            plot_state_action_occupancy,
            plot_temporal_state_heatmap,
        )

        plot_evaluation(eval_results, log_dir)
        plot_actions(eval_results, log_dir, window_start=1, window_size=3600)
        plot_state_visitation_1d(
            [eval_results], ["SAC_Recurrent"], log_dir
        )
        plot_state_visitation_2d(eval_results, log_dir)
        plot_action_distribution(
            [eval_results], ["SAC_Recurrent"], log_dir
        )
        plot_state_action_occupancy(eval_results, log_dir)
        plot_temporal_state_heatmap(eval_results, log_dir)
    except Exception as e:
        print(f"Plotting error (non-fatal): {e}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Recurrent SAC (Ray RLlib) checkpoint."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the Ray RLlib checkpoint directory.",
    )
    parser.add_argument(
        "--use_thermal",
        action="store_true",
        default=False,
        help="Use EmissionControlEnvThermal (10-dim obs with aftertreatment temps)",
    )
    args = parser.parse_args()

    # Try to load train_config from nearby files
    train_config = None
    for search_dir in [
        args.checkpoint_path,
        os.path.dirname(args.checkpoint_path),
        os.path.dirname(os.path.dirname(args.checkpoint_path)),
    ]:
        candidate = os.path.join(search_dir, "train_config.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                train_config = json.load(f)
            print(f"Loaded train config from {candidate}")
            break

    evaluate_model(
        args.checkpoint_path,
        eval_log_dir=os.path.dirname(args.checkpoint_path),
        use_thermal=args.use_thermal,
        train_config=train_config,
    )
