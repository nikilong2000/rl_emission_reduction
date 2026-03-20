import os
import sys
import argparse
import datetime
import json
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

try:
    from ...env import EmissionControlEnv
    from ...env_thermal import EmissionControlEnvThermal
    from ...plotting import (
        plot_evaluation,
        plot_actions,
        plot_state_visitation_1d,
        plot_state_visitation_2d,
        plot_action_distribution,
        plot_state_action_occupancy,
        plot_temporal_state_heatmap,
    )
    from . import config
    from .recurrent_sac_core import (
        RecurrentSACAgent,
        RunningMeanStd,
        resolve_model_path,
    )
except ImportError:
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
    )
    import config
    from recurrent_sac_core import (
        RecurrentSACAgent,
        RunningMeanStd,
        resolve_model_path,
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
            f.write(f"Partial ({distance_km:.2f} km), {nox_per_km:.2f}, {nox_pass}\n")


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


def evaluate_model(
    model_path,
    eval_log_dir=None,
    train_config=None,
    use_thermal=False,
    agent_device="auto",
):
    resolved_model_path = resolve_model_path(model_path)
    print(f"Loading Recurrent SAC model from {resolved_model_path}...")
    agent, _ = RecurrentSACAgent.load_for_evaluation(
        resolved_model_path, device=agent_device
    )

    if eval_log_dir is None:
        base_log_dir = os.path.join(
            os.path.dirname(os.path.dirname(current_dir)), "logs", "new_sac_recurrent"
        )
        run_name = datetime.datetime.now().strftime("eval_%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_log_dir, run_name)
    else:
        log_dir = eval_log_dir
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging evaluation results to {log_dir}")

    wltc_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(current_dir))),
        "internal_lstm_models",
        "NN_Application",
        "Input_data",
        "WLTC.csv",
    )

    env = (
        EmissionControlEnvThermal(dataset_path=wltc_path)
        if use_thermal
        else EmissionControlEnv(dataset_path=wltc_path)
    )

    normalizer = None
    vec_norm_path = _find_vec_normalize_path(resolved_model_path)
    if vec_norm_path:
        normalizer = RunningMeanStd(shape=(env.observation_space.shape[0],))
        normalizer.load(vec_norm_path)
        print(f"Loaded VecNormalize stats from {vec_norm_path}")
    else:
        print(
            "Warning: Could not find vec_normalize.pkl. Evaluation might be inaccurate."
        )

    obs, _ = env.reset()
    actor_state = agent.get_initial_actor_state(batch_size=1)
    done = False
    total_reward = 0.0

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
    }

    if use_thermal:
        eval_results["T_gas_eo_K"] = []
        eval_results["T_Sub_DPF_K"] = []
        eval_results["T_gas_tp_K"] = []

    eval_results["speed_actual"].append(float(obs[0]))
    eval_results["speed_target"].append(float(obs[0] + obs[1]))
    eval_results["soc"].append(float(obs[2]))
    eval_results["ice_torque"].append(float(obs[3]))
    eval_results["nox"].append(float(obs[4]))
    eval_results["fuel"].append(0.0)
    eval_results["engine_on"].append(False)
    eval_results["ice_speed_rpm"].append(0.0)
    eval_results["em2_torque_nm"].append(0.0)
    eval_results["brake_perc"].append(0.0)

    if use_thermal:
        eval_results["T_gas_eo_K"].append(float(obs[7]) if len(obs) > 7 else 0.0)
        eval_results["T_Sub_DPF_K"].append(float(obs[8]) if len(obs) > 8 else 0.0)
        eval_results["T_gas_tp_K"].append(float(obs[9]) if len(obs) > 9 else 0.0)

    while not done:
        normalized_obs = (
            normalizer.normalize(obs, clip_obs=config.CLIP_OBS)
            if normalizer is not None
            else np.asarray(obs, dtype=np.float32)
        )
        action, actor_state = agent.select_action(
            normalized_obs, actor_state, deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += reward

        eval_results["speed_actual"].append(float(obs[0]))
        eval_results["speed_target"].append(float(info.get("target_speed", obs[0] + obs[1])))
        eval_results["soc"].append(float(obs[2]))
        eval_results["ice_torque"].append(float(obs[3]))
        eval_results["nox"].append(float(obs[4]))
        eval_results["fuel"].append(float(info.get("fuel", 0.0)))
        eval_results["engine_on"].append(bool(info.get("engine_on", False)))
        eval_results["ice_speed_rpm"].append(float(info.get("ice_speed_rpm", 0.0)))
        eval_results["em2_torque_nm"].append(float(info.get("em2_torque_nm", 0.0)))
        eval_results["brake_perc"].append(float(info.get("brake_perc", 0.0)))

        if use_thermal:
            eval_results["T_gas_eo_K"].append(
                float(info.get("t_gas_eo_K", obs[7] if len(obs) > 7 else 0.0))
            )
            eval_results["T_Sub_DPF_K"].append(
                float(info.get("t_sub_dpf_K", obs[8] if len(obs) > 8 else 0.0))
            )
            eval_results["T_gas_tp_K"].append(
                float(info.get("t_gas_tp_K", obs[9] if len(obs) > 9 else 0.0))
            )

    print(f"Evaluation finished. Total Reward: {total_reward}")

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

    metrics = {"model_path": model_path}

    continued_run = False
    continued_from = None
    if train_config is not None:
        metrics["configuration"] = train_config
        continued_run = train_config.get("continued_run", False)
        continued_from = train_config.get("continued_from", None)

    metrics.update(
        {
            "continued_run": continued_run,
            "continued_from": continued_from,
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
    plot_evaluation(eval_results, log_dir)
    plot_actions(eval_results, log_dir, window_start=1, window_size=3600)
    plot_state_visitation_1d([eval_results], ["SAC_Recurrent"], log_dir)
    plot_state_visitation_2d(eval_results, log_dir)
    plot_action_distribution([eval_results], ["SAC_Recurrent"], log_dir)
    plot_state_action_occupancy(eval_results, log_dir)
    plot_temporal_state_heatmap(eval_results, log_dir)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Recurrent SAC model.")
    parser.add_argument(
        "model_path", type=str, help="Path to the trained Recurrent SAC model (.zip)"
    )
    parser.add_argument(
        "--use_thermal",
        action="store_true",
        default=False,
        help="Use EmissionControlEnvThermal (10-dim obs with aftertreatment temps)",
    )
    parser.add_argument(
        "--agent_device",
        type=str,
        default="auto",
        help="PyTorch device for the recurrent SAC agent (e.g. 'cpu', 'cuda', 'auto').",
    )
    args = parser.parse_args()

    resolved_model_path = resolve_model_path(args.model_path)
    model_dir = os.path.dirname(os.path.abspath(resolved_model_path))
    train_config = None
    for search_dir in [model_dir, os.path.dirname(model_dir)]:
        candidate = os.path.join(search_dir, "train_config.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                train_config = json.load(f)
            print(f"Loaded train config from {candidate}")
            break

    evaluate_model(
        args.model_path,
        eval_log_dir=model_dir,
        use_thermal=args.use_thermal,
        train_config=train_config,
        agent_device=args.agent_device,
    )
