"""
Evaluate a trained Recurrent SAC (Ray RLlib) checkpoint.

Mirrors the evaluation pipeline of the SB3-based models (SAC, PPO, TD3)
so that results are directly comparable.  Produces:

  evaluation_metrics.json   – summary metrics (reward, fuel, NOx, speed MAE/RMSE, SOC)
  evaluation_data.csv       – full episode trajectory
  emissions_per_km.txt      – per-kilometre NOx compliance (≤80 mg/km)
  evaluation_results.png    – speed / SOC / NOx / (thermal) overview
  action_results.png        – engine-on, ICE RPM, EM2 torque, fuel, brake
  state_visitation_1d.png   – marginal state histograms
  state_visitation_2d.png   – 2-D joint state occupancy
  action_distribution.png   – action histograms
  state_action_occupancy.png – state-action heatmaps
  temporal_state_heatmap.png – time-indexed state density
"""

import os
import sys
import json
import pickle
import datetime
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

try:
    from ...utils.evaluation_utils import calculate_emissions_per_km
except ImportError:
    from utils.evaluation_utils import calculate_emissions_per_km

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
_RL_CONTROL_DIR = os.path.abspath(os.path.join(current_dir, "..", ".."))

_MODEL_DIR = current_dir
_RL_DIR = _RL_CONTROL_DIR


# ------------------------------------------------------------------ #
#  Environment creator (identical to training)                        #
# ------------------------------------------------------------------ #


def env_creator(env_config):
    """Create environment — identical to the one used during training."""
    import sys as _sys
    import os as _os
    import numpy as _np
    import gymnasium as gym

    _os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    for p in [_MODEL_DIR, _RL_DIR]:
        if p not in _sys.path:
            _sys.path.insert(0, p)

    import config  # noqa: F811

    import tensorflow as tf

    tf.config.run_functions_eagerly(True)

    from utils.platform_utils import configure_environment, configure_tf_devices

    configure_environment()
    configure_tf_devices()

    if env_config.get("use_thermal", False):
        from env_thermal import EmissionControlEnvThermal

        base_env = EmissionControlEnvThermal(
            dataset_path=env_config.get("dataset_path"),
            config_module=config,
        )
    else:
        from env import EmissionControlEnv

        base_env = EmissionControlEnv(
            dataset_path=env_config.get("dataset_path"),
            config_module=config,
        )

    return gym.wrappers.TransformObservation(
        base_env,
        lambda obs: _np.clip(
            obs,
            base_env.observation_space.low,
            base_env.observation_space.high,
        ),
        observation_space=base_env.observation_space,
    )


# ------------------------------------------------------------------ #
#  Resolve checkpoint path                                            #
# ------------------------------------------------------------------ #


def _resolve_checkpoint_dir(path: str) -> str:
    """Accept a checkpoint *directory* or the rllib_checkpoint.json *file*
    inside it and always return the directory."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")
    expected = os.path.join(path, "rllib_checkpoint.json")
    if not os.path.exists(expected):
        raise FileNotFoundError(
            f"No rllib_checkpoint.json found in {path}. "
            "Please point to a valid Ray RLlib checkpoint directory."
        )
    return path


# ------------------------------------------------------------------ #
#  Build algorithm from config + restore policy weights               #
# ------------------------------------------------------------------ #


def _build_and_restore(checkpoint_dir: str, use_thermal: bool):
    """Build a fresh RecurrentSAC, then load policy weights from checkpoint.

    We avoid ``Algorithm.from_checkpoint`` because the old-API-stack
    v1.1 checkpoints trigger optimizer-state size mismatches on restore.
    For evaluation we only need the policy weights, not the optimiser.
    """
    from ray.rllib.algorithms.sac import SACConfig
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env

    from recurrent_sac_model import RecurrentSACTorchModel, RecurrentSAC

    ModelCatalog.register_custom_model("recurrent_sac_model", RecurrentSACTorchModel)
    register_env("EmissionControlEnv", env_creator)

    # Load the training config to reconstruct the same algorithm
    run_dir = os.path.dirname(checkpoint_dir)
    train_cfg_path = os.path.join(run_dir, "train_config.json")
    if not os.path.exists(train_cfg_path):
        # Might be nested one more level (checkpoints/<name>)
        train_cfg_path = os.path.join(os.path.dirname(run_dir), "train_config.json")
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path) as f:
            tcfg = json.load(f)
    else:
        tcfg = {}

    sys.path.insert(0, _MODEL_DIR)
    import config as cfg_module

    sac_config = (
        SACConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="EmissionControlEnv",
            env_config={"use_thermal": use_thermal},
        )
        .framework("torch")
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=1,
            rollout_fragment_length=tcfg.get(
                "rollout_fragment_length", cfg_module.ROLLOUT_FRAGMENT_LENGTH
            ),
        )
        .training(
            lr=tcfg.get("learning_rate", cfg_module.LEARNING_RATE),
            tau=tcfg.get("tau", cfg_module.TAU),
            gamma=tcfg.get("gamma", cfg_module.GAMMA),
            train_batch_size=tcfg.get("batch_size", cfg_module.BATCH_SIZE),
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": tcfg.get("buffer_size", cfg_module.BUFFER_SIZE),
            },
            num_steps_sampled_before_learning_starts=10_000,
            target_network_update_freq=0,
        )
        .resources(num_gpus=0)
        .reporting(min_sample_timesteps_per_iteration=1000)
    )

    sac_config["model"] = {
        "custom_model": "recurrent_sac_model",
        "lstm_cell_size": tcfg.get("lstm_cell_size", cfg_module.LSTM_CELL_SIZE),
        "max_seq_len": tcfg.get("max_seq_len", cfg_module.MAX_SEQ_LEN),
        "fcnet_hiddens": tcfg.get("fcnet_hiddens", cfg_module.FCNET_HIDDENS),
        "fcnet_activation": "relu",
    }

    # Build fresh algorithm
    algo = RecurrentSAC(config=sac_config)

    # Restore policy weights only (skip optimizer state)
    policy_pkl = os.path.join(
        checkpoint_dir, "policies", "default_policy", "policy_state.pkl"
    )
    if not os.path.exists(policy_pkl):
        raise FileNotFoundError(f"Policy state not found at {policy_pkl}")

    with open(policy_pkl, "rb") as f:
        policy_state = pickle.load(f)

    policy = algo.get_policy()
    policy.set_weights(policy_state["weights"])
    # Also restore target network (important for consistent Q-values)
    if hasattr(policy, "target_models"):
        for target_model in policy.target_models.values():
            target_model.load_state_dict(policy.model.state_dict())

    return algo


# ------------------------------------------------------------------ #
#  Main evaluation                                                    #
# ------------------------------------------------------------------ #


def evaluate_model(
    checkpoint_path,
    eval_log_dir=None,
    train_config=None,
    use_thermal=False,
):
    import tensorflow as tf

    tf.config.run_functions_eagerly(True)

    import ray

    wltc_path = os.path.join(
        _RL_CONTROL_DIR,
        "..",
        "internal_lstm_models",
        "NN_Application",
        "Input_data",
        "WLTC.csv",
    )

    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_path)

    ray.init(num_gpus=1, log_to_driver=False)

    print(f"Loading checkpoint from {checkpoint_dir} ...")
    algo = _build_and_restore(checkpoint_dir, use_thermal)
    print("Algorithm restored successfully.")

    # Create a local env for deterministic WLTC rollout
    env = env_creator({"use_thermal": use_thermal, "dataset_path": wltc_path})

    if eval_log_dir is None:
        base_log_dir = os.path.join(_RL_CONTROL_DIR, "logs", "sac_recurrent")
        run_name = datetime.datetime.now().strftime("eval_%Y%m%d_%H%M%S")
        log_dir = os.path.join(base_log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = eval_log_dir
        os.makedirs(log_dir, exist_ok=True)
    print(f"Logging evaluation results to {log_dir}")

    # ── Get policy and initial LSTM state ──
    policy = algo.get_policy()
    state = policy.get_initial_state()

    obs, info = env.reset()
    done = False
    total_reward = 0.0

    # Initialise results dict (matches SB3 eval structure exactly)
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

    # Thermal keys (populated only when use_thermal=True)
    if use_thermal:
        eval_results["T_gas_eo_K"] = [float(obs[7]) if len(obs) > 7 else 0.0]
        eval_results["T_Sub_DPF_K"] = [float(obs[8]) if len(obs) > 8 else 0.0]
        eval_results["T_gas_tp_K"] = [float(obs[9]) if len(obs) > 9 else 0.0]

    # ── Deterministic rollout ──
    step_count = 0
    while not done:
        action, state, _ = algo.compute_single_action(obs, state=state, explore=False)
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
        eval_results["nox"].append(float(info.get("nox", obs[4])))
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

    print(
        f"Evaluation finished — {step_count} steps, "
        f"Total Reward: {total_reward:.2f}"
    )

    # ── Metrics (matching SB3 eval_sac.py) ──
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

    continued_run = False
    continued_from = None
    metrics = {"model_path": str(checkpoint_dir)}

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

    # ── Evaluation data CSV ──
    df_res = pd.DataFrame(eval_results)
    csv_path = os.path.join(log_dir, "evaluation_data.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Evaluation data saved to {csv_path}")

    # ── Emission compliance ──
    calculate_emissions_per_km(eval_results, log_dir)

    # ── Plots (reuse existing plotting module) ──
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
    print("  ✓ evaluation_results.png")

    plot_actions(eval_results, log_dir, window_start=1, window_size=3600)
    print("  ✓ action_results.png")

    plot_state_visitation_1d([eval_results], ["SAC_Recurrent"], log_dir)
    print("  ✓ state_visitation_1d.png")

    plot_state_visitation_2d(eval_results, log_dir)
    print("  ✓ state_visitation_2d.png")

    plot_action_distribution([eval_results], ["SAC_Recurrent"], log_dir)
    print("  ✓ action_distribution.png")

    plot_state_action_occupancy(eval_results, log_dir)
    print("  ✓ state_action_occupancy.png")

    plot_temporal_state_heatmap(eval_results, log_dir)
    print("  ✓ temporal_state_heatmap.png")

    algo.stop()
    ray.shutdown()
    print("Evaluation complete.")


# ------------------------------------------------------------------ #
#  CLI entry-point                                                    #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Recurrent SAC (Ray RLlib) checkpoint."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help=(
            "Path to the Ray RLlib checkpoint directory "
            "(or to the rllib_checkpoint.json file inside it)."
        ),
    )
    parser.add_argument(
        "--use_thermal",
        action="store_true",
        default=False,
        help="Use EmissionControlEnvThermal (10-dim obs with aftertreatment temps)",
    )
    parser.add_argument(
        "--eval_log_dir",
        type=str,
        default=None,
        help="Directory to save evaluation outputs (default: parent of checkpoint dir).",
    )
    args = parser.parse_args()

    # Resolve checkpoint directory
    ckpt_dir = _resolve_checkpoint_dir(args.checkpoint_path)
    run_dir = os.path.dirname(ckpt_dir)

    # Determine eval output directory
    if args.eval_log_dir:
        eval_log_dir = args.eval_log_dir
    else:
        eval_log_dir = run_dir

    # Load train_config from the run directory hierarchy
    train_config = None
    for search_dir in [ckpt_dir, run_dir, os.path.dirname(run_dir)]:
        candidate = os.path.join(search_dir, "train_config.json")
        if os.path.exists(candidate):
            with open(candidate) as f:
                train_config = json.load(f)
            print(f"Loaded train config from {candidate}")
            break

    evaluate_model(
        ckpt_dir,
        eval_log_dir=eval_log_dir,
        use_thermal=args.use_thermal,
        train_config=train_config,
    )
