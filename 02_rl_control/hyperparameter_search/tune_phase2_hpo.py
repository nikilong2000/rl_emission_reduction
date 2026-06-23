"""
Phase-2 Optuna HPO over (W_EMISSION, W_SOC_SQUARED).

Each trial:
  1. Samples (w_emission, w_soc_squared) via TPE.
  2. Loads phase-1 best-seed checkpoint + its VecNormalize stats.
  3. Fine-tunes for --trial_timesteps with the sampled reward weights.
  4. Runs full deterministic WLTC evaluation.
  5. Returns composite scalar:
       score = total_nox_g
             + lambda_rmse * max(0, rmse_speed - 5)^2
             + lambda_soc  * max(0, max_abs_soc_drift - 0.05)^2

Intermediate RMSE is reported each 500k steps for MedianPruner.

Usage:
    python tune_phase2_hpo.py --algorithm ppo --n_trials 20
    python tune_phase2_hpo.py --algorithm sac --n_trials 2  # array-task style
"""

import os
import sys
import json
import argparse
import datetime
import time
import warnings
import numpy as np
import pandas as pd
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
rl_control_dir = os.path.dirname(current_dir)
models_dir = os.path.join(rl_control_dir, "models")
sys.path.insert(0, rl_control_dir)
sys.path.insert(0, models_dir)
sys.path.insert(0, current_dir)
from env import EmissionControlEnv
from misc.env_thermal import EmissionControlEnvThermal
from models.eval import evaluate_model
from utils.config_utils import load_config
from hpo_search_spaces import apply_config_override
from select_best_seed import find_best_seed
from tune_hpo import TrialEvalCallback, N_EVAL_EPISODES, EVAL_FREQ_TIMESTEPS, PRUNE_WARMUP_STEPS

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)

ALGO_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}
ALGO_ON_POLICY = {"ppo": True, "sac": False, "td3": False}

# Search ranges (locked by user)
W_EMISSION_LO, W_EMISSION_HI = 0.25, 1.25
W_SOC_SQUARED_LO, W_SOC_SQUARED_HI = 25.0, 250.0


def _build_continue_kwargs(algo_key, config, env, log_dir, device):
    """Match models/train.py continue-from path: skip policy_kwargs and
    learning_starts so checkpoint architecture and replay-buffer state are
    preserved."""
    kwargs = {
        "verbose": 1,
        "tensorboard_log": log_dir,
        "device": device,
    }
    if algo_key == "ppo":
        kwargs.update(
            {
                "learning_rate": config.LEARNING_RATE,
                "n_steps": config.N_STEPS,
                "batch_size": config.BATCH_SIZE,
                "n_epochs": config.N_EPOCHS,
                "gamma": config.GAMMA,
                "gae_lambda": config.GAE_LAMBDA,
                "clip_range": config.CLIP_RANGE,
            }
        )
    elif algo_key == "sac":
        kwargs.update(
            {
                "learning_rate": config.LEARNING_RATE,
                "buffer_size": config.BUFFER_SIZE,
                "batch_size": config.BATCH_SIZE,
                "tau": config.TAU,
                "gamma": config.GAMMA,
                "train_freq": config.TRAIN_FREQ,
                "gradient_steps": config.GRADIENT_STEPS,
                "ent_coef": config.ENT_COEF,
                "target_entropy": config.TARGET_ENTROPY,
                "use_sde": config.USE_SDE,
                "sde_sample_freq": config.SDE_SAMPLE_FREQ,
            }
        )
    elif algo_key == "td3":
        n_actions = env.action_space.shape[-1]
        kwargs.update(
            {
                "learning_rate": config.LEARNING_RATE,
                "buffer_size": config.BUFFER_SIZE,
                "batch_size": config.BATCH_SIZE,
                "tau": config.TAU,
                "gamma": config.GAMMA,
                "train_freq": config.TRAIN_FREQ,
                "gradient_steps": config.GRADIENT_STEPS,
                "policy_delay": config.POLICY_DELAY,
                "target_policy_noise": config.TARGET_POLICY_NOISE,
                "target_noise_clip": config.TARGET_NOISE_CLIP,
                "action_noise": NormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=config.ACTION_NOISE_SIGMA * np.ones(n_actions),
                ),
            }
        )
    return kwargs


def objective(
    trial,
    algo_key,
    args,
    base_log_dir,
    phase1_best,
    hpo_overrides,
):
    """Train one fine-tune trial and return composite NOx/RMSE/SOC score."""
    AlgoClass = ALGO_CLASSES[algo_key]
    is_on_policy = ALGO_ON_POLICY[algo_key]

    # Sample reward weights
    w_emission = trial.suggest_float("w_emission", W_EMISSION_LO, W_EMISSION_HI)
    w_soc_squared = trial.suggest_float(
        "w_soc_squared", W_SOC_SQUARED_LO, W_SOC_SQUARED_HI
    )

    trial_dir = os.path.join(
        base_log_dir,
        f"trial_{trial.number:03d}_we{w_emission:.3f}_wsq{w_soc_squared:.1f}",
    )
    os.makedirs(trial_dir, exist_ok=True)
    print(f"\n=== Trial {trial.number}: we={w_emission:.4f}, wsq={w_soc_squared:.2f} ===")

    # Fresh config + phase-1 HPO overrides + injected reward weights
    config = load_config(current_dir=models_dir, algo_key=algo_key)
    apply_config_override(config, hpo_overrides)
    config.W_EMISSION = w_emission
    config.W_SOC_SQUARED = w_soc_squared
    config.W_SOC = 0.0
    config.W_FUEL = 0.0

    with open(os.path.join(trial_dir, "trial_params.json"), "w") as f:
        json.dump(
            {"w_emission": w_emission, "w_soc_squared": w_soc_squared},
            f,
            indent=4,
        )

    # Training env
    env_cls = EmissionControlEnvThermal if args.use_thermal else EmissionControlEnv

    def make_env(rank):
        def _init():
            e = env_cls(config_module=config, random_target=True)
            e.reset(seed=trial.number * 1000 + rank)
            return Monitor(e, os.path.join(trial_dir, str(rank)))

        return _init

    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    vec_norm_path = os.path.join(phase1_best["seed_dir"], "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = True
        env.norm_reward = is_on_policy
        print(f"Loaded VecNormalize stats from {vec_norm_path}")
    else:
        print(f"Warning: vec_normalize.pkl missing at {vec_norm_path}; starting fresh.")
        env = VecNormalize(
            env, norm_obs=is_on_policy, norm_reward=is_on_policy, clip_obs=10.0
        )

    # Eval env for pruning (deterministic RMSE only)
    def make_eval_env():
        return env_cls(config_module=config, random_target=True, eval_mode=True)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(
        eval_env, norm_obs=is_on_policy, norm_reward=False, clip_obs=10.0
    )

    # Load fine-tune model from phase-1 ckpt
    model_kwargs = _build_continue_kwargs(
        algo_key, config, env, trial_dir, args.agent_device
    )
    print(f"Loading {algo_key.upper()} ckpt from {phase1_best['model_zip']}")
    model = AlgoClass.load(phase1_best["model_zip"], env=env, **model_kwargs)

    eval_callback = TrialEvalCallback(
        trial=trial,
        eval_env=eval_env,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq_timesteps=EVAL_FREQ_TIMESTEPS,
    )

    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=args.trial_timesteps, callback=[eval_callback])
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        env.close()
        eval_env.close()
        raise optuna.TrialPruned(f"Trial failed: {e}")

    if eval_callback.is_pruned:
        print(f"Trial {trial.number} pruned at step {model.num_timesteps}")
        env.close()
        eval_env.close()
        raise optuna.TrialPruned()

    duration = time.perf_counter() - t0
    duration_hms = str(datetime.timedelta(seconds=int(duration)))
    print(f"Trial {trial.number} training finished in {duration_hms}")

    # Persist final model + normaliser
    final_name = f"{algo_key}_phase2_trial_{trial.number:03d}_final"
    model_path = os.path.join(trial_dir, final_name)
    model.save(model_path)
    env.save(os.path.join(trial_dir, "vec_normalize.pkl"))

    train_config = {
        "algorithm": algo_key.upper(),
        "phase": "2_optuna",
        "trial_number": trial.number,
        "continued_from": phase1_best["model_zip"],
        "phase1_best_seed": phase1_best["seed"],
        "total_timesteps": args.trial_timesteps,
        "n_envs": args.n_envs,
        "w_speed": config.W_SPEED,
        "w_emission": config.W_EMISSION,
        "w_fuel": config.W_FUEL,
        "w_brake": config.W_BRAKE,
        "w_soc": config.W_SOC,
        "w_soc_squared": config.W_SOC_SQUARED,
        "w_flicker": config.W_FLICKER,
        "training_duration_seconds": round(duration, 3),
        "training_duration_hms": duration_hms,
    }
    with open(os.path.join(trial_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)
    env.close()
    eval_env.close()

    # Full deterministic WLTC eval
    print(f"Trial {trial.number}: running full evaluation")
    evaluate_model(
        model_path,
        eval_log_dir=trial_dir,
        train_config=train_config,
        algorithm=algo_key,
        use_thermal=args.use_thermal,
        random_target=True,
    )

    # Composite score
    with open(os.path.join(trial_dir, "evaluation_metrics.json")) as f:
        m = json.load(f)
    rmse = m["rmse_speed_kmph"]
    nox_g = m["total_nox_g"]
    max_soc = m["max_abs_soc_drift"]
    rmse_pen = args.lambda_rmse * max(0.0, rmse - 5.0) ** 2
    soc_pen = args.lambda_soc * max(0.0, max_soc - 0.05) ** 2
    score = nox_g + rmse_pen + soc_pen

    # Attach metrics to trial user attrs for all_trials.csv
    trial.set_user_attr("rmse_speed_kmph", rmse)
    trial.set_user_attr("total_nox_g", nox_g)
    trial.set_user_attr("max_abs_soc_drift", max_soc)
    trial.set_user_attr("rms_soc_drift", m.get("rms_soc_drift"))
    trial.set_user_attr("delta_soc", m.get("delta_soc"))
    trial.set_user_attr("rmse_penalty", rmse_pen)
    trial.set_user_attr("soc_penalty", soc_pen)

    print(
        f"Trial {trial.number}: rmse={rmse:.3f} nox={nox_g:.3f}g "
        f"max|dSOC|={max_soc:.4f} -> score={score:.4f}"
    )
    return score


def main(args):
    algo_key = args.algorithm.lower()
    if algo_key not in ALGO_CLASSES:
        print(f"Unknown algorithm '{algo_key}'.")
        sys.exit(1)

    try:
        from utils.platform_utils import configure_environment, configure_tf_devices

        configure_environment()
        configure_tf_devices()
    except ImportError:
        pass

    # Resolve phase-1 best ckpt + HPO config
    seeds_dir = args.seeds_dir or os.path.join(
        rl_control_dir,
        "logs_cluster_phase1",
        "logs",
        algo_key,
        "optuna",
        "seeds",
    )
    phase1_best, _ = find_best_seed(seeds_dir, algo_key)
    print(
        f"Phase-1 best seed: {phase1_best['seed']} "
        f"(rmse={phase1_best['rmse_speed_kmph']:.3f}, "
        f"nox={phase1_best['total_nox_g']:.3f}g)"
    )

    hpo_config_path = args.hpo_config or os.path.join(
        rl_control_dir,
        "logs_cluster_phase1",
        "logs",
        algo_key,
        "optuna",
        "best_params.json",
    )
    with open(hpo_config_path) as f:
        hpo_overrides = json.load(f)
    print(f"Applied HPO overrides from {hpo_config_path}")

    # Study dir
    base_log_dir = os.path.join(rl_control_dir, "logs", algo_key, "phase2_optuna")
    os.makedirs(base_log_dir, exist_ok=True)
    print(f"Optuna log directory: {base_log_dir}")

    journal_path = os.path.join(base_log_dir, "study_journal.log")
    storage = JournalStorage(JournalFileStorage(journal_path))
    study_name = args.study_name or f"{algo_key}_phase2_hpo"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=PRUNE_WARMUP_STEPS,
        interval_steps=EVAL_FREQ_TIMESTEPS,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
        pruner=pruner,
        load_if_exists=True,
    )
    print(
        f"Study '{study_name}' — {len(study.trials)} existing trials, "
        f"scheduling {args.n_trials} new trials"
    )
    study.optimize(
        lambda trial: objective(
            trial, algo_key, args, base_log_dir, phase1_best, hpo_overrides
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )

    print("\n" + "=" * 60)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials yet — best_params not written.")
        return
    print(f"Best trial:  {study.best_trial.number}")
    print(f"Best score:  {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    best_params_path = os.path.join(base_log_dir, "best_params_phase2.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print(f"\nBest params -> {best_params_path}")

    rows = []
    for t in study.trials:
        if t.datetime_start and t.datetime_complete:
            duration_s = (t.datetime_complete - t.datetime_start).total_seconds()
        else:
            duration_s = None
        row = {
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "datetime_start": t.datetime_start,
            "datetime_complete": t.datetime_complete,
            "duration_s": duration_s,
        }
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)
    csv_path = os.path.join(base_log_dir, "all_trials.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"All trials -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm", required=True, choices=["ppo", "sac", "td3"]
    )
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--trial_timesteps", type=int, default=4_000_000)
    parser.add_argument("--lambda_rmse", type=float, default=20.0)
    parser.add_argument("--lambda_soc", type=float, default=1000.0)
    parser.add_argument("--agent_device", default="auto")
    parser.add_argument(
        "--seeds_dir",
        default=None,
        help="Phase-1 seeds dir. Default: logs_cluster_phase1/logs/<algo>/optuna/seeds.",
    )
    parser.add_argument(
        "--hpo_config",
        default=None,
        help="Phase-1 best_params.json. Default: logs_cluster_phase1/logs/<algo>/optuna/best_params.json.",
    )
    parser.add_argument("--use_thermal", action="store_true", default=False)
    parser.add_argument(
        "--study_name",
        default=None,
        help="Optuna study name (default: <algo>_phase2_hpo).",
    )
    main(parser.parse_args())
