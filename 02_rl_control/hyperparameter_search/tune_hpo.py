"""
Optuna hyperparameter optimisation for SB3 RL emission-control agents.
Objective:  minimise RMSE between actual and target vehicle speed.
Storage:    JournalFileStorage (append-only log, no SQL, crash-safe).
Pruner:     MedianPruner — first check at 2 M steps, then every 500 k.
Usage:
    python tune_hpo.py --algorithm ppo --n_trials 50
    python tune_hpo.py --algorithm sac --n_trials 100 --n_jobs 4
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    sync_envs_normalization,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_control_dir = os.path.dirname(current_dir)
models_dir = os.path.join(rl_control_dir, "models")
sys.path.insert(0, rl_control_dir)
sys.path.insert(0, models_dir)
sys.path.insert(0, current_dir)
from env import EmissionControlEnv
from utils.config_utils import load_config
from hpo_search_spaces import sample_params, resolve_params

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALGO_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}
ALGO_ON_POLICY = {"ppo": True, "sac": False, "td3": False}
N_EVAL_EPISODES = 5
EVAL_FREQ_TIMESTEPS = 500_000
PRUNE_WARMUP_STEPS = 2_000_000


# ---------------------------------------------------------------------------
# Evaluation callback for Optuna pruning
# ---------------------------------------------------------------------------
class TrialEvalCallback(BaseCallback):
    """
    Periodically evaluate the current policy and report RMSE to Optuna.
    Pruning is handled by Optuna's MedianPruner via trial.should_prune().
    """

    def __init__(
        self,
        trial,
        eval_env,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq_timesteps=EVAL_FREQ_TIMESTEPS,
        verbose=0,
    ):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq_timesteps = eval_freq_timesteps
        self.eval_freq_calls = None  # set in _init_callback
        self.is_pruned = False
        self.last_rmse = float("inf")

    def _init_callback(self):
        n_envs = self.model.get_env().num_envs
        self.eval_freq_calls = max(1, self.eval_freq_timesteps // n_envs)

    def _on_step(self):
        if self.eval_freq_calls and self.n_calls % self.eval_freq_calls == 0:
            rmse = self._evaluate()
            self.last_rmse = rmse
            step = self.model.num_timesteps
            self.trial.report(rmse, step)
            if self.trial.should_prune():
                self.is_pruned = True
                return False  # stops model.learn()
        return True

    def _evaluate(self):
        """Run deterministic eval episodes and return speed-tracking RMSE."""
        training_env = self.model.get_env()
        if isinstance(training_env, VecNormalize) and isinstance(
            self.eval_env, VecNormalize
        ):
            sync_envs_normalization(training_env, self.eval_env)
            self.eval_env.training = False
            self.eval_env.norm_reward = False
        squared_errors = []
        episodes_done = 0
        obs = self.eval_env.reset()
        while episodes_done < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, infos = self.eval_env.step(action)
            if "speed_error" in infos[0]:
                squared_errors.append(infos[0]["speed_error"] ** 2)
            if dones[0]:
                episodes_done += 1
        # Restore training mode on eval env
        if isinstance(self.eval_env, VecNormalize):
            self.eval_env.training = True
        if squared_errors:
            return float(np.sqrt(np.mean(squared_errors)))
        return float("inf")


# ---------------------------------------------------------------------------
# Model-building helpers
# ---------------------------------------------------------------------------
def _build_trial_model_kwargs(algo_key, resolved, config, env, log_dir, device):
    """Build SB3 constructor kwargs from resolved HPO params + config defaults."""
    kwargs = {
        "verbose": 1,
        "tensorboard_log": log_dir,
        "device": device,
    }
    net_arch = resolved.get("net_arch")
    if net_arch is not None:
        kwargs["policy_kwargs"] = dict(net_arch=net_arch)
    elif hasattr(config, "POLICY_KWARGS"):
        kwargs["policy_kwargs"] = config.POLICY_KWARGS
    if algo_key == "ppo":
        kwargs.update(
            {
                "learning_rate": resolved.get("learning_rate", config.LEARNING_RATE),
                "n_steps": resolved.get("n_steps", config.N_STEPS),
                "batch_size": resolved.get("batch_size", config.BATCH_SIZE),
                "n_epochs": resolved.get("n_epochs", config.N_EPOCHS),
                "gamma": resolved.get("gamma", config.GAMMA),
                "gae_lambda": resolved.get("gae_lambda", config.GAE_LAMBDA),
                "clip_range": resolved.get("clip_range", config.CLIP_RANGE),
            }
        )
    elif algo_key == "sac":
        kwargs.update(
            {
                "learning_rate": resolved.get("learning_rate", config.LEARNING_RATE),
                "buffer_size": config.BUFFER_SIZE,
                "batch_size": resolved.get("batch_size", config.BATCH_SIZE),
                "tau": resolved.get("tau", config.TAU),
                "gamma": resolved.get("gamma", config.GAMMA),
                "train_freq": resolved.get("train_freq", config.TRAIN_FREQ),
                "gradient_steps": resolved.get("gradient_steps", config.GRADIENT_STEPS),
                "learning_starts": resolved.get(
                    "learning_starts", config.LEARNING_STARTS
                ),
                "ent_coef": resolved.get("ent_coef", config.ENT_COEF),
                "target_entropy": config.TARGET_ENTROPY,
                "use_sde": resolved.get("use_sde", config.USE_SDE),
                "sde_sample_freq": config.SDE_SAMPLE_FREQ,
            }
        )
    elif algo_key == "td3":
        n_actions = env.action_space.shape[-1]
        sigma = resolved.get("action_noise_sigma", config.ACTION_NOISE_SIGMA)
        kwargs.update(
            {
                "learning_rate": resolved.get("learning_rate", config.LEARNING_RATE),
                "buffer_size": config.BUFFER_SIZE,
                "batch_size": resolved.get("batch_size", config.BATCH_SIZE),
                "tau": resolved.get("tau", config.TAU),
                "gamma": resolved.get("gamma", config.GAMMA),
                "train_freq": resolved.get("train_freq", config.TRAIN_FREQ),
                "gradient_steps": resolved.get("gradient_steps", config.GRADIENT_STEPS),
                "learning_starts": resolved.get(
                    "learning_starts", config.LEARNING_STARTS
                ),
                "policy_delay": resolved.get("policy_delay", config.POLICY_DELAY),
                "target_policy_noise": resolved.get(
                    "target_policy_noise", config.TARGET_POLICY_NOISE
                ),
                "target_noise_clip": resolved.get(
                    "target_noise_clip", config.TARGET_NOISE_CLIP
                ),
                "action_noise": NormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=sigma * np.ones(n_actions),
                ),
            }
        )
    return kwargs


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------
def objective(trial, algo_key, total_timesteps, n_envs, base_log_dir, device):
    """Train one SB3 agent and return speed-tracking RMSE."""
    trial_dir = os.path.join(base_log_dir, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)
    # Fresh config (thread-safe: each call returns a new module object)
    config = load_config(current_dir=models_dir, algo_key=algo_key)
    # Sample & resolve hyperparameters
    params = sample_params(trial, algo_key)
    resolved = resolve_params(params)
    # Log trial params
    with open(os.path.join(trial_dir, "trial_params.json"), "w") as f:
        json.dump(resolved, f, indent=4, default=str)
    AlgoClass = ALGO_CLASSES[algo_key]
    is_on_policy = ALGO_ON_POLICY[algo_key]

    # -- Training environment (SubprocVecEnv) --
    def make_env(rank):
        def _init():
            e = EmissionControlEnv(config_module=config, random_target=True)
            e.reset(seed=trial.number * 1000 + rank)
            return Monitor(e, os.path.join(trial_dir, str(rank)))

        return _init

    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    env = VecNormalize(
        env, norm_obs=is_on_policy, norm_reward=is_on_policy, clip_obs=10.0
    )

    # -- Evaluation environment (DummyVecEnv, deterministic schedule) --
    def make_eval_env():
        return EmissionControlEnv(
            config_module=config, random_target=True, eval_mode=True
        )

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(
        eval_env, norm_obs=is_on_policy, norm_reward=False, clip_obs=10.0
    )
    # Build model
    model_kwargs = _build_trial_model_kwargs(
        algo_key, resolved, config, env, trial_dir, device
    )
    model = AlgoClass("MlpPolicy", env, **model_kwargs)
    # Eval callback for pruning
    eval_callback = TrialEvalCallback(
        trial=trial,
        eval_env=eval_env,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq_timesteps=EVAL_FREQ_TIMESTEPS,
    )
    # Train
    try:
        model.learn(total_timesteps=total_timesteps, callback=[eval_callback])
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
    # Final RMSE
    rmse = eval_callback.last_rmse
    # Persist model & normalizer
    model.save(os.path.join(trial_dir, f"{algo_key}_trial_{trial.number}"))
    env.save(os.path.join(trial_dir, "vec_normalize.pkl"))
    env.close()
    eval_env.close()
    print(f"Trial {trial.number} finished — RMSE: {rmse:.4f}")
    return rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    algo_key = args.algorithm.lower()
    if algo_key not in ALGO_CLASSES:
        print(f"Unknown algorithm '{algo_key}'. Choose from: {list(ALGO_CLASSES)}")
        sys.exit(1)
    # TF device setup (for LSTM plant models inside the env)
    try:
        from utils.platform_utils import configure_environment, configure_tf_devices

        configure_environment()
        configure_tf_devices()
    except ImportError:
        pass
    # Log directory
    base_log_dir = os.path.join(rl_control_dir, "logs", algo_key, "optuna")
    os.makedirs(base_log_dir, exist_ok=True)
    print(f"HPO log directory: {base_log_dir}")
    # Optuna study — file-based storage (no SQL)
    journal_path = os.path.join(base_log_dir, "study_journal.log")
    storage = JournalStorage(JournalFileStorage(journal_path))
    study_name = args.study_name or f"{algo_key}_hpo"
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=PRUNE_WARMUP_STEPS,
        interval_steps=EVAL_FREQ_TIMESTEPS,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",  # minimise RMSE
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        pruner=pruner,
        load_if_exists=True,  # resume crashed / timed-out studies
    )
    print(
        f"Study '{study_name}' — {len(study.trials)} existing trials, "
        f"scheduling {args.n_trials} new trials"
    )
    study.optimize(
        lambda trial: objective(
            trial, algo_key, args.trial_timesteps, args.n_envs, base_log_dir, args.agent_device
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )
    # ---- Save results ----
    print("\n" + "=" * 60)
    print(f"Best trial:  {study.best_trial.number}")
    print(f"Best RMSE:   {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    best_resolved = resolve_params(study.best_trial.params)
    best_params_path = os.path.join(base_log_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_resolved, f, indent=4, default=str)
    print(f"\nBest (resolved) params → {best_params_path}")
    # All-trials summary CSV
    rows = []
    for t in study.trials:
        row = {"number": t.number, "value": t.value, "state": str(t.state)}
        row.update(t.params)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(base_log_dir, "all_trials.csv"), index=False)
    print(f"All trials  → {os.path.join(base_log_dir, 'all_trials.csv')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna HPO for PPO / SAC / TD3 emission-control agents."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["ppo", "sac", "td3"],
        help="RL algorithm to tune.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        required=True,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel Optuna workers (1 = sequential, safe for shared storage).",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=8,
        help="Number of parallel environments (should match node core count).",
    )
    parser.add_argument(
        "--trial_timesteps",
        type=int,
        default=4_000_000,
        help="Training timesteps per trial.",
    )
    parser.add_argument(
        "--agent_device",
        type=str,
        default="auto",
        help="PyTorch device for the RL agent (cpu / cuda / auto).",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Optuna study name (default: {algo}_hpo).",
    )
    main(parser.parse_args())
