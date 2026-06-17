# Phase 1 — Speed-Tracking HPO + Seed Validation

Goal: find best hyperparameters for pure speed tracking, then validate stability across 10 seeds. No emission / SOC terms in reward.

## Stage 1a — Optuna HPO

### Launcher
- Script: [tune_hpo.py](tune_hpo.py)
- SLURM: [submit_hpo.sh](submit_hpo.sh) — `#SBATCH --array=1-10`, `N_TRIALS=2` per node → **20 trials total per algorithm**.
- Algorithms: PPO, SAC, TD3 (one study per algorithm).

### Optuna study
- Storage: `JournalStorage(JournalFileStorage(study_journal.log))` — append-only, crash-safe, no SQL daemon needed.
- Sampler: `TPESampler(n_startup_trials=10)`.
- Pruner: `MedianPruner(n_startup_trials=5, n_warmup_steps=2_000_000, interval_steps=500_000)`.
- Direction: `minimize` (objective = speed RMSE in km/h).
- `load_if_exists=True` → workers across SLURM array share one study and resume after crashes.

### Per-trial training
- `total_timesteps = 4_000_000`.
- Training env: `SubprocVecEnv` with `n_envs=8`, wrapped in `VecNormalize(norm_obs=on_policy, norm_reward=on_policy, clip_obs=10.0)` (i.e. obs+reward normalised for PPO; only obs untouched for SAC/TD3 since they use replay buffers).
- Eval env: `DummyVecEnv` of one `EmissionControlEnv(eval_mode=True, random_target=True)`, separate `VecNormalize(norm_reward=False)`; sync stats from training env each eval.
- Driving cycles: random target speed schedule per episode (`random_target=True`).
- Seed: `trial.number * 1000 + rank` per env.
- Per-trial directory: `logs/<algo>/optuna/trial_NNN/` (params, monitor CSVs, model zip, vec_normalize.pkl, tensorboard).

### Search spaces (`hpo_search_spaces.py`)

`net_arch` choices shared by all: `small=[64,64]`, `medium=[128,128]`, `large=[256,256]`, `deep=[128,128,128]`.

PPO:
| param | range / choices |
|---|---|
| learning_rate | log-uniform [1e-5, 1e-3] |
| n_steps | {1024, 2048, 4096} |
| batch_size | {64, 128, 256, 512} |
| n_epochs | int [3, 15] |
| gamma | uniform [0.95, 0.999] |
| gae_lambda | uniform [0.9, 1.0] |
| clip_range | uniform [0.1, 0.3] |
| net_arch | {small, medium, large, deep} |

SAC:
| param | range / choices |
|---|---|
| learning_rate | log-uniform [1e-5, 1e-3] |
| batch_size | {128, 256, 512, 1024} |
| tau | uniform [0.001, 0.02] |
| gamma | uniform [0.95, 0.999] |
| train_freq | {1, 4, 8} |
| gradient_steps | {1, 4, 8} |
| learning_starts | {1000, 5000, 10000, 20000} |
| ent_coef | {"auto", 0.01, 0.05, 0.1, 0.2} |
| use_sde | {True, False} |
| net_arch | {small, medium, large, deep} |

TD3:
| param | range / choices |
|---|---|
| learning_rate | log-uniform [1e-5, 1e-3] |
| batch_size | {128, 256, 512, 1024} |
| tau | uniform [0.001, 0.02] |
| gamma | uniform [0.95, 0.999] |
| train_freq | {1, 4, 8} |
| gradient_steps | {1, 4, 8} |
| learning_starts | {1000, 5000, 10000, 20000} |
| policy_delay | {1, 2, 4} |
| target_policy_noise | uniform [0.1, 0.4] |
| target_noise_clip | uniform [0.3, 0.7] |
| action_noise_sigma | uniform [0.05, 0.3] (NormalActionNoise) |
| net_arch | {small, medium, large, deep} |

### Evaluation callback (`TrialEvalCallback`)
- First eval at `2_000_000` steps (warmup), then every `500_000`.
- Each eval: 5 deterministic episodes, accumulate per-step squared `info["speed_error"]`, report `RMSE = sqrt(mean(squared_errors))` to `trial.report(rmse, step)`.
- `trial.should_prune()` → returns `False` from `_on_step` to stop `model.learn()` early.

### Reward (phase 1)
From `config_rewards.py` defaults:
```
W_SPEED = 1.0, W_EMISSION = 0.0, W_FUEL = 0.0,
W_BRAKE = 0.25, W_SOC = 0.0, W_SOC_SQUARED = 0.0,
W_FLICKER = 0.25
```
Reward formula (`env.py`, step):
```
reward = W_SPEED * exp(-0.5 * (speed_error / 10.0)^2)
       - W_EMISSION * min(NOx_tp / 0.4, 1.0)
       - W_FUEL    * (fuel_mg / 70.0)
       - W_BRAKE   * (brake_perc / 100.0)
       - W_SOC     * |SOC - SOC_init|
       - W_SOC_SQUARED * (SOC - SOC_init)^2
       - W_FLICKER  if engine just turned on
```
With phase-1 weights: only the Gaussian speed bonus, brake penalty, and engine-flicker penalty are active. Emission / SOC terms disabled.

### Observation (12-dim, min-max normalised to [0,1])
`[Car_Speed_kmph, Speed_Error_kmph, SOC, ICE_Torque_Nm, NOx_g_per_step, ICE_Speed_rpm, Fuel_mg, SOC_Error, Normalized_Timer, Last_EM2_Torque_Nm, Last_Brake_perc, T_Wall_SCR1_K]`

Action (4-dim, normalised to [-1, 1], rescaled in `step`):
`[ICE_Command, EM2_Torque_Nm ∈ [-421, 421], Fuel_mg ∈ [3, 70], Brake_perc ∈ [0, 100]]`

### Outputs
- `logs/<algo>/optuna/best_params.json` — resolved best hyperparameters (`net_arch` expanded to layer list, `ent_coef` cast to float when not `"auto"`).
- `logs/<algo>/optuna/all_trials.csv` — every trial's value, state, duration, and sampled params.

## Stage 1b — 10-seed validation

### Launcher
- Script: [run_seeds.py](run_seeds.py)
- SLURM: [submit_seeds.sh](submit_seeds.sh) — `#SBATCH --array=0-9`, `SEED=$SLURM_ARRAY_TASK_ID`.

### Per-seed run
- Loads `best_params.json`, applies via `apply_config_override(config, overrides)` (sets `config.POLICY_KWARGS=dict(net_arch=...)`; everything else `setattr(config, KEY.upper(), value)`).
- `set_random_seed(seed)`.
- `n_envs = 20` `SubprocVecEnv`, env seed = `seed * 1000 + rank`.
- `VecNormalize(norm_obs=on_policy, norm_reward=on_policy, clip_obs=10.0)`.
- `total_timesteps = 4_000_000` (default).
- Callbacks: `CheckpointCallback(save_freq=100_000)` + custom `VecNormalizeCheckpointCallback` (mirrored frequency).
- After training: `model.save`, `env.save("vec_normalize.pkl")`, `evaluate_model(...)` from `models/eval.py` → `evaluation_metrics.json` (`rmse_speed_kmph`, `total_nox_g`, `delta_soc`, etc.) and `train_config.json` for provenance (algo, seed, weights, durations, override path).
- Output dir: `logs/<algo>/optuna/seeds/seed_N/`.

## Stage 1c — Best-seed selection (used as phase-2 starting point)

- Script: [select_best_seed.py](select_best_seed.py) — `find_best_seed()`.
- Reads every `seed_*/evaluation_metrics.json`, requires matching `<algo>_seed_<id>_final.zip`.
- Sort key: `(rmse_speed_kmph, total_nox_g)` — primary speed RMSE, tiebreaker NOx.
- Returns top row: `{seed, rmse_speed_kmph, total_nox_g, delta_soc, model_zip, seed_dir}`.
- Optional `--write_baseline` → `phase1_best_seed.json`.
