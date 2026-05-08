# Phase 2 — Reward-Weight Sweep with SOC + NOx Terms

Goal: starting from the phase-1 best-seed checkpoint, fine-tune under augmented reward (speed + emission + SOC squared) across a 3×3 grid to map the speed/emission/SOC trade-off.

## Launcher
- Script: [run_phase2_cell.py](run_phase2_cell.py)
- SLURM: [submit_phase2_sweep.sh](submit_phase2_sweep.sh) — `#SBATCH --array=0-8`. Each array task = one grid cell.

## Reward-weight grid (`run_phase2_cell.py`)

```python
W_EMISSION_GRID    = [0.25, 0.5, 1.0]
W_SOC_SQUARED_GRID = [50.0, 150.0, 400.0]
```

Mapping (`cell_id_to_weights`): `i_e, i_s = divmod(cell_id, 3)` → `(W_EMISSION_GRID[i_e], W_SOC_SQUARED_GRID[i_s])`.

| cell_id | W_EMISSION | W_SOC_SQUARED |
|---|---|---|
| 0 | 0.25 | 50  |
| 1 | 0.25 | 150 |
| 2 | 0.25 | 400 |
| 3 | 0.5  | 50  |
| 4 | 0.5  | 150 |
| 5 | 0.5  | 400 |
| 6 | 1.0  | 50  |
| 7 | 1.0  | 150 |
| 8 | 1.0  | 400 |

Other phase-2 weight overrides (forced in code):
```
config.W_SOC         = 0.0   # linear SOC term off — only quadratic active
config.W_FUEL        = 0.0
# kept from phase 1: W_SPEED=1.0, W_BRAKE=0.25, W_FLICKER=0.25
```

Active reward (phase 2):
```
reward = W_SPEED      * exp(-0.5 * (speed_error / 10.0)^2)
       - W_EMISSION   * min(NOx_tp / 0.4, 1.0)
       - W_BRAKE      * (brake_perc / 100.0)
       - W_SOC_SQUARED * (SOC - SOC_init)^2
       - W_FLICKER     if engine just turned on
```

## Continue-from-checkpoint logic

- Phase-1 best-seed selected via `find_best_seed(seeds_dir, algo_key)` — sort by `(rmse_speed_kmph, total_nox_g)`.
- Source files used:
  - Model: `<seed_dir>/<algo>_seed_<id>_final.zip`
  - VecNormalize: `<seed_dir>/vec_normalize.pkl`
- Default `seeds_dir`: `logs_cluster_01/logs/<algo>/optuna/seeds/`.

### Hyperparameter source
- `--hpo_config` defaults to `logs_cluster_01/logs/<algo>/optuna/best_params.json` (phase-1 winner).
- `apply_config_override(config, overrides)` re-applies the same per-algorithm hyperparams used in phase-1 seed runs.

### `_build_continue_kwargs` — deliberate omissions vs. fresh training
Differences from `run_seeds.py::_build_model_kwargs`:
- **No `policy_kwargs`** — preserves the loaded checkpoint's network architecture exactly.
- **No `learning_starts`** (SAC/TD3) — replay buffer state from phase 1 is reused; we don't want to re-enter random-action warmup.
- **No `buffer_size`** override — reuses checkpoint's existing buffer.

Per-algorithm kwargs forwarded: `learning_rate`, `gamma`, on-policy: `n_steps, batch_size, n_epochs, gae_lambda, clip_range`. Off-policy: `batch_size, tau, train_freq, gradient_steps, ent_coef, target_entropy, use_sde, sde_sample_freq` (SAC) and `policy_delay, target_policy_noise, target_noise_clip` + reconstructed `NormalActionNoise(sigma=action_noise_sigma)` (TD3).

### Env construction
- `EmissionControlEnv(config_module=config, random_target=True)` (or `EmissionControlEnvThermal` if `--use_thermal`).
- `SubprocVecEnv` with `n_envs = 20`, env seed = `cell_id * 1000 + rank`.
- VecNormalize handling:
  ```python
  env = VecNormalize.load(vec_norm_path, env)
  env.training    = True
  env.norm_reward = is_on_policy
  ```
  Reuses obs running stats. Reward running stats are stale (different reward shape), but SB3 re-accumulates within ~10k steps for PPO. Off-policy SAC/TD3 don't normalise rewards → no issue.
- Fallback: if `vec_normalize.pkl` missing, log a warning and start fresh `VecNormalize(...)`.

### Model load
```python
model = AlgoClass.load(ckpt_path, env=env, **model_kwargs)
```
Drops in fresh env + new HPs while preserving network weights and (for off-policy) replay buffer.

## Training
- `total_timesteps = 4_000_000` (default).
- `CheckpointCallback(save_freq=100_000)` + `VecNormalizeCheckpointCallback(save_freq=100_000)` co-saved under `cell_dir/checkpoints/`, `name_prefix = <algo>_phase2_cell<id>`.
- Final saves: `<algo>_phase2_cell<id>_final.zip`, `vec_normalize.pkl`.
- Output dir: `logs/<algo>/phase2/cell_<id>_we<we>_wsq<wsq>/` (e.g. `cell_04_we0.5_wsq150/`).

## Evaluation
- `evaluate_model(model_path, eval_log_dir=cell_dir, train_config=train_config, algorithm=algo, use_thermal=..., random_target=True)` from `models/eval.py`.
- Writes `evaluation_metrics.json` per cell.

## Provenance JSON (`train_config.json`)
Stored fields:
```
algorithm, phase=2, cell_id,
continued_from = <ckpt_path>,
phase1_best_seed,
phase1_baseline = {rmse_speed_kmph, total_nox_g, delta_soc},
total_timesteps, n_envs,
w_speed, w_emission, w_fuel, w_brake, w_soc, w_soc_squared, w_flicker,
training_duration_seconds, training_duration_hms
```
Lets phase-2 results be diffed directly against phase-1 baseline metrics.

## Cross-cell aggregation (post-hoc)
Per cell → `evaluation_metrics.json` contains `rmse_speed_kmph`, `total_nox_g`, `delta_soc`. Across the 9 cells these define a Pareto surface in (speed RMSE, NOx, ΔSOC), each compared against the single-point phase-1 baseline.
