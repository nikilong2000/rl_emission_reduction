# Phase 1 Cluster Run — Analysis (Speed-Tracking HPO + 10-Seed Validation)

This document analyses the results stored under [logs_cluster/logs/](logs_cluster/logs/) for the three reinforcement-learning algorithms PPO, SAC and TD3, in the context of the master-thesis goal: *learning a speed and emission controller for a hybrid-electric powertrain whose ICE and PG dynamics are surrogated by LSTM models.* Phase 1 deliberately **isolates the speed-tracking sub-problem** with no emission or SOC penalties active in the reward. The findings here therefore answer two questions:

1. *Can each algorithm robustly learn the speed-tracking task in the surrogate-LSTM environment?*
2. *What does the policy do to the emission and energy variables when those variables are not part of the reward?* — i.e. the phase-2 starting baseline.

Throughout, evidence is referenced to the on-disk artefacts so that every claim can be traced back to a file.

---

## 1. Executive Summary

* All three algorithms succeed at the speed-tracking objective: every seed of every algorithm converges to a final episode reward in roughly the 3 050–3 550 range (theoretical max ≈ 3 600 for a 1 800-step Gaussian speed reward), and best-seed RMSE on the staircase evaluation cycle reaches ≈ 3.1 km/h for all three.
* **SAC is the most reliable speed tracker**: lowest mean RMSE (3.45 km/h) and the smallest seed-to-seed variance (σ = 0.43 km/h). PPO is the least reliable (σ = 0.98 km/h, one outlier seed at 6.31 km/h).
* **The emission-side outcomes are unflattering**, *as expected for a phase that does not penalise emissions*. Mean tail-pipe NOx integrals are 48 g (PPO), 64 g (SAC) and 95 g (TD3) per evaluation cycle, with TD3's seed σ approaching ±100 g — these correspond to overall NOx intensities of ~1 200–2 300 mg/km, i.e. **15–30× above the Euro 6 limit of 80 mg/km**. This is the empirical justification for switching on the emission term in Phase 2.
* **SOC ends saturated at 1.0 for the majority of seeds**, regardless of algorithm. The hybrid policies have learned to exploit the absent SOC penalty: keep the engine on at high load, let the EM2 charge the battery, and treat surplus electrical energy as free. A handful of seeds (PPO 5, SAC 8, TD3 6) instead deplete the battery, showing that without a charge-sustaining term the optimisation surface is degenerate in SOC.
* Best HPs found by Optuna are consistent with SB3 community defaults, but with one algorithm-specific surprise: PPO selects an extremely small learning rate (1.28 × 10⁻⁵) with a 256×256 architecture, while TD3 prefers `policy_delay=1` (effectively turning off the delayed-update trick that gives TD3 its name).
* **Compute cost is heterogeneous**: PPO ≈ 8 h/seed, TD3 ≈ 13 h/seed, SAC ≈ 15 h/seed on the cluster — a factor of nearly 2× between the on-policy and the off-policy methods at equal `total_timesteps = 4 M`.

---

## 2. Methodology Recap

The exact protocol is documented in [hyperparameter_search/PHASE1.md](hyperparameter_search/PHASE1.md); the relevant facts for this analysis are:

* **Environment**: [`EmissionControlEnv`](env.py) wrapping pre-trained ICE and PG LSTM surrogates from [internal_lstm_models/](../internal_lstm_models). 12-dim observation, 4-dim continuous action (`engine_on, em2_torque, fuel_mg, brake_perc`), `dt = 0.5 s`.
* **Reward (Phase 1)** — defined in [env.py:562-580](env.py#L562-L580):

  ```
  r = 1.0 · exp(−0.5 · (Δv / 10)²)        Gaussian speed bonus
    − 0.25 · (brake_perc / 100)            small brake penalty
    − 0.25                                 if engine just ignited (anti-flicker)
  ```
  No NOx, fuel or SOC term is active. The maximum per-step reward is 1.0 → episode ceiling ≈ 3 600 over 3 600 steps of 0.5 s.
* **Optuna stage**: TPE sampler with median pruner, 4 M timesteps per trial, RMSE-of-speed-error reported every 500 k steps after a 2 M-step warm-up. Trials are minimised; the best is then frozen as `best_params.json`.
* **Seed validation**: 10 seeds (`SLURM_ARRAY_TASK_ID = 0…9`) re-run with the chosen HPs at `n_envs = 20` for 4 M timesteps. Each seed is evaluated on a deterministic staircase cycle (50 → 70 → 110 → 140 → 80 → 35 km/h, 600 steps × 0.5 s per segment, 1 800 s total) plus a 5-min battery saturation cap at SOC = 1.0.

A **caveat about the emission report**: `calculate_emissions_per_km` in [utils/evaluation_utils.py:39-43](utils/evaluation_utils.py#L39-L43) labels four sub-windows of the evaluation as `low / medium / high / extra_high` using the WLTC class-3 *phase durations* (589 / 433 / 455 / 323 s). The actual evaluation speed schedule, however, is the staircase — *not* WLTC. The duration partition is therefore meaningful (it splits the cycle into similar-length chunks), but the comparison against the **80 mg/km Euro 6 limit is at best indicative**, not regulatory: a real WLTC test would expose the controller to dynamic accel/decel transients absent from the staircase. This limitation is significant for any thesis-level interpretation of "passes Euro 6".

---

## 3. Hyperparameter Optimisation Results

### 3.1 Optuna study sizes and termination states

Read from [logs/<algo>/optuna/all_trials.csv](logs_cluster/logs/) and `study_journal.log`:

| Algorithm | Total trials | Completed | Pruned | Running at snapshot |
|-----------|-------------:|----------:|-------:|--------------------:|
| PPO       | 70           | 17        | 53     | 0                   |
| SAC       | 44           | 20        | 12     | 12                  |
| TD3       | 45           | 30        | 8      | 7                   |

> The "Running" entries for SAC and TD3 indicate that the cluster snapshot in this directory was taken before all parallel SLURM workers had checkpointed their journal writes. They do not represent failed trials but unfinished ones; their hyperparameters are nonetheless visible in `all_trials.csv`. The PPO study, by contrast, was clean.

PPO has by far the highest pruning rate (75.7 %), which is consistent with PPO's well-known sensitivity to learning rate in continuous-control problems: many sampled (lr, n_epochs, n_steps) combinations diverge or stagnate, the median pruner kills them after the 2 M-step warm-up, and the search effectively concentrates on a narrow learning-rate band of ~1–3 × 10⁻⁵.

### 3.2 Best hyperparameters (`best_params.json`)

| Parameter             | PPO                | SAC                          | TD3                              |
|-----------------------|--------------------|------------------------------|----------------------------------|
| `learning_rate`       | 1.28 × 10⁻⁵        | 9.20 × 10⁻⁴                  | 1.79 × 10⁻⁴                      |
| `gamma`               | 0.9647             | 0.9601                       | 0.9647                           |
| `batch_size`          | 64                 | 512                          | 1024                             |
| `net_arch`            | [256, 256]         | [128, 128, 128]              | [128, 128]                       |
| `n_steps` / `train_freq` / `gradient_steps` | 2048 / – / – | – / 1 / 8       | – / 1 / 8                        |
| `n_epochs`            | 11                 | –                            | –                                |
| `gae_lambda` / `clip` | 0.965 / 0.203      | –                            | –                                |
| `tau`                 | –                  | 0.0080                       | 0.0082                           |
| `learning_starts`     | –                  | 1 000                        | 20 000                           |
| `ent_coef` / `use_sde`| – / –              | "auto" / False               | – / –                            |
| `policy_delay`        | –                  | –                            | **1**                            |
| `target_policy_noise` | –                  | –                            | 0.116                            |
| `target_noise_clip`   | –                  | –                            | 0.699                            |
| `action_noise_sigma`  | –                  | –                            | 0.144                            |

Three observations worth keeping in the thesis discussion:

1. **PPO converges on an unusually small `learning_rate`** (≈ 1.3 × 10⁻⁵, near the lower bound of the search range 10⁻⁵). Combined with `n_epochs = 11` over 2 048 environment steps × 20 envs and a 256×256 net, the policy moves slowly but stably — exactly the regime in which PPO's clipped trust region matters least. This is consistent with the smooth, monotonic learning curve in `seeds/training_progress.png` (PPO needs ~3 M steps to plateau; the others plateau in ~0.3 M).
2. **TD3's `policy_delay = 1`** means the actor is updated at every critic update, dropping the delayed-policy update that distinguishes TD3 from DDPG. The clipped Gaussian target-policy smoothing (`target_policy_noise = 0.116`, `clip = 0.7`) and the twin critics survive — but functionally this is closer to DDPG-with-target-noise than canonical TD3. This finding is interesting in its own right: in this LSTM-surrogate environment, the delayed-update regularisation is not what stabilises learning.
3. **All three algorithms select γ ≈ 0.96**. With `dt = 0.5 s` and 3 600-step episodes, γ = 0.96 corresponds to an effective horizon of ~25 steps = 12.5 s. This is short relative to the 300 s segment length of the speed schedule but consistent with the locality of the speed-error reward signal.

### 3.3 Best objective values (Optuna RMSE)

The `best_params` correspond to the following minimum-RMSE trials (column `value` in `all_trials.csv`):

| Algorithm | Best trial | RMSE @ HPO eval (km/h) |
|-----------|-----------:|-----------------------:|
| PPO       | trial_060  | 3.057                  |
| SAC       | trial_039  | 3.083                  |
| TD3       | trial_026  | 3.209                  |

These three numbers are within ≈ 5 % of each other. From the HPO standpoint **the algorithms are essentially tied on speed-tracking quality.** Differences between them only emerge once one inspects seed-level reproducibility and the unconstrained energy/emission behaviour, which is the subject of §4.

---

## 4. Seed-Level Validation Results

The following statistics are computed from the ten `evaluation_metrics.json` files per algorithm. The values match the bar charts in `final_metrics_bar.png` for each algorithm.

| Metric                        | PPO (mean ± σ)   | SAC (mean ± σ)   | TD3 (mean ± σ)   |
|-------------------------------|------------------|------------------|------------------|
| Total reward                  | 3 428 ± 142      | 3 503 ± 57       | 3 486 ± 44       |
| RMSE speed (km/h)             | 4.19 ± 0.98      | **3.45 ± 0.43**  | 3.81 ± 0.29      |
| MAE speed (km/h)              | 2.19 ± 1.24      | **1.54 ± 0.75**  | 1.90 ± 0.58      |
| Total NOx (g)                 | 48.3 ± 29.0      | 63.8 ± 59.8      | 95.4 ± 100.8     |
| Total fuel (g)                | **79.6 ± 27.0**  | 113.5 ± 38.5     | 122.6 ± 56.9     |
| ΔSOC                          | +0.247 ± 0.157   | +0.191 ± 0.280   | +0.146 ± 0.260   |
| Best-seed RMSE                | 3.16             | 3.11             | 3.51             |
| Worst-seed RMSE               | 6.31             | 4.58             | 4.41             |

Bold values mark the best of the three algorithms per row.

### 4.1 Speed-tracking robustness

The full per-seed speed-tracking traces are in [logs_cluster/logs/<algo>/optuna/seeds/eval_rmse_tracking.png](logs_cluster/logs/). Three qualitative observations:

* **Steady-state error ≈ 1 km/h on all flat segments**, dominated by transient overshoot at every step change. The bottom panel of every plot shows RMSE spikes of 30–50 km/h immediately after each setpoint discontinuity and then a return to ≈ 1 km/h. This pattern is identical across the three algorithms, indicating that the residual RMSE is *cycle-shaped* rather than algorithm-shaped: roughly 5–10 % of the 3 600-step episode is dominated by transient error, and that fraction sets the floor RMSE of ≈ 3 km/h that all three algorithms hit.
* **PPO has visibly higher seed-to-seed dispersion in the steady-state plateaux** (the grey 1-σ band on its plot widens to ~5 km/h between t = 350 s and t = 650 s); SAC and TD3 are tighter. Numerically this matches PPO's σ_RMSE = 0.98 km/h vs SAC's 0.43 and TD3's 0.29.
* **Seed 8 of PPO is a clear outlier** (RMSE = 6.31 km/h, MAE = 5.30 km/h, fuel 135 g, NOx 110 g, reward 3 050). Its training did *not* fail — it reached a high reward — but the resulting policy uses a different operating-point mix than the other nine. This is precisely the kind of variability that motivates `select_best_seed.py` filtering before phase 2.

### 4.2 Episode-reward learning curves

[`training_progress.png`](logs_cluster/logs/) shows the episode-reward learning curves averaged over the 10 seeds:

* **SAC**: reaches ≈ 3 500 within 0.3 M steps and remains there with two minor dips (≈ 3.2 M steps). This is essentially the SAC behaviour expected from an off-policy method with a high replay-update ratio (`gradient_steps = 8`, `train_freq = 1`).
* **TD3**: a fast initial climb to ≈ 3 000 by 0.3 M steps, then a second discrete jump to ≈ 3 350 around 1.5 M steps (after which it plateaus). The two-stage curve suggests the policy first finds a coarse high-reward solution and then refines it once the replay buffer accumulates enough diverse transitions.
* **PPO**: a smooth sigmoid that does not finish climbing until ≈ 3 M steps. With its tiny learning rate this is unsurprising; it demonstrates that, on this task, on-policy PPO "uses" the full 4 M-step budget while the off-policy methods waste roughly two thirds of it.

This has a direct compute implication: if phase 2 keeps the same 4 M-step budget but only PPO needs that budget, **the SAC and TD3 wall-clock cost can be cut substantially** without losing speed-tracking quality. Per-seed average training durations from `train_config.json`:

| Algorithm | Mean wall-clock per seed | Range across 10 seeds |
|-----------|--------------------------|-----------------------|
| PPO       | 8.05 h                   | 6.84–10.75 h          |
| TD3       | 13.03 h                  | 10.97–14.97 h         |
| SAC       | 15.28 h                  | 10.64–19.37 h         |

PPO is roughly half the cost of SAC despite needing the full 4 M steps to converge — an artefact of vectorised on-policy rollouts (20 envs collecting in parallel) being wall-clock-cheaper than the off-policy gradient updates with `gradient_steps = 8`.

### 4.3 NOx behaviour with W_EMISSION = 0

Per-seed NOx-per-km figures, computed from `emissions_per_km.txt`:

| Algorithm | NOx (mg/km) min | median | mean | max | seeds passing 80 mg/km overall |
|-----------|----------------:|-------:|-----:|----:|-------------------------------:|
| PPO       | 137             | 1 175  | 1 165 | 2 570 | 0 / 10                         |
| SAC       |  63             | 1 281  | 1 564 | 4 974 | 1 / 10                         |
| TD3       |  97             | 1 655  | 2 327 | 8 625 | 0 / 10                         |

Only one seed across the entire 30-seed cohort (SAC seed 8) is below the 80 mg/km Euro 6 threshold over the whole cycle, and that seed is also the most extreme SOC depleter (ΔSOC = −0.64). The pattern holds even in the cleaner extra-high phase, where 5/10 PPO seeds, 2/10 SAC seeds and 2/10 TD3 seeds pass 80 mg/km — i.e. the high-speed cruise is the only sub-window where the unconstrained-NOx policy occasionally satisfies the limit, and only in PPO with any consistency. *The worst seed (TD3 seed 9) emits 188 g of NOx total — almost 5 kg/km equivalent.*

The interpretation is straightforward: the speed reward is a Gaussian with σ = 10 km/h, so any operating point that keeps the actual speed within ~5 km/h of target collects ≥ 0.88 of the per-step reward. The policy is therefore free to choose any combination of (engine_speed, em2_torque, fuel) that delivers the demanded wheel torque, and Optuna is silent about which of those combinations to pick. Without an explicit NOx penalty, the policies converge on **high-RPM, high-fuel, ICE-dominated operation** because that combination has the best instantaneous controllability of wheel torque (visible in §4.5).

In the project context this is *not a failure of phase 1* — phase 1 deliberately disables `W_EMISSION` to first ensure the speed sub-task is solvable at all. The numbers above precisely quantify the **NOx ceiling that phase 2 must lower**, and they make the optimisation gradient very large: any non-zero `W_EMISSION` should produce visible improvement.

### 4.4 SOC behaviour with W_SOC = 0

Two distinct populations appear in the ΔSOC distributions:

* **Saturation cluster** (the majority): ΔSOC ≈ +0.30, final SOC = 0.9999. The environment caps SOC at 1.0; reaching this ceiling means the policy treats the battery as a free energy sink.
* **Depletion cluster** (rare):
  * PPO seed 5: ΔSOC = −0.22, final SOC = 0.48 — and notably this seed has the lowest fuel of any PPO run (19.3 g) and a low NOx (7.0 g).
  * SAC seed 8: ΔSOC = −0.64, final SOC = 0.06 — extreme depletion, lowest SAC fuel (30.9 g) and NOx (2.6 g) by a wide margin, but RMSE 3.76 km/h and a noticeable speed-tracking penalty.
  * TD3 seeds 0, 6, 8: −0.12 / −0.49 / −0.03 — also low NOx, low fuel.

**These cases reveal a Pareto-style trade-off that the phase-1 reward fails to express.** Within a single algorithm the "depletion" seed is significantly cleaner (factor of 5–20× less NOx) than the "saturation" seeds; it gets there by depleting the battery and using the resulting electrical energy to substitute for ICE torque. The reward function, with W_SOC = 0, treats both endpoints as equally good. This is precisely what the phase-2 `W_SOC_SQUARED ∈ {50, 150, 400}` grid is designed to fix.

A *thesis-level* observation here: **the seed selection in `select_best_seed.py` (sort by RMSE then NOx) will not, by default, select a charge-sustaining seed.** For PPO, ranking by RMSE alone gives seed 3 (RMSE 3.16, ΔSOC = +0.30). The thesis should either (a) document this seed-selection criterion explicitly, or (b) add a charge-sustaining filter before launching phase 2 from the chosen checkpoint, since otherwise phase 2 starts from a battery-saturating prior.

### 4.5 Action policy fingerprints

Per-seed action histograms are saved as `action_distribution.png`. Comparing seed 0 across the three algorithms reveals very different operating-point preferences:

* **PPO seed 0**: ICE almost always on, `ice_speed_rpm` tightly concentrated at ≈ 4 000 rpm (single mode), `em2_torque` confined to a narrow band around ≈ 0–60 Nm, `fuel` concentrated at 17–28 mg, brake ≈ 0. This is a "high-RPM cruise" policy with the EM2 acting mostly as a generator at low torque magnitude.
* **SAC seed 0**: ICE always on, `ice_speed_rpm` at ≈ 3 100 rpm (also a single mode but lower than PPO's), `em2_torque` very tight near 0, `fuel` multimodal between 30 and 50 mg with a tail to 70 mg. The SAC policy keeps engine speed lower and *modulates fuel* to track speed — an approach that uses more fuel per cycle (113 g vs 80 g for PPO) but less time at high RPM.
* **TD3 seed 0**: ICE always on, `ice_speed_rpm` bimodal (≈ 2 200 and ≈ 3 700 rpm), `em2_torque` strongly bimodal at the saturation values (−400 and +400 Nm) — meaning TD3 is using the EM2 in a "bang-bang" pattern, alternating between full motoring and full regen. `fuel` is correspondingly multimodal at 5, 13, 20, 28, 38 mg; brake ≈ 0. This explains both the lowest fuel of any TD3 seed evaluated (62 g) *and* the most extreme NOx variance: full-throttle accelerations followed by full regen creates exactly the load transients that NOx production scales with.

This action-fingerprint difference is independently visible in the per-seed `state_action_occupancy.png` heatmaps and is a rich object for thesis discussion: **all three algorithms learn the same speed-tracking objective but arrive at qualitatively different operating-point distributions**, none of which optimise emissions.

---

## 5. Cross-Algorithm Discussion

### 5.1 Why does SAC win on speed RMSE?

Three contributing factors line up:

1. **Off-policy data efficiency.** With `gradient_steps = 8` per environment step and `batch_size = 512`, SAC takes ≈ 8× more gradient steps than TD3 per env transition (TD3 also has `gradient_steps = 8` but `policy_delay = 1` is a coincidence here, not a multiplier). This translates into the early-flat learning curve.
2. **Maximum-entropy exploration.** With `ent_coef = "auto"` SAC adapts the temperature; this is particularly helpful in a 4-D action space where the brake and EM2 torque components have very different scales. The result is uniformly good seed-to-seed performance (σ_RMSE = 0.43).
3. **Larger and deeper network.** SAC's chosen `[128,128,128]` is the only three-layer architecture among the three winners; in a 12-dim observation space this extra depth is enough to fit the LSTM-surrogate dynamics nonlinearity.

### 5.2 Why does PPO sometimes fail (seed 8)?

PPO's tiny learning rate makes it sensitive to early exploration luck: if the policy happens to seed an operating point where the gradient is small, it can take millions of steps to escape. The fact that PPO needs the entire 4 M-step budget to plateau (vs SAC's 0.3 M) means that 25 % of PPO's training time is spent near the plateau — but for unlucky seeds the *plateau itself* is mediocre. This is a textbook PPO failure mode and is the main argument against PPO in continuous-control problems with replay buffers available.

### 5.3 Why does TD3 produce the most extreme NOx variance?

TD3 with `policy_delay = 1` and twin critics is essentially DDPG-with-target-smoothing. The twin-critic minimum-target operator is a *pessimism* mechanism that, in our environment, biases the actor toward conservative torque trajectories — but conservative for the critic is not conservative for emissions. Combined with the deterministic actor and the bimodal EM2 action histogram, TD3 produces "all or nothing" battery behaviour: some seeds settle into a regen-heavy operating mode (low NOx), others into a motoring-heavy mode (high NOx). The result is the largest seed σ_NOx (≈ 100 g) of the three.

### 5.4 Pareto picture (qualitative)

If one plots speed RMSE vs total NOx across the 30 seeds, two regions emerge:

* **The "fuel-burner" cluster**: speed RMSE 3–4 km/h, NOx 30–110 g. All three algorithms populate this cluster densely. It is the regime the phase-1 reward implicitly selects.
* **The "battery dumper" cluster**: speed RMSE 3.5–6 km/h, NOx 3–10 g. PPO seed 5, SAC seeds 6 and 8, TD3 seeds 0 and 6. These are the natural anchors for a low-NOx phase-2 starting point — but they are not what `select_best_seed.py` returns by default.

The phase-2 grid (`W_EMISSION ∈ {0.25, 0.5, 1.0}, W_SOC_SQUARED ∈ {50, 150, 400}`) is specifically designed to convert the fuel-burner cluster into a controlled battery-aware policy. The phase-1 numbers in this report establish the **single point** from which that 3×3 sweep starts.

---

## 6. Implications for the Thesis Goal

The overarching project goal is *speed control + emission reduction* in an LSTM-surrogated hybrid powertrain. Phase 1 evidence informs that goal in five concrete ways:

1. **Surrogate environment is RL-tractable.** All three algorithms reach > 90 % of the maximum speed-tracking reward. The LSTM ICE/PG models are differentiable enough through the SB3 stack (via VecNormalize and SubprocVecEnv) to support stable continuous-control training.
2. **Best-case speed RMSE ≈ 3 km/h** sets the "speed-tracking floor" for any subsequent reward shape. Phase 2 cannot drop below this without changing the cycle or the action space.
3. **Unconstrained NOx is 15–30× the Euro 6 limit.** This is the headline number for the introduction/motivation: it shows that *"learning to drive accurately"* is **not** the same as *"learning to drive cleanly"* — the central thesis of the work.
4. **SOC ceiling exploitation reveals an environment-design issue.** The hybrid powertrain reward must include a charge-sustaining term, otherwise the optimisation problem is degenerate. Phase 2's `W_SOC_SQUARED` term is the right fix.
5. **Algorithm choice is decoupled from speed-tracking quality.** Since all three algorithms reach equivalent speed performance, the algorithm decision should be based on (a) seed-to-seed reproducibility (SAC wins), (b) compute cost (PPO wins), and (c) likely behaviour under emission penalties (TD3's bimodal action policy may transfer best to a low-NOx regime, but is the riskiest in variance — to be tested in phase 2).

A defensible thesis recommendation is therefore: **carry SAC and TD3 forward into phase 2, drop PPO**. SAC because it minimises RMSE variance and provides the most "reliable" baseline; TD3 because its EM2 bang-bang behaviour is structurally close to a charge-sustaining policy and could benefit most from explicit NOx shaping. PPO costs least but its outlier seed (PPO 8) is a reproducibility risk under reward changes.

---

## 7. Limitations and Threats to Validity

1. **Evaluation cycle is not WLTC.** The "WLTC phases" labels in `emissions_per_km.txt` are only durational windows over the staircase cycle. Any thesis statement of the form "this controller passes Euro 6 on WLTC" cannot be supported by these results; only "passes 80 mg/km on the staircase evaluation" can. A separate WLTC evaluation with a real WLTC speed trace should be added before claims about real-world emission compliance are made.
2. **Single deterministic evaluation episode per seed.** With one episode the reported RMSE/NOx are point estimates. Repeating each seed's evaluation over, e.g., 10 stochastic eval episodes (the eval env supports `random_target=True`) would let the thesis report confidence intervals on the per-seed metrics rather than on the seed mean alone.
3. **Optuna study sizes differ.** PPO had 70 trials, SAC 44, TD3 45. The ranking among the three best HPs is therefore not strictly comparable — PPO had ~50 % more chances to find a good configuration. A leveled comparison would re-run all three at the same trial budget.
4. **Action space includes a real-valued `engine_on` command.** Looking at the action distribution, every algorithm has saturated `engine_on` to 1.0, i.e. the engine is always on. The flicker penalty (W_FLICKER = 0.25) is therefore never paid in any seed. This means the engine on/off discrete choice has been effectively bypassed in phase 1 — to be revisited in phase 2 or in a future phase that adds a stop/start incentive.
5. **The pruner discards roughly half of the PPO and SAC search.** With 53/70 PPO trials and 12/44 SAC trials pruned (and 12 still running at snapshot time), the trial-level statistics in `all_trials.csv` are *biased toward configurations Optuna already considered promising*. This is the expected behaviour of a median pruner but should be acknowledged when reporting "median trial RMSE" to avoid implying the full search distribution.
6. **VecNormalize stats are reset at evaluation.** `vec_normalize.pkl` is loaded back, but reward normalisation is intentionally turned off. This is correct for SAC and TD3 (which do not normalise rewards) and is documented in PHASE2.md, but a sentence acknowledging it should be in the thesis "Evaluation protocol" subsection so readers do not assume that the reported reward numbers are normalised.

---

## 8. Recommendations for Phase 2 and Beyond

* **Best-seed selection rule.** Replace the current `(rmse_speed_kmph, total_nox_g)` sort with `(rmse_speed_kmph, |delta_soc|, total_nox_g)` so that battery-saturating and battery-depleting seeds are de-prioritised before phase 2 starts from a degenerate-SOC checkpoint.
* **Add a stochastic eval pass.** Run each best seed across 10 random-target episodes in addition to the deterministic staircase. This gives the thesis a secondary metric (mean RMSE on random cycles) that is closer to the training distribution than the staircase.
* **Add a real WLTC evaluation cycle.** The data needed is the WLTC class-3 speed schedule as a CSV under [data_train/](data_train/); using the existing `dataset_path` argument in `EmissionControlEnv` it can be evaluated alongside the staircase. This will make the Euro 6 comparison meaningful.
* **Drop PPO from phase 2 unless its compute advantage is needed.** Equal-quality speed tracking, larger seed variance and slow convergence make it the weakest candidate for the phase-2 reward sweep. If thesis space allows, a single PPO cell can be kept for completeness.
* **Sanity-check the `engine_on` action.** Because all seeds saturate engine_on = 1, the discrete engine-toggle dimension is currently inert. Phase 2 (or a new phase) can probe whether an additional fuel-saving stop/start behaviour exists, by adding an `idle_penalty` term or by changing the action representation to a discrete on/off plus continuous fuel command.
* **Report 95 % CIs, not just σ.** With n = 10 seeds, σ is noisy. A bootstrap CI on the per-seed mean RMSE is a small change to `plot_seeds.py` and is worth making for the thesis figures.

---

## 9. Reproducibility Pointers

* Optuna search spaces: [hyperparameter_search/hpo_search_spaces.py](hyperparameter_search/hpo_search_spaces.py)
* Per-trial training: [hyperparameter_search/tune_hpo.py](hyperparameter_search/tune_hpo.py) launched by [hyperparameter_search/submit_hpo.sh](hyperparameter_search/submit_hpo.sh)
* Seed runs: [hyperparameter_search/run_seeds.py](hyperparameter_search/run_seeds.py) launched by [hyperparameter_search/submit_seeds.sh](hyperparameter_search/submit_seeds.sh)
* Best-seed selection: [hyperparameter_search/select_best_seed.py](hyperparameter_search/select_best_seed.py)
* Plot scripts: [hyperparameter_search/plot_seeds.py](hyperparameter_search/plot_seeds.py)
* Reward + observation + action: [env.py:200-260, 540-660](env.py#L200-L260)
* Per-seed evaluation pipeline: [utils/evaluation_utils.py](utils/evaluation_utils.py) and `models/eval.py`

The full result set required to reproduce this analysis is the set of `evaluation_metrics.json`, `emissions_per_km.txt` and `train_config.json` files under [logs_cluster/logs/<algo>/optuna/seeds/seed_<n>/](logs_cluster/logs/) plus the three `best_params.json` and `all_trials.csv` files at the algorithm root.
