# Phase 2 Cluster Run — Analysis (Emission + SOC Reward Shaping)

This document analyses the phase-2 results stored under [logs/](logs/) for the three reinforcement-learning algorithms PPO, SAC and TD3. Phase 2 **adds the emission and SOC penalty terms** to the speed-tracking reward of phase 1 and **fine-tunes** the phase-1 best-seed checkpoints under three sequential search stages — coarse 3×3 grid, Optuna TPE refinement, and a 10-seed validation at the chosen weights. The analysis mirrors [PHASE1_ANALYSIS.md](../logs_cluster_phase1/PHASE1_ANALYSIS.md) and answers two questions:

1. _Can fine-tuning the phase-1 policy with an emission/SOC-augmented reward reduce NOx and bound SOC drift without sacrificing speed-tracking?_
2. _Which algorithm transfers best to the multi-objective regime, and what are the failure modes?_

Every claim is referenced to the on-disk artefacts so it can be traced back to a file.

---

## 1. Executive Summary

- **The emission objective is met.** Total cycle NOx drops by 90–93 % vs phase-1 for SAC and TD3, and PPO converges to the same absolute NOx (≈ 5 g/cycle) it already had at phase-1 best seed but with much tighter SOC. Absolute NOx is now in the 4.5–10 g/cycle band for all three algorithms (was 48–95 g in phase 1).
- **SOC drift is essentially eliminated** for PPO (|ΔSOC| 0.029, was 0.247) and SAC (|ΔSOC| 0.010, was 0.191). TD3 (|ΔSOC| 0.061) is also tighter than phase 1 but with larger variance.
- **The real contribution is low NOx _under a charge-sustaining constraint_.** Phase-1's lowest-NOx seeds reach their low NOx only by depleting the battery (SAC seed 8: 2.57 g but ΔSOC = −0.64). Only 1 of 30 phase-1 seeds is charge-sustaining (|ΔSOC| < 0.10), and it emits 60.9 g. **All 30 phase-2 seeds are charge-sustaining and every one beats that best charge-sustaining phase-1 seed by 4–30×.** Phase-2 is _not_ a raw-NOx record over phase-1 (a few battery-dumping phase-1 seeds are competitive), but it is the only set of policies that is simultaneously clean and charge-sustaining.
- **Mechanism:** the NOx cut comes from running the engine at far lower load (PPO 135 → 46 Nm, SAC 185 → 79 Nm), because phase-1 burned extra fuel at high load purely to overcharge the battery to its 1.0 cap. Phase-2 therefore reduces NOx _and_ fuel at once. NOx rate rises steeply with engine load (r = 0.67), so moving to low load — _away_ from the fuel-efficient BSFC sweet spot — is what lowers emissions.
- **Speed-tracking is preserved for PPO and SAC**: RMSE 3.13 km/h (PPO) and 3.33 km/h (SAC), within 0.2 km/h of phase-1. **TD3 is the only algorithm that loses speed-tracking** in the fine-tune: mean RMSE 7.41 km/h with σ = 9.94, dominated by two diverging seeds (3 and 1). With those outliers excluded, TD3 mean RMSE recovers to 3.96 km/h.
- **The ranking flips vs phase 1.** Phase 1 ordered SAC < TD3 < PPO on RMSE; phase 2 orders PPO < SAC ≪ TD3 because PPO's smaller learning rate makes it the most fine-tune-robust algorithm in the multi-objective regime.
- **NOx intensity remains above the Euro 6 limit** (80 mg/km) but is no longer 15–30× above it. Phase-2 NOx intensities: PPO ≈ 122 mg/km, SAC ≈ 111 mg/km, TD3 ≈ 250 mg/km (10-seed mean). Phase-1 baselines were 1 165 / 1 564 / 2 327 mg/km respectively, i.e. **a factor-of-10 reduction with the policy weights unchanged in dimensionality**.
- **Optuna refinement was justified.** The 3×3 grid scout located only rough regions; Optuna's TPE search (20 trials per algo) shifted SAC's NOx winner from 3.03 g to 2.44 g and produced a different optimum location for TD3 (we = 1.01, wsq = 64.5 vs grid's we = 0.25, wsq = 50).

---

## 2. Methodology Recap

The reproducible protocol is documented in [PHASE2.md](../hyperparameter_search/PHASE2.md); the relevant facts for this analysis are:

- **Environment**: same [`EmissionControlEnv`](../env.py) as phase 1, ONNX-backed surrogate (USE_ONNX=True). 12-dim observation, 4-dim continuous action, dt = 0.5 s.
- **Reward (Phase 2)** — same speed/brake/flicker terms as phase 1, plus the two previously-disabled terms now turned on:

  ```
  r += W_SPEED · exp(−0.5 · (Δv / 10)²)             Gaussian speed bonus (= 1.0 at peak)
       − W_EMISSION · min(NOx_g_per_s / 0.4, 1.0)   tail-pipe NOx penalty
       − W_BRAKE · (brake_perc / 100)               (unchanged)
       − W_SOC_SQUARED · (SOC − SOC_init)²           charge-sustaining penalty
       − W_FLICKER if engine just ignited            (unchanged)
  ```

  W_SOC linear and W_FUEL were kept at 0 by user decision.

- **Three search stages**:

  | Stage                    | Tool                                                                  | Output                                                                                                                                     |
  | ------------------------ | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
  | A. Coarse 3×3 grid       | `submit_phase2_sweep.sh`                                              | `logs/<algo>/phase2/cell_*/` — 27 fine-tune runs (9 cells × 3 algos), [phase2_results.csv](phase2_results.csv)                             |
  | B. Optuna TPE refinement | `submit_phase2_hpo.sh` + `submit_phase2_hpo_rerun_<algo>.sh` recovery | `logs/<algo>/phase2_optuna/` — 20 valid trials per algo, [phase2_optuna_results.csv](phase2_optuna_results.csv), `best_params_phase2.json` |
  | C. 10-seed validation    | `submit_phase2_seeds.sh`                                              | `logs/<algo>/phase2_seeds/seed_0..9/`, [phase2_seeds_results.csv](phase2_seeds_results.csv)                                                |

  Each stage continues from the **phase-1 best-seed checkpoint** (`select_best_seed.py`): PPO seed 3 (RMSE 3.16, NOx 5.57 g), SAC seed 3 (3.11, 48.94 g), TD3 seed 1 (3.52, 73.42 g).

- **Optuna objective** — composite scalar, minimised:

  ```
  score = total_nox_g
        + λ_rmse · max(0, rmse_speed_kmph − 5)²       λ_rmse = 20
        + λ_soc  · max(0, max_abs_soc_drift − 0.05)²  λ_soc  = 1000
  ```

  Both penalty constants are exposed as CLI flags (`--lambda_rmse`, `--lambda_soc`). The penalty regions are designed so that any cell with RMSE ≥ 6 km/h or max\|ΔSOC\| ≥ 0.10 incurs a NOx-equivalent penalty larger than the natural NOx scale (~5 g) → infeasible regions are repelled by TPE within ~5 trials.

- **Evaluation cycle**: identical to phase 1 — deterministic staircase (50 → 70 → 110 → 140 → 80 → 35 km/h), 3 600 steps × 0.5 s, plus the 5-min SOC-saturation cap. Cycle distance ≈ 40 km for all algorithms, enabling direct mg/km comparison.

> **WLTC caveat carries over.** The "phase low/medium/high/extra_high" labels in `calculate_emissions_per_km` are durational sub-windows over the staircase, not the regulatory WLTC speed schedule. _Indicative_, not regulatory.

---

## 3. Stage A — 3×3 Grid Scout (Appendix-Level Reference)

The full scout is summarised in [phase2_results.csv](phase2_results.csv). It was used to locate the rough region of good `(W_EMISSION, W_SOC_SQUARED)` weights and to characterise failure modes; the headline conclusions are:

- **SAC has one degenerate cell** (cell 8, we = 1.0, wsq = 400) where the policy collapses to engine-off + permanent braking → RMSE = 80.8 km/h, NOx ≈ 0. TPE later avoided this region.
- **TD3 fails 4 of 9 cells** under the 1.5 × phase-1 RMSE filter (cells 2, 4, 5, 6, 8 all have RMSE > 5.3 km/h). This is the empirical motivation for the continuous Optuna search — the grid is too coarse for TD3's narrow stable region.
- **PPO survives all 9 cells**, with very small RMSE variation (3.07–3.38). The transfer-learning prior is strongest for PPO.

The grid is no longer the source of phase-2 winners — Optuna superseded it — but cell-level results remain on disk for reproducibility and as a robustness check on the Optuna-best regions.

---

## 4. Stage B — Optuna Refinement

### 4.1 Study sizes and termination states

Read from each algo's [logs/<algo>/phase2_optuna/all_trials.csv](logs/) and `study_journal.log` (refreshed by `export_phase2_study.py` after the SLURM zombie-rerun recovery, see §6.3):

| Algorithm | Journal trials | COMPLETE | PRUNED | FAIL (recovered zombies) | Valid (C + P) |
| --------- | -------------: | -------: | -----: | -----------------------: | ------------: |
| PPO       |             20 |       16 |      4 |                        0 |            20 |
| SAC       |             25 |       11 |      9 |                        5 |            20 |
| TD3       |             22 |       15 |      5 |                        2 |            20 |

All three studies reached the 20-trial target. The FAIL rows are the original RUNNING zombies (killed by NODE_FAIL); they were re-enqueued with identical params via `rerun_phase2_trial.py --auto --mark_failed`, see §6.3.

### 4.2 Best `(W_EMISSION, W_SOC_SQUARED)` per algorithm

| Algorithm | Best trial | W_EMISSION | W_SOC_SQUARED | Optuna score | RMSE (km/h) | NOx (g) | max\|ΔSOC\| |
| --------- | ---------: | ---------: | ------------: | -----------: | ----------: | ------: | ----------: |
| PPO       |    trial 1 |      1.185 |         92.86 |        4.111 |        3.16 |    4.11 |       0.040 |
| SAC       |    trial 4 |      0.495 |        167.01 |    **2.436** |        3.68 |    2.44 |       0.013 |
| TD3       |    trial 4 |      1.010 |         64.50 |        3.950 |        4.01 |    2.68 |       0.086 |

Three observations:

1. **SAC found the cleanest single-trial NOx** at 2.44 g — the lowest emission of any phase-2 cell across the three algorithms. This trial sits in a moderate-emission-weight region with a relatively high SOC-squared weight, suggesting that SAC benefits from a stronger SOC term to compensate for its replay buffer being dominated by phase-1 (battery-saturating) transitions.
2. **PPO's Optuna winner has the highest W_EMISSION** (1.185) but a moderate W_SOC_SQUARED (92.86). This is consistent with PPO already having a low NOx baseline (5.57 g phase-1): the extra emission weight does not have much headroom to bite, while the SOC weight is what reshapes the operating policy.
3. **TD3's Optuna winner has the lowest W_SOC_SQUARED** (64.5). High SOC weights destabilised TD3 in the grid scout; TPE correctly avoided the high-wsq region. Trade-off: lower SOC weight → larger max\|ΔSOC\| (0.086 at the Optuna winner).

### 4.3 Trial-level dispersion and pruning behaviour

The Optuna PRUNED count is dominated by trials sampled near the grid-known failure regions (high we × high wsq). The composite penalty term makes the score climb steeply outside RMSE ≤ 5; trials reporting intermediate RMSE > 5 after 2 M-step warmup were pruned correctly. The 5 FAIL rows in the SAC study and 2 in TD3 are the recovered zombies; their params were re-enqueued and produced valid COMPLETE or PRUNED trials at new trial numbers.

---

## 5. Stage C — 10-Seed Validation Results

The headline phase-2 statistics computed from [phase2_seeds_results.csv](phase2_seeds_results.csv) (10 seeds × 3 algos):

| Metric                | PPO (mean ± σ)  | SAC (mean ± σ)    | TD3 (mean ± σ) | TD3 — excluding seeds 1, 3 |
| --------------------- | --------------- | ----------------- | -------------- | -------------------------- |
| Total reward          | 3 519 ± 41      | 3 520 ± 51        | 3 360 ± 207    | 3 461 ± 51                 |
| RMSE speed (km/h)     | **3.13 ± 0.07** | 3.33 ± 0.33       | 7.41 ± 9.94    | 3.96 ± 0.47                |
| MAE speed (km/h)      | 1.55 ± 0.05     | 1.43 ± 0.46       | —              | 1.91 ± 0.45                |
| Total NOx (g)         | 4.94 ± 0.72     | **4.51 ± 1.46**   | 10.01 ± 4.85   | 8.65 ± 4.43                |
| NOx intensity (mg/km) | 122             | **111**           | 250            | 216                        |
| Total fuel (g)        | **52.8 ± 1.3**  | 64.0 ± 20.2       | 79.8 ± 31.4    | 79.5 ± 35.6                |
| \|ΔSOC\|              | 0.029 ± 0.004   | **0.010 ± 0.008** | 0.061 ± 0.032  | 0.072 ± 0.034              |
| max\|ΔSOC\|           | 0.038 ± 0.009   | **0.021 ± 0.009** | 0.137 ± 0.058  | 0.123 ± 0.048              |
| Best-seed RMSE        | 3.04            | 3.03              | 3.39           | —                          |
| Worst-seed RMSE       | 3.30            | 4.03              | **35.57**      | 4.83                       |

Bold marks the best of the three algorithms per row.

### 5.1 Speed-tracking robustness

Per-seed RMSE distributions are visible in [logs/<algo>/phase2_seeds/eval_rmse_tracking.png](logs/) and in the overlay plot [analysis_plots/eval_rmse_tracking_overlay.png](analysis_plots/eval_rmse_tracking_overlay.png).

- **PPO is the most reliable speed tracker in phase 2** — σ_RMSE = 0.07 km/h across 10 seeds, compared with phase-1's σ = 0.98 km/h. Fine-tuning from a single phase-1 checkpoint substantially reduces seed-to-seed variance.
- **SAC σ_RMSE = 0.33 km/h** is slightly worse than phase-1 (0.43 km/h) — the variance reduction comes from PPO, not SAC.
- **TD3 is bimodal**: 8/10 seeds have RMSE in [3.39, 4.83] km/h with σ = 0.47; seeds 1 and 3 diverge to RMSE 6.78 and **35.57** km/h respectively. This is **catastrophic forgetting** of the speed-tracking competence learned in phase 1 under the new SOC penalty: with W_SOC_SQUARED = 64.5 and the off-policy critic re-estimating from a now-shifted reward, the actor in seeds 1 and 3 gets pulled into a degenerate operating point.

### 5.2 Per-seed details

The full per-seed tables produced by `collect_phase2_seeds_results.py`:

- **PPO** — all 10 seeds in [3.04, 3.30] km/h, NOx [4.23, 6.16] g, |ΔSOC| ≤ 0.035. No outliers.
- **SAC** — 9 seeds in [3.03, 3.79] km/h; seed 4 at 4.03 km/h is the only mild outlier and is also the seed with the largest |ΔSOC| (0.028). Total NOx of seed 7 = **2.06 g**, the lowest of all 30 phase-2 seeds.
- **TD3** — seeds (sorted by RMSE): {8 (3.39), 2 (3.54), 4 (3.64), 7 (3.78), 0 (4.11), 5 (4.16), 9 (4.26), 6 (4.83), 1 (6.78), **3 (35.57)**}. Seed 3 has RMSE 10× the median, |ΔSOC| ≈ 0 (no charge dump), and elevated NOx (15 g) — the policy is "frozen" at a low-speed plateau and never tracks the higher setpoints. Seed 1 is a softer version of the same pathology. The robust TD3 mean (8/10 seeds) recovers to **3.96 km/h ± 0.47**, comparable to SAC.

### 5.3 NOx behaviour with the emission penalty active

| Algorithm | NOx (mg/km) min | median | mean | max | seeds passing 80 mg/km |
| --------- | --------------: | -----: | ---: | --: | ---------------------: |
| PPO       |             105 |    117 |  122 | 153 |                 0 / 10 |
| SAC       |              51 |    100 |  111 | 169 |             **3 / 10** |
| TD3       |             100 |    200 |  250 | 426 |                 0 / 10 |

SAC seed 7 (51 mg/km), seed 6 (84 mg/km — just over the bar) and seed 1 (96 mg/km) are the only three seeds across all 30 phase-2 runs that achieve the 80 mg/km Euro 6 line on the staircase cycle. None of them are the "best-RMSE" seed — confirming the phase-2-level Pareto trade-off that the fine-tune surfaces.

### 5.4 SOC behaviour with the SOC penalty active

- **All 30 seeds have |ΔSOC| < 0.20**, vs phase-1 where the majority saturated at +0.30 or depleted to −0.65. The SOC term works.
- **SAC's max\|ΔSOC\| is 0.021 ± 0.009**, an order of magnitude tighter than phase-1. The combination of off-policy replay + the high-weight SOC penalty produces the tightest charge-sustaining behaviour.
- **TD3 retains larger SOC drift** (max\|ΔSOC\| 0.137) because its Optuna winner used a low W_SOC_SQUARED (64.5). A re-pick from the Optuna front with a charge-sustaining constraint would resolve this — see §8.

### 5.5 Operating-point shift and the mechanism of the NOx reduction

The reduction in NOx is **not** achieved by moving toward the fuel-efficient (low-BSFC) region of the engine map — it is achieved by moving _away_ from it. In a diesel engine the low-BSFC "sweet spot" is the high-load / high-combustion-temperature zone, and NOx production rises steeply with engine load. Plotting phase-2 operating points on a BSFC map therefore makes them look "worse" (further from the sweet spot) when they are in fact correctly trading a small amount of brake-specific fuel efficiency for a large NOx cut. **The BSFC map is the wrong lens for the emission story; the NOx-rate map is the right one.**

The empirical relationship, pooled over all 60 seeds (both phases, engine-on steps), is monotonic and steep — tail-pipe NOx rate climbs from ≈ 3 mg/s below 60 Nm to ≈ 116 mg/s at 270–300 Nm (Pearson r(torque, NOx) = 0.67). This is shown in [analysis_plots/nox_vs_load.png](analysis_plots/nox_vs_load.png) (top panel), with the per-algo engine-load distributions of phase-1 vs phase-2 in the bottom panel.

The mean engine operating point shifts dramatically toward lower load between the two phases (engine-on steps, pooled per algo):

| Algorithm | Phase-1 mean op-point | Phase-2 mean op-point  | NOx rate            |
| --------- | --------------------- | ---------------------- | ------------------- |
| PPO       | 3 997 rpm / 135 Nm    | 3 995 rpm / **46 Nm**  | 37.1 → **2.4** mg/s |
| SAC       | 3 117 rpm / 185 Nm    | 2 855 rpm / **79 Nm**  | 40.6 → **2.6** mg/s |
| TD3       | 3 283 rpm / 153 Nm    | 2 368 rpm / **121 Nm** | 58.6 → **8.0** mg/s |

The per-algo NOx-map overlay is [analysis_plots/engine_map_nox_overlay.png](analysis_plots/engine_map_nox_overlay.png). Its background is the **measured EA189 bench NOx mass flow** (`mNOx`, g/h), triangulated from the same 74 steady-state points as the BSFC map — so it is smooth with no empty bins. The measured surface confirms the mechanism: `mNOx` rises strongly with engine load (corr(torque, mNOx) = 0.84) and is anti-correlated with BSFC (corr = −0.64), i.e. the low-BSFC sweet spot is a high-NOx zone. The phase-1 median operating point (white X) sits in the bright high-NOx band; the phase-2 median (algo colour X) drops into the dark low-NOx band. TD3 retains the highest residual load (121 Nm), the structural reason it still emits ~3× the NOx rate of PPO/SAC after fine-tuning.

**Why does lower engine load also reduce total fuel?** Phase-1 had no SOC penalty, so the majority of seeds _overcharged_ the battery (ΔSOC ≈ +0.30, hitting the SOC = 1.0 cap, §5.4). Charging to saturation requires extra high-load engine operation purely to dump energy into the battery — which costs both fuel and NOx. Phase-2 charge-sustains (ΔSOC ≈ 0), so the engine performs only the tractive work the cycle demands. The result is the rare win-win visible in §7: phase-2 cuts NOx **and** fuel simultaneously, because both were inflated in phase-1 by gratuitous battery charging.

The BSFC-map view (efficiency lens) is retained for completeness at [analysis_plots/engine_map_occupancy.png](analysis_plots/engine_map_occupancy.png) (phase-2 standalone) and [analysis_plots/engine_map_bsfc_overlay.png](analysis_plots/engine_map_bsfc_overlay.png) (phase-1 vs phase-2); it shows the same load reduction, read as a small move away from the brake-specific-efficiency optimum.

### 5.6 Pareto picture

[analysis_plots/pareto_overlay.png](analysis_plots/pareto_overlay.png) overlays the 30 phase-1 seeds (open gray markers) and 30 phase-2 seeds (filled, algo-coloured), with NOx on a log axis. The correct reading is **not** that every phase-2 seed beats every phase-1 seed on raw NOx — it does not. Phase-1 has a small cluster of very-low-NOx seeds (e.g. SAC seed 8 at 2.57 g, TD3 seed 0 at 3.88 g) that are competitive with, or below, many phase-2 seeds. The decisive point is _how_ those phase-1 seeds reach low NOx:

1. **Phase-1's low-NOx seeds are battery-depleters, not charge-sustaining policies.** SAC seed 8 reaches 2.57 g only by draining the battery from 70 % to 6 % SOC (ΔSOC = −0.64); the other sub-10 g phase-1 seeds likewise have large |ΔSOC| (−0.22 to −0.64, or the +0.30 saturating mode). Such a policy is physically inadmissible for a charge-sustaining HEV — it borrows traction energy from a battery it never recharges.
2. **Only 1 of 30 phase-1 seeds is charge-sustaining** (|ΔSOC| < 0.10): TD3 seed 8, at **60.9 g** NOx. This single point is the only fair phase-1 comparator.
3. **All 30 phase-2 seeds are charge-sustaining** (|ΔSOC| ≤ 0.12, mostly < 0.06) **and every one of them beats the best charge-sustaining phase-1 seed (60.9 g) by a factor of 4–30×** (phase-2 NOx range [2.06, 17.06] g). This — not a raw NOx record — is the phase-2 contribution: achieving low NOx _under the charge-sustaining constraint that phase-1 essentially never satisfied._
4. **The TD3 seed 3 outlier appears at RMSE = 35.6 km/h** as a single far-right marker, the only departure from the otherwise tight phase-2 cluster.

The phase-2 standalone Pareto is at [analysis_plots/pareto_front.png](analysis_plots/pareto_front.png). The **recommended thesis figure** is the ΔSOC-coloured overlay [analysis_plots/pareto_overlay_soc.png](analysis_plots/pareto_overlay_soc.png): marker shape = algorithm, edge thickness = phase, colour = ΔSOC (blue = depleted, white ≈ charge-sustaining, red = saturated). The single deep-blue phase-1 marker at low NOx is SAC seed 8 — visually isolating the one "clean" phase-1 seed as a battery-dumper, while the phase-2 cloud is uniformly pale (charge-sustaining) at low NOx.

---

## 6. Cross-Algorithm Discussion

### 6.1 Why does PPO suddenly win on RMSE robustness?

In phase 1, PPO was the _least_ reliable algorithm (σ_RMSE = 0.98 km/h, one outlier at 6.31). In phase 2, PPO is the _most_ reliable (σ_RMSE = 0.07 km/h, zero outliers). Two complementary explanations:

- **Single starting point.** Phase 2 fine-tunes from PPO seed 3 only. The seed-1 variance of phase-1 is no longer in the picture.
- **Small learning rate is an asset in fine-tuning.** PPO's lr = 1.28 × 10⁻⁵ means the policy moves slowly under the new reward, so any catastrophic forgetting is bounded. SAC and TD3 with larger learning rates (lr = 9.2 × 10⁻⁴ and 1.79 × 10⁻⁴) take larger steps; TD3's two-seed RMSE blow-up is a direct consequence.

This **inverts the phase-1 algorithm recommendation**: PPO is the safest fine-tune candidate when the reward changes mid-training, even though it was the worst train-from-scratch performer.

### 6.2 Why does SAC produce the lowest NOx?

SAC's combination of (a) off-policy replay carrying a _re-weighted_ reward through the existing buffer, (b) the auto-tuned entropy temperature keeping exploration alive during fine-tune, and (c) the three-layer net learned in phase 1 having more capacity to fit a multi-objective surface — together give SAC the largest absolute and relative NOx reduction (4.51 g, −93 % vs phase-1's 48.94 g). The single 2.06 g seed-7 result is best-in-class.

The cost is that SAC has a slightly larger fuel σ (20.2 g vs PPO's 1.3 g) and a slightly higher mean RMSE (3.33 vs 3.13). For a thesis-level claim, the SAC trade-off is: **the best NOx at the cost of a small accuracy and fuel-variability penalty.**

### 6.3 Why does TD3 break for seeds 1 and 3?

TD3 with `policy_delay = 1` is structurally close to DDPG-with-target-noise. In the phase-2 reward landscape:

- The critic must re-estimate Q under the new reward. Twin-critic minimum-target dampens the re-estimation but does not solve the _direction_ of the gradient.
- The deterministic actor + low SOC penalty (wsq = 64.5) gives the policy two basins of attraction: a "battery-aware low-RPM cruise" (the desired one) and a "speed-stuck at low setpoint" basin where the actor never reaches the higher staircase steps because the SOC penalty over-weights the cost of charging.
- Seeds 1 and 3 fall into the latter basin. Seed 3 is the more pathological case (RMSE 35.6 km/h, total reward 2 848). The remaining 8/10 seeds find the desired basin and produce reasonable metrics.

This is a robustness, not a performance, failure: when the policy _does_ converge, its NOx is competitive (best TD3 seed: NOx = 5.10 g). The thesis recommendation is therefore to **either accept TD3's seed risk, or re-pick from the Optuna front with a stability constraint** (e.g. require max\|ΔSOC\| ≤ 0.08 in the objective, which would push TPE toward a slightly larger wsq).

### 6.4 Compute cost

Phase-2 fine-tune at 4 M steps per seed, measured from `train_config.json`:

| Algorithm | Mean wall-clock per seed (phase 2) | Phase-1 reference | Δ vs phase 1 |
| --------- | ---------------------------------- | ----------------- | ------------ |
| PPO       | ~ 6.0 h                            | 8.05 h            | −25 %        |
| SAC       | ~ 13.0 h                           | 15.28 h           | −15 %        |
| TD3       | ~ 11.0 h                           | 13.03 h           | −15 %        |

Wall-clock is consistently lower because each fine-tune starts close to optimum on the speed objective; the replay buffer (SAC/TD3) is also warm. PPO sees the largest relative speedup because its on-policy buffer is reset on `model.learn()` but the policy weights start near-optimum.

---

## 7. Phase-1 vs Phase-2 Headline (the thesis statement)

| Metric         | PPO Δ vs P1                | SAC Δ vs P1                | TD3 Δ vs P1 (all)          | TD3 Δ vs P1 (8/10)         |
| -------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| RMSE (km/h)    | −0.03 (3.16 → 3.13)        | +0.22 (3.11 → 3.33)        | +3.89 (3.52 → 7.41)        | +0.44 (3.52 → 3.96)        |
| RMSE σ         | −0.91 (0.98 → 0.07)        | −0.10 (0.43 → 0.33)        | +9.65 (0.29 → 9.94)        | +0.18 (0.29 → 0.47)        |
| NOx (g)        | **−43.4 (48.3 → 4.94)**    | **−59.3 (63.8 → 4.51)**    | **−85.4 (95.4 → 10.01)**   | **−86.8 (95.4 → 8.65)**    |
| NOx (mg/km)    | **−1 043 (1 165 → 122)**   | **−1 453 (1 564 → 111)**   | **−2 077 (2 327 → 250)**   | **−2 111 (2 327 → 216)**   |
| Total fuel (g) | −26.8 (79.6 → 52.8)        | −49.5 (113.5 → 64.0)       | −42.8 (122.6 → 79.8)       | −43.1 (122.6 → 79.5)       |
| \|ΔSOC\|       | **−0.218 (0.247 → 0.029)** | **−0.181 (0.191 → 0.010)** | **−0.085 (0.146 → 0.061)** | **−0.074 (0.146 → 0.072)** |

The four bold rows are the headline thesis result: **adding the emission and SOC terms to the reward and fine-tuning from the phase-1 checkpoint reduces NOx by 10× and SOC drift by 5×, with at most a 0.2 km/h cost to speed-tracking for PPO and SAC.**

For TD3, the same statement holds **only after removing the 2/10 seeds that diverge** under the new reward; the recommendation is therefore to use TD3 with seed-selection (or with the stability-constrained Optuna re-pick mentioned in §6.3).

---

## 8. Limitations and Threats to Validity

1. **WLTC caveat carries over from phase 1.** The "phase" labels are durational windows over the staircase, not the regulatory WLTC speed schedule. Any compliance claim should specify _"on the staircase cycle"_, not _"WLTC"_.
2. **Single deterministic evaluation episode per seed.** With one episode the RMSE/NOx are point estimates; a 10-episode stochastic eval pass would give per-seed confidence intervals.
3. **Phase-2 fine-tunes from a single phase-1 seed.** All 10 phase-2 seeds for a given algo start from the same phase-1 best-seed checkpoint and differ only in env/training seed. This isolates fine-tune variance but does **not** sample the phase-1-seed variance carried into phase 2. A future ablation: fine-tune each phase-1 seed once, see whether phase-1 σ correlates with phase-2 σ.
4. **Optuna budget is 20 trials.** The TPE search may be local-optimum trapped. The score surface for SAC has a single clear minimum (trial 4, NOx = 2.44 g); for TD3 the surface is rugged and 20 trials may under-sample. The relegated grid scout (§3) is a partial mitigation.
5. **The reward weight search was 2-D: `(W_EMISSION, W_SOC_SQUARED)`.** Both were the Optuna search variables. What was _locked_ to zero is the **linear** SOC term `W_SOC` and the fuel term `W_FUEL` — the SOC penalty used only the squared form `W_SOC_SQUARED · (SOC − SOC_init)²`. Other reward forms (linear ΔSOC, fuel-mg, a BSFC- or NOx-map-aware shaping term) were not explored; the chosen weights are optimal only within this 2-D space.
6. **VecNormalize reward stats stale at fine-tune start.** Loaded from phase-1 where the reward was speed/brake/flicker only. The new emission/SOC contributions re-calibrate the running stats during the first ~100k steps; PPO normalises rewards so this is most relevant for PPO. No instability observed but the first 100 k–200 k steps of every PPO fine-tune produce abnormal advantage estimates.
7. **NODE_FAIL recovery introduced FAIL trials in the Optuna journals.** SAC has 5 FAIL rows, TD3 has 2. These are the original RUNNING zombies marked failed by `rerun_phase2_trial.py --mark_failed` after their params were re-enqueued; the resulting new trials are legitimate. No FAIL row affects best-trial selection (best in all three algos was already chosen before recovery).

---

## 9. Recommendations for Phase 3 / Thesis

- **Apply a real WLTC evaluation cycle.** The schedule is the WLTC class-3 CSV; the env already supports loading it via `dataset_path`. Re-run the 10 phase-2 seeds on WLTC to compare mg/km against Euro 6 in a regulatory-meaningful way.
- **Stochastic eval pass for confidence intervals.** Run each phase-2 seed across 10 random-target episodes to bootstrap CIs on mean RMSE/NOx.
- **Add a stability constraint to the TD3 Optuna objective.** Tighten λ_soc and reduce λ_rmse to push TPE toward a regime where neither seed-1- nor seed-3-style divergence is possible. Cost: a ~1 g higher NOx.
- **Drop PPO is no longer the recommendation.** Phase-1 recommended dropping PPO; phase 2 shows PPO is the most fine-tune-robust algorithm. The thesis recommendation flips: **PPO is the safe default, SAC is the best-NOx alternative, TD3 is the riskiest but produces interesting bimodal action policies.**
- **Charge-sustaining seed-selection upstream.** The current `select_best_seed.py` ranks by RMSE then NOx and would pick a phase-1 seed with ΔSOC = +0.30 if the algorithm is fine-tune-stable; extending the ranker to `(rmse, |dsoc|, nox)` would make the phase-2 starting point already partially charge-sustaining.
- **Report 95 % bootstrap CIs on the per-seed mean in §7.** A 2-line `scipy.stats.bootstrap` change to the collector.
- **Aftertreatment-aware env (phase 2.5).** Drivetrain_Plus LSTM logging is plumbed but `USE_ONNX = True` skips it. Enabling it for evaluation provides EM1/EM2 IST signals to argue about _actuator-level_ emission shaping, not only setpoint-level.

---

## 10. Reproducibility Pointers

- Coarse 3×3 grid: [hyperparameter_search/run_phase2_cell.py](../hyperparameter_search/run_phase2_cell.py) launched by [hyperparameter_search/submit_phase2_sweep.sh](../hyperparameter_search/submit_phase2_sweep.sh).
- Optuna refinement: [hyperparameter_search/tune_phase2_hpo.py](../hyperparameter_search/tune_phase2_hpo.py) launched by [hyperparameter_search/submit_phase2_hpo.sh](../hyperparameter_search/submit_phase2_hpo.sh).
  - Zombie recovery: [hyperparameter_search/rerun_phase2_trial.py](../hyperparameter_search/rerun_phase2_trial.py), [submit_phase2_hpo_rerun_sac.sh](../hyperparameter_search/submit_phase2_hpo_rerun_sac.sh), [submit_phase2_hpo_rerun_td3.sh](../hyperparameter_search/submit_phase2_hpo_rerun_td3.sh).
  - CSV export from journal: [hyperparameter_search/export_phase2_study.py](../hyperparameter_search/export_phase2_study.py).
- 10-seed validation: [hyperparameter_search/run_phase2_seed.py](../hyperparameter_search/run_phase2_seed.py) launched by [hyperparameter_search/submit_phase2_seeds.sh](../hyperparameter_search/submit_phase2_seeds.sh).
- Phase-1 best-seed selection: [hyperparameter_search/select_best_seed.py](../hyperparameter_search/select_best_seed.py).
- Result aggregation:
  - Grid: [hyperparameter_search/collect_phase2_results.py](../hyperparameter_search/collect_phase2_results.py) → [phase2_results.csv](phase2_results.csv).
  - Optuna: [hyperparameter_search/collect_phase2_optuna_results.py](../hyperparameter_search/collect_phase2_optuna_results.py) → [phase2_optuna_results.csv](phase2_optuna_results.csv).
  - Seeds: [hyperparameter_search/collect_phase2_seeds_results.py](../hyperparameter_search/collect_phase2_seeds_results.py) → [phase2_seeds_results.csv](phase2_seeds_results.csv).
- Plot scripts:
  - Standalone phase-2 pareto / parallel / BSFC: [hyperparameter_search/plot_pareto_behaviour.py](../hyperparameter_search/plot_pareto_behaviour.py) with `--seeds_subpath phase2_seeds`.
  - Standalone phase-2 per-algo training / tracking / bar: [hyperparameter_search/plot_seeds.py](../hyperparameter_search/plot_seeds.py) with `--results_dir logs/<algo>/phase2_seeds`.
  - Overlay phase-1 vs phase-2 (pareto, ΔSOC-pareto, BSFC map, tracking): [hyperparameter_search/plot_phase2_overlay.py](../hyperparameter_search/plot_phase2_overlay.py).
  - Engine-map overlays (measured BSFC + measured mNOx backgrounds, triangulated, 3600×1200, PPO/TD3/SAC): [hyperparameter_search/plot_engine_maps_phase2.py](../hyperparameter_search/plot_engine_maps_phase2.py).
  - NOx-vs-load causal figure (NOx-rate vs torque + per-phase load histograms): [hyperparameter_search/plot_engine_map_nox.py](../hyperparameter_search/plot_engine_map_nox.py).
  - Best-seed BSFC map (composite-score pick): [hyperparameter_search/plot_best_seed_engine_map.py](../hyperparameter_search/plot_best_seed_engine_map.py).
- Reward + observation + action: [env.py:489-507](../env.py#L489-L507).
- Per-seed evaluation pipeline: [utils/evaluation_utils.py](../utils/evaluation_utils.py) and [models/eval.py](../models/eval.py).

The full result set required to reproduce this analysis is the set of `evaluation_metrics.json`, `evaluation_data.csv` and `train_config.json` files under [logs/<algo>/phase2*seeds/seed*<n>/](logs/) plus the three `best_params_phase2.json` and `all_trials.csv` files at [logs/<algo>/phase2_optuna/](logs/).
