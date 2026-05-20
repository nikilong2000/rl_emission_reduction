# Best-Seed Action-Policy Comparison (PPO vs SAC vs TD3)

Companion to [PHASE1_ANALYSIS.md](PHASE1_ANALYSIS.md). The fingerprint comparison in §4.5 of that document used **seed 0** of each algorithm, which is convenient but arbitrary. This document instead compares the **best seed of each algorithm — the one with the lowest speed RMSE on the deterministic staircase evaluation cycle**, i.e. the policy that would be carried into phase 2 by `select_best_seed.py`. Looking at the *deployed* policy rather than an arbitrary seed changes several conclusions and exposes one outright error in the original analysis.

## Best seed per algorithm

Selected by minimum `rmse_speed_kmph` over the ten `evaluation_metrics.json` files (matching the `select_best_seed.py` primary sort key):

| Algorithm | Best seed | RMSE (km/h) | MAE (km/h) | Total NOx (g) | Total fuel (g) | ΔSOC | Total reward |
|-----------|----------:|------------:|-----------:|--------------:|---------------:|-----:|-------------:|
| PPO       | **seed_3** | 3.163      | 1.001      | **5.57**      | **74.46**      | +0.299 | 3 546.0    |
| SAC       | **seed_3** | **3.106**  | **0.891**  | 48.94         | 109.31         | +0.299 | 3 544.4    |
| TD3       | **seed_1** | 3.515      | 1.621      | 73.42         | 163.94         | +0.300 | 3 499.5    |

Bold = best of the three for that column. SAC seed_3 is the single best speed tracker in the whole cohort (RMSE 3.106, MAE 0.891). PPO seed_3 is by far the cleanest deployed policy (NOx 5.57 g, lowest fuel). TD3 seed_1 is the weakest of the three best (highest fuel and NOx, worst RMSE).

> **Note on directory:** the result tree is now under [logs_cluster_phase1/logs/](logs_cluster_phase1/logs/). The links in PHASE1_ANALYSIS.md still point at the old `logs_cluster/` name and should be updated.

---

## Fingerprints

Source plots: `action_distribution.png` and `state_action_occupancy.png` in each best-seed directory.

### PPO — seed_3 — "lean high-RPM cruise" (cleanest)

[action_distribution.png](logs_cluster_phase1/logs/ppo/optuna/seeds/seed_3/action_distribution.png) · [state_action_occupancy.png](logs_cluster_phase1/logs/ppo/optuna/seeds/seed_3/state_action_occupancy.png)

* **engine_on**: pinned to 1.0 — engine always on.
* **ice_speed_rpm**: a single sharp spike at ≈ 4 000 rpm (the rev-limit ceiling). The occupancy map confirms the engine sits at ~4 000 rpm across *every* target speed from 35 to 140 km/h — engine speed is decoupled from vehicle speed.
* **em2_torque**: tight around 0, with mass in the 0…+80 Nm band (mild generator) and a negligible spike at the +420 Nm limit.
* **fuel**: very tight unimodal at ≈ 17–23 mg — the leanest and narrowest fuel distribution of the three. This is the mechanism behind the 5.57 g NOx: high RPM but low fuel-per-stroke keeps the in-cylinder mixture lean, and the NOx-vs-fuel occupancy panel shows essentially all mass at < 1 g/s NOx.
* **brake**: ≈ 0.

The PPO policy converges on one operating point and parks the ICE there, modulating wheel torque almost entirely through the (small) EM2 contribution. Counter-intuitively, the **highest engine speed produces the lowest NOx**, because NOx tracks fuel/load, not RPM.

### SAC — seed_3 — "engine start/stop + fuel modulation" (best tracker)

[action_distribution.png](logs_cluster_phase1/logs/sac/optuna/seeds/seed_3/action_distribution.png) · [state_action_occupancy.png](logs_cluster_phase1/logs/sac/optuna/seeds/seed_3/state_action_occupancy.png)

* **engine_on**: mostly 1.0 but with a **distinct mass at 0.0** — the engine is switched **off** for a non-trivial fraction of the cycle. This is the only best-seed policy that uses the engine-off command at all.
* **ice_speed_rpm**: a spike at 0 rpm (engine off) plus a broad distribution from ≈ 2 400 to 4 000 rpm peaking around 3 500. The occupancy map shows engine speed rising with target speed — a more "proportional" use of the ICE than PPO's fixed point.
* **em2_torque**: large mass at 0, a negative lobe at −50…−100 Nm (regen/charging), a small +80 Nm band and a tiny +420 spike.
* **fuel**: a spike at 0 mg (coincident with engine-off), a main mode at ≈ 30–40 mg and a tail to 60 mg — richer and higher than PPO, giving 109 g fuel and 49 g NOx.
* **brake**: ≈ 0.

SAC achieves the lowest RMSE by *modulating* the powertrain (variable RPM, fuel cuts, occasional engine-off) rather than parking at one operating point. The price is ~9× the NOx and ~1.5× the fuel of PPO's best seed.

### TD3 — seed_1 — "low-RPM, bimodal fuel, bang-bang EM2" (dirtiest of the three)

[action_distribution.png](logs_cluster_phase1/logs/td3/optuna/seeds/seed_1/action_distribution.png) · [state_action_occupancy.png](logs_cluster_phase1/logs/td3/optuna/seeds/seed_1/state_action_occupancy.png)

* **engine_on**: pinned to 1.0 — engine always on.
* **ice_speed_rpm**: bimodal at ≈ 900–1 300 rpm and ≈ 2 000–2 500 rpm (both well below PPO/SAC), with a negligible 4 000 rpm spike. TD3 runs the engine *slow*.
* **em2_torque**: main mass at 0, a wide negative lobe at −150…−280 Nm and a positive lobe at +300…+420 Nm — the EM2 is used in a near-bang-bang split between strong regen and strong motoring.
* **fuel**: strongly bimodal — a low cluster at 10–20 mg and a high cluster at 50–70 mg. The high cluster is what pushes total fuel to 164 g (highest of all three best seeds) and NOx to 73 g.
* **brake**: ≈ 0.

TD3 compensates for low engine speed with high fuel-per-stroke in its high cluster, which is the worst combination for NOx. Its bimodal EM2 use is the same "all-or-nothing" battery pattern noted for TD3 seed_0 in the original analysis — so this trait is *seed-robust* for TD3, not a seed_0 artefact.

---

## Side-by-side

| Dimension            | PPO seed_3          | SAC seed_3                    | TD3 seed_1                       |
|----------------------|---------------------|-------------------------------|----------------------------------|
| engine_on            | always 1            | **1 + off bursts**            | always 1                         |
| ICE speed            | fixed ≈ 4 000 rpm   | variable 2 400–4 000 rpm      | low, bimodal 1 000 / 2 200 rpm   |
| EM2 torque           | ~0…+80 Nm (mild)    | ~0, −50…−100 regen            | bang-bang ±300–420 Nm            |
| Fuel                 | tight 17–23 mg (lean)| 30–40 mg + off                | bimodal 10–20 / 50–70 mg         |
| Strategy             | park at one point   | modulate + start/stop         | low-RPM + rich bursts            |
| NOx (g)              | **5.57**            | 48.94                         | 73.42                            |
| Fuel (g)             | **74.46**           | 109.31                        | 163.94                           |
| RMSE (km/h)          | 3.163               | **3.106**                     | 3.515                            |

**The clean–accurate trade-off is explicit here:** PPO's best seed wins on emissions/fuel by ~9× NOx but is 0.06 km/h worse on RMSE than SAC's best; SAC's best wins on tracking but at a large emission cost; TD3's best loses on both. Fuel ordering (74 < 109 < 164 g) predicts NOx ordering (5.6 < 49 < 73 g) exactly — confirming that **integrated fuel, not engine speed, is the NOx driver** in this surrogate.

---

## What changes relative to the seed_0 comparison in PHASE1_ANALYSIS.md

1. **Correction to Limitation #4.** PHASE1_ANALYSIS.md §7.4 states *"every algorithm has saturated `engine_on` to 1.0, the engine is always on … the flicker penalty is never paid."* This is **false for the deployed SAC policy**: SAC seed_3 commands engine-off for a visible fraction of the cycle and therefore *does* pay the W_FLICKER penalty on re-ignition. The "always-on" claim was a seed_0 sampling artefact. The discrete engine-toggle dimension is **not** globally inert — it is active in the best SAC policy and dormant in the best PPO/TD3 policies.

2. **Best-by-RMSE selection systematically picks battery-saturating seeds.** All three best seeds have ΔSOC ≈ +0.30 (battery driven to the 1.0 cap), whereas the seed_0 set included a TD3 depleter (ΔSOC = −0.12). This sharpens the §4.4 / §8 recommendation: ranking by RMSE alone guarantees a charge-saturating phase-2 starting checkpoint for *all three* algorithms. A `|ΔSOC|` tiebreaker is needed if a charge-neutral prior is wanted.

3. **Per-algorithm NOx ranking flips vs the seed-mean.** On seed means, PPO had the lowest mean NOx (48 g) and TD3 the highest (95 g). On *best seeds*, PPO is still cleanest but by a much wider margin (5.6 g — an order of magnitude below its own mean), because PPO seed_3 happens to combine the fixed-RPM strategy with the leanest fuel. This shows PPO's emission outcome is high-variance and that its best seed is unusually clean rather than typical.

4. **TD3's bang-bang EM2 is confirmed seed-robust.** The bimodal ±400 Nm EM2 pattern appears in both TD3 seed_0 and TD3 seed_1, so it is a property of the TD3 solution class in this environment, not of one seed.

---

## Implication for phase 2

If phase 2 starts from the lowest-RMSE checkpoints, it inherits:
* PPO: a clean but rigid single-operating-point policy with little headroom to reduce NOx further (already at 5.6 g).
* SAC: an accurate, modulating policy *that already uses start/stop* — the most promising substrate for adding an emission term, since it has the most behavioural degrees of freedom to exploit.
* TD3: a rich, dirty policy with the largest NOx-reduction headroom but the worst starting RMSE.

This supports the PHASE1_ANALYSIS.md recommendation to carry SAC and TD3 into the phase-2 reward sweep, and adds a concrete reason: **SAC's deployed policy is the only one already exercising the engine-off control authority that an emission-shaped reward will want to amplify.**
