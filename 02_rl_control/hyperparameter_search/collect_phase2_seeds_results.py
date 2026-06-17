"""
Aggregate phase-2 10-seed validation metrics into one CSV.

Reads logs_cluster/logs/<algo>/phase2_seeds/seed_*/evaluation_metrics.json,
joins with each train_config.json (for reward weights actually used), and
appends per-row deltas vs phase-1 baseline (best seed per algo).

Usage:
    python collect_phase2_seeds_results.py
    python collect_phase2_seeds_results.py \
        --logs_dir 02_rl_control/logs_cluster/logs \
        --output  02_rl_control/logs_cluster/phase2_seeds_results.csv
"""

import os
import re
import json
import glob
import argparse
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
rl_control_dir = os.path.dirname(current_dir)

# Phase-1 baselines (best seed per algo) — same source as collect_phase2_*.py
PHASE1_BASELINE = {
    "ppo": {"seed": 3, "rmse": 3.163, "nox_g": 5.57,  "delta_soc": 0.2992},
    "sac": {"seed": 3, "rmse": 3.106, "nox_g": 48.94, "delta_soc": 0.2987},
    "td3": {"seed": 1, "rmse": 3.515, "nox_g": 73.42, "delta_soc": 0.2999},
}


def collect(logs_dir: str) -> pd.DataFrame:
    rows = []
    for algo in ["ppo", "sac", "td3"]:
        seeds = sorted(
            glob.glob(os.path.join(logs_dir, algo, "phase2_seeds", "seed_*"))
        )
        for sd in seeds:
            # Skip auxiliary dirs like "seed_7_wltc" or "seed_7 copy"; only
            # canonical "seed_<int>" runs are part of the 10-seed validation.
            if not re.fullmatch(r"seed_\d+", os.path.basename(sd)):
                continue
            mfile = os.path.join(sd, "evaluation_metrics.json")
            cfile = os.path.join(sd, "train_config.json")
            if not (os.path.exists(mfile) and os.path.exists(cfile)):
                print(f"Skipping (missing files): {sd}")
                continue
            with open(mfile) as f:
                m = json.load(f)
            with open(cfile) as f:
                c = json.load(f)
            seed_id = int(os.path.basename(sd).replace("seed_", ""))
            rows.append(
                {
                    "algo": algo,
                    "seed": seed_id,
                    "phase1_best_seed": c.get("phase1_best_seed"),
                    "w_emission": c.get("w_emission"),
                    "w_soc_squared": c.get("w_soc_squared"),
                    "total_timesteps": c.get("total_timesteps"),
                    "training_duration_hms": c.get("training_duration_hms"),
                    "total_reward": m.get("total_reward"),
                    "rmse_speed_kmph": m.get("rmse_speed_kmph"),
                    "mae_speed_kmph": m.get("mae_speed_kmph"),
                    "total_nox_g": m.get("total_nox_g"),
                    "total_fuel_g": m.get("total_fuel_g"),
                    "initial_soc": m.get("initial_soc"),
                    "final_soc": m.get("final_soc"),
                    "delta_soc": m.get("delta_soc"),
                    "max_abs_soc_drift": m.get("max_abs_soc_drift"),
                    "rms_soc_drift": m.get("rms_soc_drift"),
                    "abs_path": sd,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["p1_rmse"] = df["algo"].map(lambda a: PHASE1_BASELINE[a]["rmse"])
    df["p1_nox"] = df["algo"].map(lambda a: PHASE1_BASELINE[a]["nox_g"])
    df["p1_dsoc"] = df["algo"].map(lambda a: PHASE1_BASELINE[a]["delta_soc"])
    df["drmse_pct"] = (df["rmse_speed_kmph"] - df["p1_rmse"]) / df["p1_rmse"] * 100
    df["dnox_pct"] = (df["total_nox_g"] - df["p1_nox"]) / df["p1_nox"] * 100
    df["abs_dsoc"] = df["delta_soc"].abs()
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs_dir",
        default=os.path.join(rl_control_dir, "logs_cluster", "logs"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(rl_control_dir, "logs_cluster", "phase2_seeds_results.csv"),
    )
    args = parser.parse_args()

    df = collect(args.logs_dir)
    if df.empty:
        print(f"No phase-2 seeds found under {args.logs_dir}")
        return
    df = df.sort_values(["algo", "seed"]).reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows -> {args.output}\n")

    for algo in df["algo"].unique():
        sub = df[df["algo"] == algo]
        p1 = PHASE1_BASELINE[algo]
        print(f"=== {algo.upper()} (n={len(sub)}) ===")
        print(
            f"  RMSE  : {sub.rmse_speed_kmph.mean():.3f} ± {sub.rmse_speed_kmph.std():.3f} "
            f"(p1={p1['rmse']:.3f})"
        )
        print(
            f"  NOx g : {sub.total_nox_g.mean():.3f} ± {sub.total_nox_g.std():.3f} "
            f"(p1={p1['nox_g']:.2f}, Δ={sub.dnox_pct.mean():+.1f}%)"
        )
        print(
            f"  fuel g: {sub.total_fuel_g.mean():.2f} ± {sub.total_fuel_g.std():.2f}"
        )
        print(
            f"  |dSOC|: {sub.abs_dsoc.mean():.4f} ± {sub.abs_dsoc.std():.4f} "
            f"(p1={abs(p1['delta_soc']):.3f})"
        )
        print(
            f"  max|dSOC|: {sub.max_abs_soc_drift.mean():.4f} ± "
            f"{sub.max_abs_soc_drift.std():.4f}"
        )


if __name__ == "__main__":
    main()
