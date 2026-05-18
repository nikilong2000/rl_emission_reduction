"""
Merge per-algo phase-2 Optuna `all_trials.csv` into a single sortable CSV
with phase-1 baseline deltas, mirroring `collect_phase2_results.py`.

Usage:
    python collect_phase2_optuna_results.py
    python collect_phase2_optuna_results.py \
        --logs_dir 02_rl_control/logs_cluster/logs \
        --output 02_rl_control/logs_cluster/phase2_optuna_results.csv
"""

import os
import glob
import argparse
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
rl_control_dir = os.path.dirname(current_dir)

# Phase-1 baselines (best seed per algo) — same source as
# collect_phase2_results.py / phase2_analysis.ipynb
PHASE1_BASELINE = {
    "ppo": {"seed": 3, "rmse": 3.163, "nox_g": 5.57,  "delta_soc": 0.2992},
    "sac": {"seed": 3, "rmse": 3.106, "nox_g": 48.94, "delta_soc": 0.2987},
    "td3": {"seed": 1, "rmse": 3.515, "nox_g": 73.42, "delta_soc": 0.2999},
}


def _resolve_trial_dir(optuna_dir, trial_number):
    matches = sorted(
        glob.glob(os.path.join(optuna_dir, f"trial_{int(trial_number):03d}_*"))
    )
    return matches[0] if matches else None


def collect(logs_dir: str) -> pd.DataFrame:
    frames = []
    for algo in ["ppo", "sac", "td3"]:
        optuna_dir = os.path.join(logs_dir, algo, "phase2_optuna")
        csv_path = os.path.join(optuna_dir, "all_trials.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping (missing all_trials.csv): {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df["algo"] = algo
        df["trial_dir"] = df["number"].apply(
            lambda n: os.path.basename(_resolve_trial_dir(optuna_dir, n))
            if _resolve_trial_dir(optuna_dir, n) else ""
        )
        df["abs_path"] = df["number"].apply(
            lambda n: _resolve_trial_dir(optuna_dir, n) or ""
        )

        p1 = PHASE1_BASELINE[algo]
        df["phase1_best_seed"] = p1["seed"]
        df["p1_rmse"] = p1["rmse"]
        df["p1_nox"] = p1["nox_g"]
        df["p1_dsoc"] = p1["delta_soc"]
        df["drmse_pct"] = (df["rmse_speed_kmph"] - p1["rmse"]) / p1["rmse"] * 100
        df["dnox_pct"] = (df["total_nox_g"] - p1["nox_g"]) / p1["nox_g"] * 100
        df["abs_dsoc"] = df["delta_soc"].abs() if "delta_soc" in df.columns else float("nan")
        # Normalise state column (TrialState.COMPLETE -> COMPLETE)
        if "state" in df.columns:
            df["state"] = df["state"].astype(str).str.replace("TrialState.", "", regex=False)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    # Column ordering — mirrors phase2_results.csv layout where possible
    front = [
        "algo", "number", "state", "trial_dir", "w_emission", "w_soc_squared",
        "phase1_best_seed",
        "value", "rmse_speed_kmph", "total_nox_g",
        "delta_soc", "abs_dsoc", "max_abs_soc_drift", "rms_soc_drift",
        "rmse_penalty", "soc_penalty",
        "drmse_pct", "dnox_pct",
        "p1_rmse", "p1_nox", "p1_dsoc",
        "duration_s", "datetime_start", "datetime_complete",
        "abs_path",
    ]
    cols = [c for c in front if c in out.columns] + [
        c for c in out.columns if c not in front
    ]
    return out[cols]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs_dir",
        default=os.path.join(rl_control_dir, "logs_cluster", "logs"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            rl_control_dir, "logs_cluster", "phase2_optuna_results.csv"
        ),
    )
    args = parser.parse_args()

    df = collect(args.logs_dir)
    if df.empty:
        print(f"No Optuna phase-2 results found under {args.logs_dir}")
        return
    df = df.sort_values(["algo", "number"]).reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows -> {args.output}")

    # Quick summary per algo: count by state + best COMPLETE row
    print()
    for algo in df["algo"].unique():
        sub = df[df["algo"] == algo]
        states = sub["state"].value_counts().to_dict() if "state" in sub.columns else {}
        print(f"{algo.upper()}: {len(sub)} trials  states={states}")
        completed = sub[sub.get("state") == "COMPLETE"].dropna(subset=["value"])
        if not completed.empty:
            best = completed.sort_values("value").iloc[0]
            print(
                f"  best: trial {int(best['number'])}  "
                f"we={best['w_emission']:.4f}  wsq={best['w_soc_squared']:.2f}  "
                f"score={best['value']:.4f}  "
                f"rmse={best['rmse_speed_kmph']:.3f}  "
                f"nox={best['total_nox_g']:.3f}g  "
                f"max|dSOC|={best['max_abs_soc_drift']:.4f}"
            )


if __name__ == "__main__":
    main()
