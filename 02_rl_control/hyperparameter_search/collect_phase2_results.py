"""
Collect phase-2 sweep evaluation metrics into a single CSV for analysis.

Usage:
    python collect_phase2_results.py
    python collect_phase2_results.py --logs_dir 02_rl_control/logs_cluster/logs \
        --output 02_rl_control/logs_cluster/phase2_results.csv
"""

import os
import json
import glob
import argparse
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
rl_control_dir = os.path.dirname(current_dir)


def collect(logs_dir: str) -> pd.DataFrame:
    rows = []
    for algo in ["ppo", "sac", "td3"]:
        cells = sorted(glob.glob(os.path.join(logs_dir, algo, "phase2", "cell_*")))
        for cd in cells:
            mfile = os.path.join(cd, "evaluation_metrics.json")
            cfile = os.path.join(cd, "train_config.json")
            if not (os.path.exists(mfile) and os.path.exists(cfile)):
                print(f"Skipping (missing files): {cd}")
                continue
            with open(mfile) as f:
                m = json.load(f)
            with open(cfile) as f:
                c = json.load(f)
            rows.append(
                {
                    "algo": algo,
                    "cell_id": c.get("cell_id"),
                    "cell_dir": os.path.basename(cd),
                    "w_emission": c.get("w_emission"),
                    "w_soc_squared": c.get("w_soc_squared"),
                    "phase1_best_seed": c.get("phase1_best_seed"),
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
                    "abs_path": cd,
                }
            )
    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs_dir",
        default=os.path.join(rl_control_dir, "logs_cluster", "logs"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(rl_control_dir, "logs_cluster", "phase2_results.csv"),
    )
    args = parser.parse_args()

    df = collect(args.logs_dir)
    if df.empty:
        print(f"No phase-2 cells found under {args.logs_dir}")
        return
    df = df.sort_values(["algo", "cell_id"]).reset_index(drop=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows -> {args.output}")
    print(df[["algo", "cell_id", "w_emission", "w_soc_squared",
              "rmse_speed_kmph", "total_nox_g", "max_abs_soc_drift"]])


if __name__ == "__main__":
    main()
