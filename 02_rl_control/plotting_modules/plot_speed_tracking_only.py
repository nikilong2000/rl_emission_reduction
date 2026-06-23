"""
Speed-tracking trajectory plot (top panel only) for every algorithm and phase.

Recreates the *upper* subplot of ``eval_rmse_tracking.png`` produced by
``plot_seeds.py`` -- mean actual speed with a +/-1 std band against the target
-- but as a standalone single-panel figure (no absolute-error subplot), for
PPO/SAC/TD3 in both Phase 1 and Phase 2.

Seed evaluation data is read from each seed's ``evaluation_data.csv``:
  Phase 1: logs_cluster_phase1/logs/<algo>/optuna/seeds/seed_*/
  Phase 2: logs_cluster_phase2/logs/<algo>/phase2_seeds/seed_*/

One figure per (algo, phase) is written next to the seeds, named
``eval_speed_tracking.png``.

Usage:
    python plot_speed_tracking_only.py
"""

from __future__ import annotations

import argparse
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Match the styling of plot_seeds.py
MEAN_COLOR = "#1565C0"
STD_COLOR = "#BDBDBD"
TARGET_COLOR = "#E53935"
GRID_ALPHA = 0.35
FIG_DPI = 200
DT = 0.5  # seconds per step

ALGOS = ["ppo", "sac", "td3"]
SEED_RE = re.compile(r"^seed_\d+$")  # excludes stray dirs like "seed_7 copy"


def _seed_dirs(root):
    out = []
    for d in sorted(glob(os.path.join(root, "seed_*"))):
        if os.path.isdir(d) and SEED_RE.match(os.path.basename(d)):
            out.append(d)
    return out


def plot_speed_tracking(seed_dirs, out_path, title):
    actual, target = [], []
    for sd in seed_dirs:
        csv_path = os.path.join(sd, "evaluation_data.csv")
        if not os.path.exists(csv_path):
            print(f"  warning: {csv_path} missing -- skipped")
            continue
        df = pd.read_csv(csv_path)
        actual.append(df["speed_actual"].values)
        target.append(df["speed_target"].values)

    if not actual:
        print(f"  no evaluation data for {title} -- skipped")
        return

    min_len = min(len(a) for a in actual)
    actual_mat = np.array([a[:min_len] for a in actual])
    target_mat = np.array([t[:min_len] for t in target])
    actual_mean = actual_mat.mean(axis=0)
    actual_std = actual_mat.std(axis=0)
    target_mean = target_mat.mean(axis=0)
    time_s = np.arange(min_len) * DT

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(time_s, target_mean, color=TARGET_COLOR, linewidth=1.5,
            linestyle="--", label="Target")
    ax.plot(time_s, actual_mean, color=MEAN_COLOR, linewidth=1.2,
            label="Actual (mean)")
    ax.fill_between(time_s, actual_mean - actual_std, actual_mean + actual_std,
                    color=STD_COLOR, alpha=0.5, label="$\\pm$ 1 Std")
    ax.set_title(f"{title} ({len(actual)} Seeds)", fontsize=13,
                 fontweight="bold", pad=10)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Speed (km/h)", fontsize=11)
    ax.grid(True, alpha=GRID_ALPHA, linewidth=0.5)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase1_logs",
                   default=os.path.join(here, "..", "logs_cluster_phase1", "logs"))
    p.add_argument("--phase2_logs",
                   default=os.path.join(here, "..", "logs_cluster_phase2", "logs"))
    args = p.parse_args()

    phases = [
        ("Phase 1", args.phase1_logs, "optuna/seeds"),
        ("Phase 2", args.phase2_logs, "phase2_seeds"),
    ]
    for phase_label, root, sub in phases:
        for algo in ALGOS:
            seeds_root = os.path.join(root, algo, sub)
            seed_dirs = _seed_dirs(seeds_root)
            print(f"{phase_label} {algo.upper()}: {len(seed_dirs)} seeds")
            if not seed_dirs:
                continue
            title = f"{algo.upper()} {phase_label} — Speed Tracking"
            out_path = os.path.join(seeds_root, "eval_speed_tracking.png")
            plot_speed_tracking(seed_dirs, out_path, title)


if __name__ == "__main__":
    main()
