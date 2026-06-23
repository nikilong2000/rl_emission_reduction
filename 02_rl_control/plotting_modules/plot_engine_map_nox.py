"""
NOx engine-map overlay — the correct lens for the phase-2 emission story.

The BSFC map (fuel-efficiency) is *anti-correlated* with NOx in a diesel: the
low-BSFC "sweet spot" is the high-load / high-combustion-temperature region
where NOx production is highest. A NOx-minimising policy therefore moves AWAY
from the BSFC sweet spot toward lower-load operation. Plotting operating points
on a BSFC map makes phase-2 look "worse" when it is in fact correctly trading a
little fuel-efficiency for a large NOx reduction.

Produces two standalone figures:
  nox_rate_vs_torque.png       — pooled NOx rate (mean + IQR) vs engine torque
  engine_load_distribution.png — per-algo torque KDE + histogram, Phase-1 (top)
                                 and Phase-2 (bottom) as separate subplots

Usage:
    python plot_engine_map_nox.py
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from plot_pareto_behaviour import ALGO_STYLE

FIG_DPI = 180


def pool_steps(root, sub, algo):
    frames = []
    for f in sorted(glob(os.path.join(root, algo, sub, "seed_*", "evaluation_data.csv"))):
        frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[(df.engine_on > 0.5) & (df.ice_torque > 0)]


def plot_nox_rate_curve(p1_by_algo, p2_by_algo, algos, out_path):
    """Standalone: pooled NOx rate (mean + IQR) vs engine torque."""
    pooled = pd.concat(
        [d for d in list(p1_by_algo.values()) + list(p2_by_algo.values()) if not d.empty],
        ignore_index=True,
    )
    edges = np.arange(0, 330, 30)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(pooled.ice_torque.values, edges) - 1
    mean_nox, lo, hi = [], [], []
    for b in range(len(centers)):
        v = pooled.nox.values[idx == b] * 1000.0
        if len(v) >= 20:
            mean_nox.append(v.mean())
            lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75))
        else:
            mean_nox.append(np.nan)
            lo.append(np.nan)
            hi.append(np.nan)
    mean_nox = np.array(mean_nox)
    lo = np.array(lo)
    hi = np.array(hi)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(centers, mean_nox, "o-", color="#b30000", lw=2, label="mean")
    ax.fill_between(centers, lo, hi, color="#b30000", alpha=0.18, label="IQR")
    ax.set_xlabel("Engine torque (Nm)")
    ax.set_ylabel("Tail-pipe NOx rate (mg/s)")
    ax.set_title("NOx rate grows steeply with engine load (pooled, both phases)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_load_distribution(p1_by_algo, p2_by_algo, algos, out_path):
    """Two subplots (Phase-1 top, Phase-2 bottom): per-algo torque histogram
    (semi-transparent fill) with KDE density curves overlaid."""
    try:
        from scipy.stats import gaussian_kde
        have_kde = True
    except ImportError:
        have_kde = False

    bins = np.arange(0, 320, 20)
    xgrid = np.linspace(0, 310, 600)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    phase_pairs = [("Phase 1", p1_by_algo), ("Phase 2", p2_by_algo)]

    for ax, (phase_label, by_algo) in zip(axes, phase_pairs):
        handles = []
        for algo in algos:
            df = by_algo[algo]
            if df.empty:
                continue
            color = ALGO_STYLE[algo]["color"]
            label = ALGO_STYLE[algo]["label"]
            trq = df.ice_torque.values

            ax.hist(
                trq,
                bins=bins,
                density=True,
                alpha=0.25,
                color=color,
                edgecolor="none",
            )

            if have_kde and len(trq) > 10 and np.std(trq) > 1e-6:
                kde = gaussian_kde(trq)
                ax.plot(xgrid, kde(xgrid), color=color, lw=2.0, label=label)
                handles.append(
                    Line2D([0], [0], color=color, lw=2.0, label=label)
                )

        ax.set_ylabel("Density")
        ax.set_title(phase_label)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 310)
        if handles:
            ax.legend(handles=handles, fontsize=9, ncol=len(algos))

    axes[-1].set_xlabel("Engine torque (Nm)")
    fig.suptitle("Engine-load distribution per algorithm", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase1_logs",
                   default=os.path.join(here, "..", "logs_cluster_phase1", "logs"))
    p.add_argument("--phase2_logs",
                   default=os.path.join(here, "..", "logs_cluster_phase2", "logs"))
    p.add_argument("--phase1_subpath", default="optuna/seeds")
    p.add_argument("--phase2_subpath", default="phase2_seeds")
    p.add_argument("--out_dir",
                   default=os.path.join(here, "..", "logs_cluster_phase2", "analysis_plots"))
    p.add_argument("--algos", nargs="+", default=["ppo", "sac", "td3"])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    p1_by_algo, p2_by_algo = {}, {}
    for algo in args.algos:
        p1_by_algo[algo] = pool_steps(args.phase1_logs, args.phase1_subpath, algo)
        p2_by_algo[algo] = pool_steps(args.phase2_logs, args.phase2_subpath, algo)

    plot_nox_rate_curve(
        p1_by_algo, p2_by_algo, args.algos,
        os.path.join(args.out_dir, "nox_rate_vs_torque.png"),
    )
    plot_load_distribution(
        p1_by_algo, p2_by_algo, args.algos,
        os.path.join(args.out_dir, "engine_load_distribution.png"),
    )

    print("\nMean engine operating point (engine-on steps):")
    for algo in args.algos:
        s1, s2 = p1_by_algo[algo], p2_by_algo[algo]
        print(
            f"  {algo.upper()}: P1 {s1.ice_speed_rpm.mean():.0f}rpm/"
            f"{s1.ice_torque.mean():.0f}Nm/{s1.nox.mean()*1000:.1f}mg·s⁻¹  ->  "
            f"P2 {s2.ice_speed_rpm.mean():.0f}rpm/{s2.ice_torque.mean():.0f}Nm/"
            f"{s2.nox.mean()*1000:.1f}mg·s⁻¹"
        )


if __name__ == "__main__":
    main()
