"""
NOx reduction across the full training pipeline (Phase 1 -> Phase 2).

Pools the cumulative tail-pipe NOx of every run at each of the four pipeline
stages and visualises how the emission level descends as the pipeline
progresses:

    Stage 1  Phase-1 seeds      speed-tracking only, emission term OFF  (baseline)
    Stage 2  Grid scout         3x3 (W_EMISSION, W_SOC_SQUARED) cells
    Stage 3  Optuna refinement  TPE trials (completed only)
    Stage 4  Phase-2 seeds      10-seed validation at the chosen weights

The metric is the **total cycle NOx in grams** (`total_nox_g` from each run's
`evaluation_metrics.json`). It is used in preference to mg/km because the
Phase-1 metrics predate the per-run distance logging, and the evaluation cycle
(the deterministic staircase) is identical across all stages, so total grams is
directly comparable. The Euro-6 reference is drawn at the gram-equivalent of
80 mg/km over the nominal ~40.4 km cycle (~3.2 g).

Two standalone figures are produced:

  nox_pipeline_box.png          per-stage distribution (box + jittered strip),
                                dodged by algorithm  (Option 3)
  nox_pipeline_swarm_traj.png   all runs as a faint swarm + a bold per-stage
                                mean trajectory per algorithm + Euro-6 line
                                (Option 4)

Usage:
    python plot_nox_pipeline_progression.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from plot_pareto_behaviour import ALGO_STYLE

FIG_DPI = 180
ALGOS = ["ppo", "sac", "td3"]
STAGES = ["phase1", "grid", "optuna", "phase2"]
STAGE_LABELS = {
    "phase1": "Phase 1\nBaseline\n(10 Seeds)",
    "grid": "Reward Grid\n(3$\\times$3)",
    "optuna": "Reward Optuna\n(TPE)",
    "phase2": "Phase 2\nFine-Tuned\n(10 Seeds)",
}
# Nominal staircase cycle distance (km); used only for the Euro-6 reference line.
CYCLE_KM = 40.4
EURO6_G = 80.0 * CYCLE_KM / 1000.0  # 80 mg/km -> g over the nominal cycle


def _read_nox(metrics_path):
    try:
        with open(metrics_path) as f:
            d = json.load(f)
        v = d.get("total_nox_g")
        return float(v) if v is not None else None
    except (OSError, ValueError):
        return None


def collect(phase1_root, phase2_root, seeds_csv):
    """Return data[stage][algo] = np.ndarray of total_nox_g."""
    data = {s: {a: [] for a in ALGOS} for s in STAGES}

    # Stage 1 -- Phase-1 seeds
    for a in ALGOS:
        for f in sorted(
            glob(
                os.path.join(
                    phase1_root,
                    a,
                    "optuna",
                    "seeds",
                    "seed_*",
                    "evaluation_metrics.json",
                )
            )
        ):
            v = _read_nox(f)
            if v is not None:
                data["phase1"][a].append(v)

    # Stage 2 -- 3x3 grid scout
    for a in ALGOS:
        for f in sorted(
            glob(
                os.path.join(
                    phase2_root, a, "phase2", "cell_*", "evaluation_metrics.json"
                )
            )
        ):
            v = _read_nox(f)
            if v is not None:
                data["grid"][a].append(v)

    # Stage 3 -- Optuna refinement (completed trials only)
    for a in ALGOS:
        for f in sorted(
            glob(
                os.path.join(
                    phase2_root,
                    a,
                    "phase2_optuna",
                    "trial_*",
                    "evaluation_metrics.json",
                )
            )
        ):
            v = _read_nox(f)
            if v is not None:
                data["optuna"][a].append(v)

    # Stage 4 -- Phase-2 10-seed validation (from the collected CSV)
    df = pd.read_csv(seeds_csv)
    for a in ALGOS:
        data["phase2"][a] = df[df.algo == a].total_nox_g.to_numpy().tolist()

    for s in STAGES:
        for a in ALGOS:
            data[s][a] = np.array(data[s][a], dtype=float)
    return data


# Focus the log axis on the meaningful 2-100 g band. The lower bound also drops
# the single degenerate SAC grid cell (engine-off policy collapse, NOx ~1e-3 g)
# off the axis; it is flagged with a figure footnote instead.
Y_LIM = (1.5, 600.0)


def _euro6(ax):
    ax.axhline(EURO6_G, ls=":", lw=1.5, color="black", zorder=1)
    ax.text(
        0.012,
        EURO6_G,
        "Euro 6 ($\\approx$80 mg/km)",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="left",
        fontsize=8.5,
        color="black",
    )


def _degenerate_footnote(fig):
    fig.text(
        0.542,
        0.176,
        "Degenerate SAC grid cell (engine-off, NOx $\\approx$ 0) omitted "
        "by the axis range.",
        fontsize=7.5,
        color="dimgray",
        ha="left",
    )


# --------------------------------------------------------------------------- #
# Option 3 -- per-stage box + strip, dodged by algorithm
# --------------------------------------------------------------------------- #
def plot_box(data, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    n_alg = len(ALGOS)
    box_w = 0.22
    offsets = np.linspace(-(n_alg - 1) / 2, (n_alg - 1) / 2, n_alg) * (box_w + 0.04)
    rng = np.random.default_rng(0)

    for si, stage in enumerate(STAGES):
        for ai, algo in enumerate(ALGOS):
            vals = data[stage][algo]
            if vals.size == 0:
                continue
            x = si + offsets[ai]
            color = ALGO_STYLE[algo]["color"]
            bp = ax.boxplot(
                vals,
                positions=[x],
                widths=box_w,
                patch_artist=True,
                showfliers=False,
                zorder=2,
                medianprops=dict(color="black", lw=1.3),
                whiskerprops=dict(color=color, lw=1.1),
                capprops=dict(color=color, lw=1.1),
                boxprops=dict(facecolor=color, alpha=0.30, edgecolor=color, lw=1.2),
            )
            jx = x + (rng.random(vals.size) - 0.5) * box_w * 0.7
            ax.scatter(
                jx,
                vals,
                s=16,
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

    _euro6(ax)
    ax.set_yscale("log")
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGES])
    ax.set_ylabel("Total cycle NOx (g, log scale)")
    ax.set_title(
        "NOx reduction across the training pipeline " "(per-stage distribution)"
    )
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.set_axisbelow(True)

    handles = [
        Patch(
            facecolor=ALGO_STYLE[a]["color"],
            alpha=0.4,
            edgecolor=ALGO_STYLE[a]["color"],
            label=ALGO_STYLE[a]["label"],
        )
        for a in ALGOS
    ]
    ax.legend(handles=handles, title="Algorithm", fontsize=9, loc="upper right")

    _degenerate_footnote(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# Option 4 -- faint swarm of all runs + bold per-stage mean trajectory
# --------------------------------------------------------------------------- #
def plot_swarm_trajectory(data, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    n_alg = len(ALGOS)
    offsets = np.linspace(-0.16, 0.16, n_alg)
    rng = np.random.default_rng(1)

    for ai, algo in enumerate(ALGOS):
        color = ALGO_STYLE[algo]["color"]
        label = ALGO_STYLE[algo]["label"]
        means = []
        xs_mean = []
        for si, stage in enumerate(STAGES):
            vals = data[stage][algo]
            if vals.size == 0:
                means.append(np.nan)
                xs_mean.append(si + offsets[ai])
                continue
            x = si + offsets[ai]
            jx = x + (rng.random(vals.size) - 0.5) * 0.12
            ax.scatter(
                jx, vals, s=18, color=color, alpha=0.28, edgecolor="none", zorder=2
            )
            means.append(float(np.mean(vals)))
            xs_mean.append(x)
        ax.plot(
            xs_mean,
            means,
            "-",
            color=color,
            lw=2.4,
            zorder=4,
            marker="o",
            ms=8,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=label,
        )
        # annotate the mean value at each stage
        for x, m in zip(xs_mean, means):
            if np.isfinite(m):
                ax.annotate(
                    f"{m:.1f}",
                    (x, m),
                    textcoords="offset points",
                    xytext=(0, 9),
                    ha="center",
                    fontsize=7.5,
                    color=color,
                    fontweight="bold",
                    zorder=5,
                )

    _euro6(ax)
    ax.set_yscale("log")
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGES])
    ax.set_xlim(-0.45, len(STAGES) - 0.55)
    ax.set_ylabel("Total cycle NOx (g, log scale)")
    ax.set_title(
        "NOx reduction trajectory across the training pipeline "
        "(all runs + per-stage mean)"
    )
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.set_axisbelow(True)

    # caveat note on the Phase-1 spread (battery-depleting low-NOx outliers)
    ax.annotate(
        "Phase-1 low-NOx runs are battery-depleting\n(not charge-sustaining)",
        xy=(0.06, 2.7),
        xytext=(0.09, 0.02),
        textcoords="axes fraction",
        fontsize=8,
        color="dimgray",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8),
    )

    ax.legend(title="Algorithm (line = stage mean)", fontsize=9, loc="upper right")
    _degenerate_footnote(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--phase1_logs", default=os.path.join(here, "..", "logs_cluster_phase1", "logs")
    )
    p.add_argument(
        "--phase2_logs", default=os.path.join(here, "..", "logs_cluster_phase2", "logs")
    )
    p.add_argument(
        "--seeds_csv",
        default=os.path.join(
            here, "..", "logs_cluster_phase2", "phase2_seeds_results.csv"
        ),
    )
    p.add_argument(
        "--out_dir",
        default=os.path.join(here, "..", "logs_cluster_phase2", "analysis_plots"),
    )
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = collect(args.phase1_logs, args.phase2_logs, args.seeds_csv)

    print("Per-stage run counts and mean NOx (g):")
    for s in STAGES:
        parts = []
        for a in ALGOS:
            v = data[s][a]
            parts.append(
                f"{a.upper()} n={v.size} mean={v.mean():.2f}"
                if v.size
                else f"{a.upper()} n=0"
            )
        print(f"  {s:7s}: " + " | ".join(parts))

    plot_box(data, os.path.join(args.out_dir, "nox_pipeline_box.png"))
    plot_swarm_trajectory(
        data, os.path.join(args.out_dir, "nox_pipeline_swarm_traj.png")
    )


if __name__ == "__main__":
    main()
