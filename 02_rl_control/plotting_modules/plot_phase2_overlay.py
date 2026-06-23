"""
Phase-1 vs phase-2 overlay plots for thesis.

Produces three figures from both result trees:

  pareto_overlay.png         - speed RMSE vs NOx, both phases. Phase-1 gray
                               markers (open), phase-2 colored (filled).
                               Phase-1 + phase-2 Pareto fronts both shown.
  engine_map_overlay.png     - one panel per algo. BSFC contours + phase-1
                               operating-point density (gray) + phase-2
                               operating-point density (algo color). Shows
                               how the policy shifts on the engine map.
  eval_rmse_tracking_overlay.png - one panel per algo. Phase-1 mean ± std
                               speed-tracking band (gray) overlaid with
                               phase-2 mean ± std (algo color).

Usage:
    python plot_phase2_overlay.py \
        --phase1_logs 02_rl_control/logs_cluster_phase1/logs \
        --phase2_logs 02_rl_control/logs_cluster/logs \
        --engine_map  02_rl_control/engine_map/191011_Kennfeld_EA189_neu.xlsx \
        --out_dir     02_rl_control/logs_cluster/analysis_plots
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from plot_pareto_behaviour import (
    ALGO_STYLE,
    load_bsfc,
    load_scalars,
    load_steps,
    pareto_front_indices,
)

FIG_DPI = 180


# --------------------------------------------------------------------------- #
# Plot 1 — Pareto overlay
# --------------------------------------------------------------------------- #
def plot_pareto_overlay(df_p1: pd.DataFrame, df_p2: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Phase 1: white face + algo-color hatch lines (mpl 3.7+ scatter hatch)
    for algo, style in ALGO_STYLE.items():
        s1 = df_p1[df_p1["algo"] == algo]
        if not s1.empty:
            ax.scatter(
                s1["rmse_speed_kmph"],
                s1["total_nox_g"],
                marker=style["marker"],
                s=120,
                facecolors="white",
                edgecolors=style["color"],
                linewidths=1.2,
                hatch="////",
                alpha=0.9,
                zorder=2,
            )

    # Phase 2: solid fill with algo color
    for algo, style in ALGO_STYLE.items():
        s2 = df_p2[df_p2["algo"] == algo]
        if not s2.empty:
            ax.scatter(
                s2["rmse_speed_kmph"],
                s2["total_nox_g"],
                marker=style["marker"],
                s=130,
                c=style["color"],
                edgecolors="black",
                linewidths=0.7,
                alpha=0.9,
                zorder=4,
            )

    for label, df, color, ls in [
        ("Phase-1 front", df_p1, "0.5", ":"),
        ("Phase-2 front", df_p2, "0.15", "--"),
    ]:
        x = df["rmse_speed_kmph"].to_numpy()
        y = df["total_nox_g"].to_numpy()
        front = pareto_front_indices(x, y)
        fx, fy = x[front], y[front]
        o = np.argsort(fx)
        ax.step(
            fx[o],
            fy[o],
            where="post",
            color=color,
            lw=1.6,
            ls=ls,
            zorder=3,
            label=label,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Speed RMSE (km/h)")
    ax.set_ylabel("Total NOx (g, log scale)")
    ax.set_title("Phase-1 (hatched) vs Phase-2 (solid) — Pareto picture")
    ax.grid(True, which="both", alpha=0.25)

    algo_handles = [
        Line2D(
            [0],
            [0],
            marker=s["marker"],
            color="w",
            markerfacecolor=s["color"],
            markeredgecolor="black",
            markersize=10,
            label=s["label"],
        )
        for s in ALGO_STYLE.values()
    ]
    phase_handles = [
        Patch(
            facecolor="0.75",
            edgecolor="0.4",
            hatch="///",
            linewidth=0.8,
            label="Phase 1 (hatched)",
        ),
        Patch(
            facecolor="0.4", edgecolor="black", linewidth=0.8, label="Phase 2 (solid)"
        ),
        Line2D([0], [0], color="0.5", ls=":", lw=1.6, label="P1 Pareto front"),
        Line2D([0], [0], color="0.15", ls="--", lw=1.6, label="P2 Pareto front"),
    ]
    ax.legend(
        handles=algo_handles + phase_handles, loc="upper right", fontsize=8, ncol=2
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_pareto_overlay_soc(df_p1: pd.DataFrame, df_p2: pd.DataFrame, out_path: str):
    """Pareto overlay coloured by ΔSOC. Open markers = phase-1, filled = phase-2,
    marker shape = algorithm. Exposes phase-1's low-NOx seeds as battery-depleters
    (blue) vs the charge-sustaining phase-2 cloud (near-white)."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    norm = plt.Normalize(vmin=-0.65, vmax=0.31)
    cmap = plt.cm.coolwarm

    for algo, style in ALGO_STYLE.items():
        s1 = df_p1[df_p1["algo"] == algo]
        s2 = df_p2[df_p2["algo"] == algo]
        if not s1.empty:
            ax.scatter(
                s1["rmse_speed_kmph"],
                s1["total_nox_g"],
                marker=style["marker"],
                s=95,
                c=s1["delta_soc"],
                cmap=cmap,
                norm=norm,
                edgecolors="0.4",
                linewidths=1.6,
                alpha=0.55,
                zorder=2,
            )
        if not s2.empty:
            ax.scatter(
                s2["rmse_speed_kmph"],
                s2["total_nox_g"],
                marker=style["marker"],
                s=140,
                c=s2["delta_soc"],
                cmap=cmap,
                norm=norm,
                edgecolors="black",
                linewidths=1.4,
                alpha=0.95,
                zorder=4,
            )

    # charge-sustaining band annotation
    ax.axhspan(0, 0, color="none")
    ax.set_yscale("log")
    ax.set_xlabel("Speed RMSE (km/h)")
    ax.set_ylabel("Total NOx (g, log scale)")
    ax.set_title(
        "Pareto coloured by ΔSOC — Phase-1 (thin edge) vs Phase-2 (saturated edge)\n"
        "blue = battery depleted, white ≈ charge-sustaining, red = saturated"
    )
    ax.grid(True, which="both", alpha=0.25)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label("ΔSOC")

    handles = [
        Line2D(
            [0],
            [0],
            marker=s["marker"],
            color="w",
            markerfacecolor="0.6",
            markeredgecolor="black",
            markersize=10,
            label=s["label"],
        )
        for s in ALGO_STYLE.values()
    ] + [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="0.8",
            markeredgecolor="0.4",
            markersize=9,
            markeredgewidth=1.6,
            label="Phase-1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="0.5",
            markeredgecolor="black",
            markersize=11,
            markeredgewidth=1.4,
            label="Phase-2",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# Plot 2 — Engine map overlay (per algo)
# --------------------------------------------------------------------------- #
def _density_contour(ax, xpts, ypts, extent, color, alpha=0.6):
    """KDE-style density via 2D histogram + contour."""
    if len(xpts) < 10:
        return
    h, xe, ye = np.histogram2d(
        xpts,
        ypts,
        bins=[40, 30],
        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
    )
    if h.max() == 0:
        return
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc)
    levels = [0.05, 0.25, 0.5, 0.8]
    Z = h.T / h.max()
    ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        colors=[color],
        linewidths=[0.5, 0.9, 1.3, 1.6],
        alpha=alpha,
    )


def plot_engine_map_overlay(
    phase1_logs: str,
    phase2_logs: str,
    bsfc: pd.DataFrame,
    out_path: str,
    p1_subpath: str,
    p2_subpath: str,
    algos: list[str],
):
    tri = mtri.Triangulation(bsfc["rpm"], bsfc["torque"])
    extent = [bsfc.rpm.min(), bsfc.rpm.max(), bsfc.torque.min(), bsfc.torque.max()]

    fig, axes = plt.subplots(
        1, len(algos), figsize=(5.2 * len(algos), 5.5), sharey=True
    )
    if len(algos) == 1:
        axes = [axes]

    levels = np.linspace(bsfc["bsfc"].min(), min(bsfc["bsfc"].max(), 800), 16)
    sweet = bsfc.loc[bsfc["bsfc"].idxmin()]

    for ax, algo in zip(axes, algos):
        c = ax.tricontourf(
            tri, bsfc["bsfc"], levels=levels, cmap="viridis_r", alpha=0.85
        )
        ax.tricontour(
            tri, bsfc["bsfc"], levels=8, colors="k", linewidths=0.4, alpha=0.4
        )
        ax.scatter(
            [sweet["rpm"]],
            [sweet["torque"]],
            marker="*",
            s=160,
            c="white",
            edgecolors="black",
            lw=0.6,
            zorder=6,
        )

        s1 = load_steps(phase1_logs, algo, p1_subpath)
        s2 = load_steps(phase2_logs, algo, p2_subpath)

        if not s1.empty:
            mask1 = (s1["engine_on"] > 0.5) & (s1["ice_torque"] > 0)
            _density_contour(
                ax,
                s1.loc[mask1, "ice_speed_rpm"].values,
                s1.loc[mask1, "ice_torque"].values,
                extent,
                color="0.25",
                alpha=0.6,
            )
        if not s2.empty:
            mask2 = (s2["engine_on"] > 0.5) & (s2["ice_torque"] > 0)
            _density_contour(
                ax,
                s2.loc[mask2, "ice_speed_rpm"].values,
                s2.loc[mask2, "ice_torque"].values,
                extent,
                color=ALGO_STYLE[algo]["color"],
                alpha=0.95,
            )

        ax.set_title(f"{ALGO_STYLE[algo]['label']}  (P1 gray, P2 color)")
        ax.set_xlabel("Engine speed (rpm)")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Engine torque (Nm)")

    cb = fig.colorbar(c, ax=axes, shrink=0.85, pad=0.02, fraction=0.04)
    cb.set_label("BSFC (g/kWh)")

    fig.suptitle(
        "Operating-point density on EA189 BSFC map — Phase-1 vs Phase-2", fontsize=12
    )
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# Plot 3 — Eval RMSE tracking overlay (per algo)
# --------------------------------------------------------------------------- #
def _stack_speed(seed_dirs):
    rows_a, rows_t = [], []
    for sd in seed_dirs:
        f = os.path.join(sd, "evaluation_data.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        rows_a.append(d["speed_actual"].values)
        rows_t.append(d["speed_target"].values)
    if not rows_a:
        return None
    min_len = min(len(a) for a in rows_a)
    A = np.array([a[:min_len] for a in rows_a])
    T = np.array([t[:min_len] for t in rows_t])
    return A, T


def plot_tracking_overlay(
    phase1_logs: str,
    phase2_logs: str,
    out_path: str,
    p1_subpath: str,
    p2_subpath: str,
    algos: list[str],
):
    from glob import glob

    fig, axes = plt.subplots(len(algos), 1, figsize=(12, 3.4 * len(algos)), sharex=True)
    if len(algos) == 1:
        axes = [axes]
    dt = 0.5

    for ax, algo in zip(axes, algos):
        p1_seeds = sorted(glob(os.path.join(phase1_logs, algo, p1_subpath, "seed_*")))
        p2_seeds = sorted(glob(os.path.join(phase2_logs, algo, p2_subpath, "seed_*")))
        s1 = _stack_speed(p1_seeds)
        s2 = _stack_speed(p2_seeds)

        if s1 is not None:
            A1, T1 = s1
            err1 = np.abs(A1 - T1)
            t = np.arange(err1.shape[1]) * dt
            ax.plot(
                t,
                err1.mean(axis=0),
                color="0.45",
                lw=1.2,
                label=f"P1 mean (n={A1.shape[0]})",
            )
            ax.fill_between(
                t,
                np.maximum(err1.mean(0) - err1.std(0), 0),
                err1.mean(0) + err1.std(0),
                color="0.7",
                alpha=0.4,
                label="P1 ±1σ",
            )
        if s2 is not None:
            A2, T2 = s2
            err2 = np.abs(A2 - T2)
            t = np.arange(err2.shape[1]) * dt
            color = ALGO_STYLE[algo]["color"]
            ax.plot(
                t,
                err2.mean(axis=0),
                color=color,
                lw=1.4,
                label=f"P2 mean (n={A2.shape[0]})",
            )
            ax.fill_between(
                t,
                np.maximum(err2.mean(0) - err2.std(0), 0),
                err2.mean(0) + err2.std(0),
                color=color,
                alpha=0.25,
                label="P2 ±1σ",
            )

        ax.set_ylabel("|Speed err| (km/h)")
        ax.set_title(
            f"{ALGO_STYLE[algo]['label']} — Phase-1 vs Phase-2 speed-tracking error"
        )
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.set_ylim(0, None)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
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
    p.add_argument("--phase1_subpath", default="optuna/seeds")
    p.add_argument("--phase2_subpath", default="phase2_seeds")
    p.add_argument(
        "--engine_map",
        default=os.path.join(
            here, "..", "engine_map", "191011_Kennfeld_EA189_neu.xlsx"
        ),
    )
    p.add_argument(
        "--out_dir",
        default=os.path.join(here, "..", "logs_cluster_phase2", "analysis_plots"),
    )
    p.add_argument("--algos", nargs="+", default=["ppo", "sac", "td3"])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_p1 = load_scalars(args.phase1_logs, args.algos, args.phase1_subpath)
    df_p2 = load_scalars(args.phase2_logs, args.algos, args.phase2_subpath)
    print(f"P1 {len(df_p1)} seeds, P2 {len(df_p2)} seeds")

    plot_pareto_overlay(df_p1, df_p2, os.path.join(args.out_dir, "pareto_overlay.png"))
    plot_pareto_overlay_soc(
        df_p1, df_p2, os.path.join(args.out_dir, "pareto_overlay_soc.png")
    )

    bsfc = load_bsfc(args.engine_map)
    plot_engine_map_overlay(
        args.phase1_logs,
        args.phase2_logs,
        bsfc,
        os.path.join(args.out_dir, "engine_map_overlay.png"),
        args.phase1_subpath,
        args.phase2_subpath,
        args.algos,
    )

    plot_tracking_overlay(
        args.phase1_logs,
        args.phase2_logs,
        os.path.join(args.out_dir, "eval_rmse_tracking_overlay.png"),
        args.phase1_subpath,
        args.phase2_subpath,
        args.algos,
    )


if __name__ == "__main__":
    main()
