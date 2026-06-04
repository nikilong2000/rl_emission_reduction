"""
Phase-2 engine-map figures — consolidated, thesis-spec.

Produces per-algorithm (PPO, TD3, SAC) operating-point overlays on two
*measured* EA189 bench surfaces, both built by linear triangulation of the 74
steady-state measurement points (no empty histogram bins):

  engine_map_bsfc_overlay.png  - background = BSFC (g/kWh), fuel efficiency.
  engine_map_nox_overlay.png   - background = measured NOx mass flow mNOx (g/h),
                                 the rigorous emission lens. mNOx rises with
                                 load (corr(torque,mNOx)=0.84) and is
                                 anti-correlated with BSFC (corr=-0.47): the
                                 low-BSFC "sweet spot" is a high-NOx zone, so a
                                 NOx-minimising policy must move away from it.

Each panel overlays phase-1 (gray, dashed density contour + hollow median X)
and phase-2 (algo colour, solid density contour + filled median X).

Layout matches PHASE1_ANALYSIS engine map: figsize (6·n, 6) at dpi 200 →
3600×1200 for n=3. Order and colours follow ALGO_STYLE (PPO blue, TD3 red,
SAC green).

Usage:
    python plot_engine_maps_phase2.py
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from plot_pareto_behaviour import ALGO_STYLE, load_steps

# Canonical thesis order
ALGOS = ["ppo", "td3", "sac"]


def load_engine_map(xlsx_path: str) -> pd.DataFrame:
    """Measured EA189 bench points: rpm, torque, BSFC (g/kWh), mNOx (g/h)."""
    df = pd.read_excel(xlsx_path, sheet_name="Zenon", header=0)
    cols = {
        "Drehzahl Bremse": "rpm",
        "Drehmoment Summe": "torque",
        "Spezifischer Verbrauch": "bsfc",
        "mNOx": "mnox",
    }
    df = df[list(cols)].rename(columns=cols)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df[(df["rpm"] > 0) & (df["torque"] > 0) & (df["bsfc"] > 0)]
    return df.reset_index(drop=True)


def _density_contour(ax, xpts, ypts, extent, color, ls, lw=1.7, halo="black"):
    """2D-histogram density contour (outer ~15% / inner ~50% of peak) with a
    contrasting halo so it reads on any background colormap."""
    if len(xpts) < 50:
        return
    h, xe, ye = np.histogram2d(
        xpts, ypts, bins=40, range=[[extent[0], extent[1]], [extent[2], extent[3]]]
    )
    h = h.T
    if h.max() <= 0:
        return
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    levels = [h.max() * f for f in (0.15, 0.5)]
    cs = ax.contour(xc, yc, h, levels=levels, colors=[color], linewidths=lw,
                    linestyles=ls, zorder=5)
    cs.set(path_effects=[pe.withStroke(linewidth=lw + 1.6, foreground=halo)])


def _in_map(steps, extent):
    on = steps["engine_on"] > 0.5
    fired = steps[on & (steps["ice_torque"] > 0)]
    return fired[
        fired["ice_speed_rpm"].between(extent[0], extent[1])
        & fired["ice_torque"].between(extent[2], extent[3])
    ]


def plot_overlay(emap, quantity, label, cmap, out_path,
                 phase1_logs, phase2_logs, p1_sub, p2_sub,
                 mark_extreme="min"):
    tri = mtri.Triangulation(emap["rpm"], emap["torque"])
    extent = [emap["rpm"].min(), emap["rpm"].max(),
              emap["torque"].min(), emap["torque"].max()]
    z = emap[quantity].to_numpy()

    n = len(ALGOS)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.0), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    vmax = min(z.max(), np.percentile(z, 99))
    levels = np.linspace(z.min(), vmax, 16)
    cf = None

    # extreme background point (min BSFC = "sweet spot"; or min/max NOx)
    if mark_extreme == "min":
        ext_pt = emap.loc[emap[quantity].idxmin()]
        ext_lbl = f"min {label.split('(')[0].strip()}"
    else:
        ext_pt = emap.loc[emap[quantity].idxmax()]
        ext_lbl = f"max {label.split('(')[0].strip()}"

    for ax, algo in zip(axes, ALGOS):
        cf = ax.tricontourf(tri, z, levels=levels, cmap=cmap, alpha=0.92,
                            extend="max")
        cl = ax.tricontour(tri, z, levels=8, colors="k", linewidths=0.4, alpha=0.45)
        ax.clabel(cl, inline=True, fontsize=6, fmt="%.0f")

        color = ALGO_STYLE[algo]["color"]
        s1 = load_steps(phase1_logs, algo, p1_sub)
        s2 = load_steps(phase2_logs, algo, p2_sub)
        m1 = _in_map(s1, extent) if not s1.empty else pd.DataFrame()
        m2 = _in_map(s2, extent) if not s2.empty else pd.DataFrame()

        if not m1.empty:
            _density_contour(ax, m1["ice_speed_rpm"].to_numpy(),
                             m1["ice_torque"].to_numpy(), extent, "white", ":",
                             1.7, halo="black")
            ax.scatter([m1["ice_speed_rpm"].median()], [m1["ice_torque"].median()],
                       s=230, marker="X", color="white", edgecolors="black",
                       linewidths=2.0, zorder=6)
        if not m2.empty:
            _density_contour(ax, m2["ice_speed_rpm"].to_numpy(),
                             m2["ice_torque"].to_numpy(), extent, color, "-",
                             2.1, halo="white")
            ax.scatter([m2["ice_speed_rpm"].median()], [m2["ice_torque"].median()],
                       s=240, marker="X", color=color, edgecolors="white",
                       linewidths=1.8, zorder=7)

        # phase-2 mean op-point text
        if not m2.empty:
            txt = (f"P2 median: {m2['ice_speed_rpm'].median():.0f} rpm, "
                   f"{m2['ice_torque'].median():.0f} Nm")
            ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=8,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
                    zorder=8)

        ax.scatter([ext_pt["rpm"]], [ext_pt["torque"]], marker="*", s=300,
                   color="gold", edgecolors="k", linewidths=0.8, zorder=9)
        ax.set_title(ALGO_STYLE[algo]["label"], fontsize=13, fontweight="bold")
        ax.set_xlabel("ICE speed (rpm)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("ICE torque (Nm)")

    fig.subplots_adjust(left=0.05, right=0.89, top=0.86, bottom=0.10, wspace=0.08)
    cax = fig.add_axes([0.905, 0.10, 0.015, 0.76])
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label(label)

    p1_line = Line2D([0], [0], color="white", ls=":", lw=1.8,
                     label="Phase-1 density (15/50%)",
                     path_effects=[pe.withStroke(linewidth=3.4, foreground="black")])
    p2_line = Line2D([0], [0], color="0.3", ls="-", lw=2.0,
                     label="Phase-2 density (15/50%, algo colour)",
                     path_effects=[pe.withStroke(linewidth=3.6, foreground="white")])
    handles = [
        Line2D([0], [0], marker="X", color="w", markerfacecolor="white",
               markeredgecolor="black", markeredgewidth=1.8, markersize=12,
               label="Phase-1 median op-point"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="0.4",
               markeredgecolor="white", markersize=13, label="Phase-2 median op-point"),
        p1_line,
        p2_line,
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="k", markersize=15, label=ext_lbl),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.47, 0.965), framealpha=0.95, fontsize=9)
    fig.suptitle(
        f"Phase-1 vs Phase-2 operating points on the measured EA189 {label.split('(')[0].strip()} map",
        y=0.995, fontsize=13,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}  ({6.0*n*200:.0f}x{6.0*200:.0f})")


def plot_single(emap, quantity, label, cmap, out_path, logs, sub,
                phase_label, mark_extreme="max"):
    """Single-phase engine-map occupancy: per-algo density + median X on the
    measured background. Mirrors plot_overlay but one phase only."""
    tri = mtri.Triangulation(emap["rpm"], emap["torque"])
    extent = [emap["rpm"].min(), emap["rpm"].max(),
              emap["torque"].min(), emap["torque"].max()]
    z = emap[quantity].to_numpy()
    n = len(ALGOS)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.0), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    vmax = min(z.max(), np.percentile(z, 99))
    levels = np.linspace(z.min(), vmax, 16)
    cf = None
    ext_pt = (emap.loc[emap[quantity].idxmax()] if mark_extreme == "max"
              else emap.loc[emap[quantity].idxmin()])
    ext_lbl = f"{'max' if mark_extreme=='max' else 'min'} {label.split('(')[0].strip()}"

    for ax, algo in zip(axes, ALGOS):
        cf = ax.tricontourf(tri, z, levels=levels, cmap=cmap, alpha=0.92, extend="max")
        cl = ax.tricontour(tri, z, levels=8, colors="k", linewidths=0.4, alpha=0.45)
        ax.clabel(cl, inline=True, fontsize=6, fmt="%.0f")
        color = ALGO_STYLE[algo]["color"]
        s = load_steps(logs, algo, sub)
        m = _in_map(s, extent) if not s.empty else pd.DataFrame()
        if not m.empty:
            _density_contour(ax, m["ice_speed_rpm"].to_numpy(),
                             m["ice_torque"].to_numpy(), extent, color, "-",
                             2.1, halo="white")
            ax.scatter([m["ice_speed_rpm"].median()], [m["ice_torque"].median()],
                       s=240, marker="X", color=color, edgecolors="white",
                       linewidths=1.8, zorder=7)
            txt = (f"median: {m['ice_speed_rpm'].median():.0f} rpm, "
                   f"{m['ice_torque'].median():.0f} Nm")
            ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=8,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
                    zorder=8)
        ax.scatter([ext_pt["rpm"]], [ext_pt["torque"]], marker="*", s=300,
                   color="gold", edgecolors="k", linewidths=0.8, zorder=9)
        ax.set_title(ALGO_STYLE[algo]["label"], fontsize=13, fontweight="bold")
        ax.set_xlabel("ICE speed (rpm)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("ICE torque (Nm)")

    fig.subplots_adjust(left=0.05, right=0.89, top=0.86, bottom=0.10, wspace=0.08)
    cax = fig.add_axes([0.905, 0.10, 0.015, 0.76])
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label(label)
    handles = [
        Line2D([0], [0], marker="X", color="w", markerfacecolor="0.4",
               markeredgecolor="white", markersize=13, label=f"{phase_label} median op-point"),
        Line2D([0], [0], color="0.3", ls="-", lw=2.0,
               label=f"{phase_label} density (15/50%, algo colour)",
               path_effects=[pe.withStroke(linewidth=3.6, foreground="white")]),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="k", markersize=15, label=ext_lbl),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.47, 0.965), framealpha=0.95, fontsize=9)
    fig.suptitle(
        f"{phase_label} operating points on the measured EA189 {label.split('(')[0].strip()} map",
        y=0.995, fontsize=13)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}  ({6.0*n*200:.0f}x{6.0*200:.0f})")


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
    p.add_argument("--engine_map",
                   default=os.path.join(here, "..", "engine_map",
                                        "191011_Kennfeld_EA189_neu.xlsx"))
    p.add_argument("--out_dir",
                   default=os.path.join(here, "..", "logs_cluster_phase2", "analysis_plots"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    emap = load_engine_map(args.engine_map)
    print(f"engine map: {len(emap)} pts; "
          f"BSFC[{emap.bsfc.min():.0f},{emap.bsfc.max():.0f}] g/kWh; "
          f"mNOx[{emap.mnox.min():.1f},{emap.mnox.max():.0f}] g/h")

    plot_overlay(emap, "bsfc", "BSFC (g/kWh) — lower = more efficient",
                 "viridis_r", os.path.join(args.out_dir, "engine_map_bsfc_overlay.png"),
                 args.phase1_logs, args.phase2_logs,
                 args.phase1_subpath, args.phase2_subpath, mark_extreme="min")

    plot_overlay(emap, "mnox", "NOx mass flow mNOx (g/h)",
                 "inferno", os.path.join(args.out_dir, "engine_map_nox_overlay.png"),
                 args.phase1_logs, args.phase2_logs,
                 args.phase1_subpath, args.phase2_subpath, mark_extreme="max")

    # Standalone single-phase NOx maps (pair the existing BSFC occupancy plots).
    p1_out = os.path.join(here, "..", "logs_cluster_phase1", "analysis_plots")
    os.makedirs(p1_out, exist_ok=True)
    plot_single(emap, "mnox", "NOx mass flow mNOx (g/h)", "inferno",
                os.path.join(p1_out, "engine_map_nox_occupancy.png"),
                args.phase1_logs, args.phase1_subpath, "Phase-1", mark_extreme="max")
    plot_single(emap, "mnox", "NOx mass flow mNOx (g/h)", "inferno",
                os.path.join(args.out_dir, "engine_map_nox_occupancy.png"),
                args.phase2_logs, args.phase2_subpath, "Phase-2", mark_extreme="max")


if __name__ == "__main__":
    main()
