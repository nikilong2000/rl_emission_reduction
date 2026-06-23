"""Phase-1 Pareto / behaviour visualisations.

Produces three thesis figures from a phase-1 result tree:

  A1  pareto_front.png        - speed RMSE vs total NOx, Pareto front, colour = dSOC.
  A3  parallel_coords.png     - per-seed polylines across the 6 evaluation metrics.
  B1  engine_map_occupancy.png- per-algo operating points overlaid on the real
                                EA189 BSFC engine map, to show how each policy
                                exploits (or avoids) the efficient low-BSFC zone.

Standalone: reads only the on-disk artefacts, no project imports. Reusable for
phase 2 by pointing --logs_dir at a different result tree.

Example:
    python plot_pareto_behaviour.py \
        --logs_dir ../logs_cluster_phase1/logs \
        --engine_map ../engine_map/191011_Kennfeld_EA189_neu.xlsx \
        --out_dir ../logs_cluster_phase1/analysis_plots
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ALGO_STYLE = {
    "ppo": {"label": "PPO", "marker": "o", "color": "#1f77b4"},
    "td3": {"label": "TD3", "marker": "^", "color": "#d62728"},
    "sac": {"label": "SAC", "marker": "s", "color": "#2ca02c"},
}

# Metrics shown in the parallel-coordinates plot. (key, axis label, lower_is_better)
PC_METRICS = [
    ("rmse_speed_kmph", "RMSE\n(km/h)", True),
    ("mae_speed_kmph", "MAE\n(km/h)", True),
    ("total_fuel_g", "Fuel\n(g)", True),
    ("total_nox_g", "NOx\n(g)", True),
    ("delta_soc", "ΔSOC", None),
    ("total_reward", "Reward", False),
]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_scalars(
    logs_dir: str, algos: list[str], seeds_subpath: str = "optuna/seeds"
) -> pd.DataFrame:
    """One row per seed with the scalar evaluation metrics."""
    rows = []
    for algo in algos:
        seed_glob = os.path.join(logs_dir, algo, seeds_subpath, "seed_*")
        for seed_dir in sorted(glob(seed_glob)):
            f = os.path.join(seed_dir, "evaluation_metrics.json")
            if not os.path.isfile(f):
                continue
            d = json.load(open(f))
            d["algo"] = algo
            d["seed"] = int(os.path.basename(seed_dir).split("_")[1])
            rows.append(d)
    if not rows:
        raise SystemExit(f"No evaluation_metrics.json found under {logs_dir}")
    return pd.DataFrame(rows)


def load_steps(
    logs_dir: str, algo: str, seeds_subpath: str = "optuna/seeds"
) -> pd.DataFrame:
    """Pooled per-step evaluation data across all seeds of one algorithm."""
    frames = []
    seed_glob = os.path.join(logs_dir, algo, seeds_subpath, "seed_*")
    for seed_dir in sorted(glob(seed_glob)):
        f = os.path.join(seed_dir, "evaluation_data.csv")
        if os.path.isfile(f):
            frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_bsfc(xlsx_path: str) -> pd.DataFrame:
    """Real EA189 BSFC measurement points (rpm, torque, BSFC g/kWh)."""
    df = pd.read_excel(xlsx_path, sheet_name="Zenon", header=0)
    cols = {
        "Drehzahl Bremse": "rpm",
        "Drehmoment Summe": "torque",
        "Spezifischer Verbrauch": "bsfc",
    }
    df = df[list(cols)].rename(columns=cols)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    # drop physically implausible / idle outliers below the mapped envelope
    df = df[(df["rpm"] > 0) & (df["torque"] > 0) & (df["bsfc"] > 0)]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# A1 - Pareto front
# --------------------------------------------------------------------------- #
def pareto_front_indices(x: np.ndarray, y: np.ndarray) -> list[int]:
    """Indices of the non-dominated set for minimising both x and y."""
    order = np.argsort(x)
    front, best_y = [], np.inf
    for i in order:
        if y[i] < best_y - 1e-9:
            front.append(i)
            best_y = y[i]
    return front


def plot_pareto(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))

    norm = plt.Normalize(
        vmin=-0.65, vmax=0.31
    )  # dSOC: deplete (blue) -> saturate (red)
    cmap = plt.cm.coolwarm

    for algo, style in ALGO_STYLE.items():
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        ax.scatter(
            sub["rmse_speed_kmph"],
            sub["total_nox_g"],
            marker=style["marker"],
            s=130,
            c=sub["delta_soc"],
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )

    # global Pareto front (minimise RMSE and NOx)
    x = df["rmse_speed_kmph"].to_numpy()
    y = df["total_nox_g"].to_numpy()
    front = pareto_front_indices(x, y)
    fx, fy = x[front], y[front]
    o = np.argsort(fx)
    ax.step(
        fx[o],
        fy[o],
        where="post",
        color="0.35",
        lw=1.6,
        ls="--",
        zorder=2,
        label="Pareto front",
    )

    # annotate best-RMSE seed of each algo with a leader line to its marker
    best_offsets = {  # (dx_pts, dy_pts, ha) chosen to escape the dense blob
        "ppo": (40, 24, "left"),
        "td3": (40, 0, "left"),
        "sac": (30, 36, "right"),
    }
    for algo in df["algo"].unique():
        sub = df[df["algo"] == algo]
        r = sub.loc[sub["rmse_speed_kmph"].idxmin()]
        dx, dy, ha = best_offsets.get(algo, (10, 10, "left"))
        ax.annotate(
            f"{ALGO_STYLE[algo]['label']} best\nseed {int(r.seed)}",
            (r.rmse_speed_kmph, r.total_nox_g),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
            ha=ha,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="0.4"),
        )
    rc = df.loc[df["total_nox_g"].idxmin()]
    ax.annotate(
        f"cleanest: {ALGO_STYLE[rc.algo]['label']} seed {int(rc.seed)}\n"
        f"{rc.total_nox_g:.1f} g, ΔSOC={rc.delta_soc:+.2f}",
        (rc.rmse_speed_kmph, rc.total_nox_g),
        textcoords="offset points",
        xytext=(22, 22),
        fontsize=8,
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="0.4"),
    )

    ax.set_yscale("log")
    ax.set_xlabel("Speed RMSE (km/h)")
    ax.set_ylabel("Total NOx (g, log scale)")
    ax.set_title("Phase-1 Pareto picture: speed accuracy vs NOx (30 seeds)")
    ax.grid(True, which="both", alpha=0.25)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label("ΔSOC  (blue = battery depleted, red = saturated to 1.0)")

    handles = [
        Line2D(
            [0],
            [0],
            marker=s["marker"],
            color="w",
            markerfacecolor="0.6",
            markeredgecolor="k",
            markersize=11,
            label=s["label"],
        )
        for s in ALGO_STYLE.values()
    ]
    handles.append(Line2D([0], [0], color="0.35", ls="--", label="Pareto front"))
    ax.legend(handles=handles, loc="upper right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# A3 - parallel coordinates
# --------------------------------------------------------------------------- #
def plot_parallel(df: pd.DataFrame, out_path: str) -> None:
    keys = [k for k, _, _ in PC_METRICS]
    labels = [lab for _, lab, _ in PC_METRICS]
    data = df[keys].to_numpy(dtype=float)

    lo = data.min(axis=0)
    hi = data.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    norm = (data - lo) / span  # 0..1 per axis

    xs = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot(
            xs,
            norm[i],
            color=ALGO_STYLE[row["algo"]]["color"],
            alpha=0.55,
            lw=1.3,
            zorder=2,
        )

    # axis ticks with real min/max so readers can decode the normalised lines
    for j, x in enumerate(xs):
        ax.axvline(x, color="0.8", lw=1.0, zorder=1)
        ax.text(
            x, 1.04, f"{hi[j]:.1f}", ha="center", va="bottom", fontsize=7, color="0.3"
        )
        ax.text(
            x, -0.04, f"{lo[j]:.1f}", ha="center", va="top", fontsize=7, color="0.3"
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.1, 1.12)
    ax.set_yticks([])
    ax.set_title(
        "Per-seed evaluation metrics (parallel coordinates, normalised per axis)"
    )

    handles = [
        Line2D([0], [0], color=s["color"], lw=2, label=s["label"])
        for s in ALGO_STYLE.values()
    ]
    ax.legend(handles=handles, loc="upper left", ncol=3, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# B1 - operating points over the real BSFC map
# --------------------------------------------------------------------------- #
def _density_contour(ax, xpts, ypts, extent, color):
    """Light 2D-histogram density contour (no scipy dependency)."""
    if len(xpts) < 50:
        return
    h, xe, ye = np.histogram2d(
        xpts,
        ypts,
        bins=40,
        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
    )
    h = h.T
    if h.max() <= 0:
        return
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    levels = [h.max() * f for f in (0.15, 0.5)]  # ~ outer / inner mass
    ax.contour(xc, yc, h, levels=levels, colors=[color], linewidths=1.6, zorder=5)


def plot_engine_map(logs_dir, algos, bsfc, out_path, seeds_subpath="optuna/seeds"):
    tri = mtri.Triangulation(bsfc["rpm"], bsfc["torque"])
    extent = [
        bsfc["rpm"].min(),
        bsfc["rpm"].max(),
        bsfc["torque"].min(),
        bsfc["torque"].max(),
    ]

    n = len(algos)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 6.0), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    # most efficient measured point (min BSFC) - the "sweet spot"
    sweet = bsfc.loc[bsfc["bsfc"].idxmin()]
    levels = np.linspace(bsfc["bsfc"].min(), min(bsfc["bsfc"].max(), 800), 16)
    cf = None

    for ax, algo in zip(axes, algos):
        cf = ax.tricontourf(
            tri, bsfc["bsfc"], levels=levels, cmap="viridis_r", alpha=0.92
        )
        cl = ax.tricontour(
            tri, bsfc["bsfc"], levels=8, colors="k", linewidths=0.4, alpha=0.45
        )
        ax.clabel(cl, inline=True, fontsize=6, fmt="%.0f")

        steps = load_steps(logs_dir, algo, seeds_subpath)
        stat = ""
        if not steps.empty:
            on = steps["engine_on"] > 0.5
            fired = steps[on & (steps["ice_torque"] > 0)]
            in_map = fired[
                fired["ice_speed_rpm"].between(extent[0], extent[1])
                & fired["ice_torque"].between(extent[2], extent[3])
            ]
            off_frac = 1.0 - on.mean()
            ax.scatter(
                in_map["ice_speed_rpm"],
                in_map["ice_torque"],
                s=5,
                color=ALGO_STYLE[algo]["color"],
                alpha=0.045,
                edgecolors="none",
                zorder=4,
            )
            _density_contour(
                ax,
                in_map["ice_speed_rpm"].to_numpy(),
                in_map["ice_torque"].to_numpy(),
                extent,
                ALGO_STYLE[algo]["color"],
            )
            mr, mt = in_map["ice_speed_rpm"].median(), in_map["ice_torque"].median()
            ax.scatter(
                [mr],
                [mt],
                s=190,
                marker="X",
                color=ALGO_STYLE[algo]["color"],
                edgecolors="white",
                linewidths=1.6,
                zorder=6,
            )
            stat = (
                f"median op: {mr:.0f} rpm, {mt:.0f} Nm\n"
                f"engine off: {off_frac*100:.0f}% of steps"
            )
            ax.text(
                0.03,
                0.03,
                stat,
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9),
                zorder=8,
            )

        ax.scatter(
            [sweet["rpm"]],
            [sweet["torque"]],
            marker="*",
            s=300,
            color="gold",
            edgecolors="k",
            linewidths=0.8,
            zorder=7,
        )
        ax.set_title(ALGO_STYLE[algo]["label"], fontsize=13, fontweight="bold")
        ax.set_xlabel("ICE speed (rpm)")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("ICE torque (Nm)")

    # reserve a right margin and give the colourbar its own axes there so it sits
    # beside the last panel (on the background), never on top of the BSFC map
    fig.subplots_adjust(left=0.05, right=0.89, top=0.86, bottom=0.10, wspace=0.08)
    cax = fig.add_axes([0.905, 0.10, 0.015, 0.76])
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label("BSFC (g/kWh)  — lower = more efficient")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="0.4",
            markeredgecolor="white",
            markersize=13,
            label="median operating point",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markeredgecolor="k",
            markersize=16,
            label="most efficient measured point",
        ),
        Line2D(
            [0],
            [0],
            color="0.4",
            lw=1.6,
            label="operating-point density (≥15% / ≥50% of peak)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.47, 0.965),
        framealpha=0.95,
        fontsize=9,
    )
    fig.suptitle(
        "How each policy exploits the EA189 BSFC efficiency map", y=0.995, fontsize=13
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--logs_dir", default=os.path.join(here, "..", "logs_cluster_phase1", "logs")
    )
    p.add_argument(
        "--engine_map",
        default=os.path.join(
            here, "..", "engine_map", "191011_Kennfeld_EA189_neu.xlsx"
        ),
    )
    p.add_argument(
        "--out_dir",
        default=os.path.join(here, "..", "logs_cluster_phase1", "analysis_plots"),
    )
    p.add_argument("--algos", nargs="+", default=["ppo", "td3", "sac"])
    p.add_argument(
        "--seeds_subpath",
        default="optuna/seeds",
        help="Subpath under <logs_dir>/<algo>/ containing seed_* dirs. "
        "Use 'phase2_seeds' for phase-2 validation runs.",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_scalars(args.logs_dir, args.algos, args.seeds_subpath)
    print(
        f"loaded {len(df)} seeds: "
        + ", ".join(f"{a}={ (df.algo==a).sum() }" for a in args.algos)
    )

    plot_pareto(df, os.path.join(args.out_dir, "pareto_front.png"))
    plot_parallel(df, os.path.join(args.out_dir, "parallel_coords.png"))

    bsfc = load_bsfc(args.engine_map)
    print(
        f"BSFC map: {len(bsfc)} pts, rpm[{bsfc.rpm.min():.0f},{bsfc.rpm.max():.0f}], "
        f"torque[{bsfc.torque.min():.0f},{bsfc.torque.max():.0f}], "
        f"bsfc[{bsfc.bsfc.min():.0f},{bsfc.bsfc.max():.0f}]"
    )
    plot_engine_map(
        args.logs_dir,
        args.algos,
        bsfc,
        os.path.join(args.out_dir, "engine_map_occupancy.png"),
        args.seeds_subpath,
    )


if __name__ == "__main__":
    main()
