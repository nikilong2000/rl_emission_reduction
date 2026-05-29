"""
Engine-map BSFC plot showing ONLY the best phase-2 seed per algorithm.

Best seed is picked by the same composite score Optuna minimised when choosing
the (W_EMISSION, W_SOC_SQUARED) weights:

    score = total_nox_g
          + λ_rmse · max(0, rmse_speed_kmph − 5)²        λ_rmse = 20
          + λ_soc  · max(0, max_abs_soc_drift − 0.05)²   λ_soc  = 1000

Three-component scalar: NOx (always counts), RMSE penalty (only kicks in above
5 km/h), SOC penalty (only kicks in above 0.05 max drift). The seed with the
minimum score per algorithm is plotted.

Usage:
    python plot_best_seed_engine_map.py
    python plot_best_seed_engine_map.py --lambda_rmse 50 --lambda_soc 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from plot_pareto_behaviour import ALGO_STYLE, load_bsfc

FIG_DPI = 180


def _density_contour(ax, xpts, ypts, extent, color, alpha=0.85, lw_base=0.7):
    """KDE-style density via 2D histogram + contour."""
    if len(xpts) < 10:
        ax.scatter(xpts, ypts, s=14, c=color, alpha=alpha, edgecolors="none")
        return
    h, xe, ye = np.histogram2d(
        xpts, ypts, bins=[40, 30],
        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
    )
    if h.max() == 0:
        return
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc)
    levels = [0.05, 0.25, 0.5, 0.8]
    Z = h.T / h.max()
    ax.contour(X, Y, Z, levels=levels, colors=[color],
               linewidths=[lw_base, lw_base + 0.4, lw_base + 0.8, lw_base + 1.2],
               alpha=alpha)


def best_seed_per_algo(logs_dir, algos, lambda_rmse, lambda_soc):
    """Compute composite score for every seed_*; return dict algo -> winner row."""
    winners = {}
    for algo in algos:
        rows = []
        for sd in sorted(glob(os.path.join(logs_dir, algo, "phase2_seeds", "seed_*"))):
            m_path = os.path.join(sd, "evaluation_metrics.json")
            if not os.path.exists(m_path):
                continue
            m = json.load(open(m_path))
            rmse = m["rmse_speed_kmph"]
            nox = m["total_nox_g"]
            ma = m["max_abs_soc_drift"]
            rmse_pen = lambda_rmse * max(0.0, rmse - 5.0) ** 2
            soc_pen = lambda_soc * max(0.0, ma - 0.05) ** 2
            score = nox + rmse_pen + soc_pen
            seed = int(os.path.basename(sd).replace("seed_", ""))
            rows.append({
                "algo": algo, "seed": seed, "seed_dir": sd,
                "rmse_speed_kmph": rmse, "total_nox_g": nox,
                "max_abs_soc_drift": ma,
                "rmse_pen": rmse_pen, "soc_pen": soc_pen, "score": score,
            })
        if not rows:
            print(f"WARN no seeds for {algo}")
            continue
        rows.sort(key=lambda r: r["score"])
        winners[algo] = rows[0]
        print(
            f"{algo.upper()} best: seed {rows[0]['seed']}  score={rows[0]['score']:.4f} "
            f"(rmse={rows[0]['rmse_speed_kmph']:.3f}, nox={rows[0]['total_nox_g']:.3f}g, "
            f"max|dSOC|={rows[0]['max_abs_soc_drift']:.4f})"
        )
    return winners


def plot_engine_map_best(winners, bsfc, out_path):
    tri = mtri.Triangulation(bsfc["rpm"], bsfc["torque"])
    extent = [bsfc.rpm.min(), bsfc.rpm.max(), bsfc.torque.min(), bsfc.torque.max()]
    fig, ax = plt.subplots(figsize=(11, 6.5))

    levels = np.linspace(bsfc["bsfc"].min(), min(bsfc["bsfc"].max(), 800), 16)
    c = ax.tricontourf(tri, bsfc["bsfc"], levels=levels, cmap="viridis_r", alpha=0.85)
    ax.tricontour(tri, bsfc["bsfc"], levels=8, colors="k", linewidths=0.4, alpha=0.4)

    sweet = bsfc.loc[bsfc["bsfc"].idxmin()]
    ax.scatter([sweet["rpm"]], [sweet["torque"]], marker="*", s=240,
               c="white", edgecolors="black", lw=0.8, zorder=6,
               label=f"BSFC min ({sweet['bsfc']:.0f} g/kWh)")

    for algo, w in winners.items():
        ed_path = os.path.join(w["seed_dir"], "evaluation_data.csv")
        if not os.path.exists(ed_path):
            print(f"WARN missing {ed_path}")
            continue
        d = pd.read_csv(ed_path)
        mask = (d["engine_on"] > 0.5) & (d["ice_torque"] > 0)
        _density_contour(
            ax,
            d.loc[mask, "ice_speed_rpm"].values,
            d.loc[mask, "ice_torque"].values,
            extent,
            color=ALGO_STYLE[algo]["color"],
            alpha=0.95,
        )
        ax.scatter(
            [], [],
            marker=ALGO_STYLE[algo]["marker"],
            s=130,
            c=ALGO_STYLE[algo]["color"],
            edgecolors="black",
            label=f"{ALGO_STYLE[algo]['label']} seed {w['seed']}  "
                  f"(score={w['score']:.2f}, NOx={w['total_nox_g']:.2f} g, "
                  f"RMSE={w['rmse_speed_kmph']:.2f} km/h)",
        )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Engine speed (rpm)")
    ax.set_ylabel("Engine torque (Nm)")
    ax.set_title(
        "Best phase-2 seed per algorithm — operating-point density on EA189 BSFC map\n"
        "(score = NOx_g + 20·max(0, RMSE−5)² + 1000·max(0, max|ΔSOC|−0.05)²)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    cb = fig.colorbar(c, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("BSFC (g/kWh)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs_dir",
                   default=os.path.join(here, "..", "logs_cluster", "logs"))
    p.add_argument("--engine_map",
                   default=os.path.join(here, "..", "engine_map",
                                        "191011_Kennfeld_EA189_neu.xlsx"))
    p.add_argument("--out_dir",
                   default=os.path.join(here, "..", "logs_cluster", "analysis_plots"))
    p.add_argument("--algos", nargs="+", default=["ppo", "sac", "td3"])
    p.add_argument("--lambda_rmse", type=float, default=20.0)
    p.add_argument("--lambda_soc", type=float, default=1000.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Scoring with λ_rmse={args.lambda_rmse}, λ_soc={args.lambda_soc}\n")
    winners = best_seed_per_algo(
        args.logs_dir, args.algos, args.lambda_rmse, args.lambda_soc
    )

    bsfc = load_bsfc(args.engine_map)
    out_path = os.path.join(args.out_dir, "engine_map_best_seed.png")
    plot_engine_map_best(winners, bsfc, out_path)


if __name__ == "__main__":
    main()
