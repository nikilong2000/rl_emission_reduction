#!/usr/bin/env python3
"""
Compare phase-2 RL agents (PPO, SAC) against the rule-based controller on the
real-world WLTC drive cycle (zero-shot for the RL agents, which were trained on
the staircase target schedule).

Reads:
  - RL:         logs_cluster_phase2/logs/<algo>/phase2_seeds/seed_*/eval_<cycle>/
  - Rule-based: rule_based_results/_adapted/eval_wltc/   (isoSOC, charge-sustaining)

Writes to plots/wltc_comparison/:
  - summary_table_<cycle>.{csv,md}   per-algo mean±std + rule-based
  - speed_tracking_<cycle>.png       target + RL mean±std band + rule-based line
  - pareto_<cycle>.png               speed RMSE vs NOx (mg/km), RL seeds + rule-based
  - phase_nox_<cycle>.png            WLTC phase-wise NOx (mg/km) grouped bars

WLTC_high is RL-only (no rule-based baseline exists for the 2400 s cycle).

Run AFTER eval_wltc_baseline.py and load_rule_based.py.
"""

import os
import sys
import json
import argparse
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, RL_DIR)

from plot_pareto_behaviour import ALGO_STYLE, pareto_front_indices  # noqa: E402

DT = 0.5
LOGS = os.path.join(RL_DIR, "logs_cluster_phase2", "logs")
RB_ADAPTED = os.path.join(RL_DIR, "rule_based_results", "_adapted", "eval_wltc")
OUT_DIR = os.path.join(RL_DIR, "plots", "wltc_comparison")
ALGOS = ["ppo", "sac"]
RB_STYLE = {"label": "Rule-based", "marker": "D", "color": "#F200FF"}

# WLTC phase durations (s) — mirrors utils/evaluation_utils.calculate_emissions_per_km
PHASES = [("low", 589), ("medium", 433), ("high", 455), ("extra_high", 323)]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_rl_metrics(cycle, algos=ALGOS):
    rows = []
    for algo in algos:
        pat = os.path.join(
            LOGS,
            algo,
            "phase2_seeds",
            "seed_*",
            f"eval_{cycle}",
            "evaluation_metrics.json",
        )
        for f in sorted(glob(pat)):
            d = json.load(open(f))
            d["algo"] = algo
            # .../seed_<n>/eval_<cycle>/evaluation_metrics.json
            d["seed"] = int(
                os.path.basename(os.path.dirname(os.path.dirname(f))).split("_")[1]
            )
            rows.append(d)
    return pd.DataFrame(rows)


def load_rl_steps(cycle, algo):
    """Return list of (seed, DataFrame) for each seed of one algo/cycle."""
    out = []
    pat = os.path.join(
        LOGS, algo, "phase2_seeds", "seed_*", f"eval_{cycle}", "evaluation_data.csv"
    )
    for f in sorted(glob(pat)):
        seed = int(os.path.basename(os.path.dirname(os.path.dirname(f))).split("_")[1])
        out.append((seed, pd.read_csv(f)))
    return out


def load_rule_based():
    mf = os.path.join(RB_ADAPTED, "evaluation_metrics.json")
    df = os.path.join(RB_ADAPTED, "evaluation_data.csv")
    if not (os.path.isfile(mf) and os.path.isfile(df)):
        return None, None
    return json.load(open(mf)), pd.read_csv(df)


# --------------------------------------------------------------------------- #
# Phase-wise NOx
# --------------------------------------------------------------------------- #
def phase_nox_mgkm(speed_actual, nox_gs):
    speed_actual = np.asarray(speed_actual, float)
    nox_gs = np.asarray(nox_gs, float)
    n = min(len(speed_actual), len(nox_gs))
    dist_km = speed_actual[:n] * DT / 3600.0
    nox_mg = nox_gs[:n] * DT * 1000.0
    res, start = {}, 0
    for name, dur in PHASES:
        steps = int(round(dur / DT))
        end = min(start + steps, n)
        d = float(np.sum(dist_km[start:end]))
        m = float(np.sum(nox_mg[start:end]))
        res[name] = m / d if d > 1e-9 else float("nan")
        start = end
    return res


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
TABLE_COLS = [
    ("total_nox_g", "NOx (g)", "{:.2f}"),
    ("nox_mg_per_km", "NOx (mg/km)", "{:.1f}"),
    ("total_fuel_burned_g", "Fuel burned (g)", "{:.0f}"),
    ("rmse_speed_kmph", "RMSE (km/h)", "{:.2f}"),
    ("mae_speed_kmph", "MAE (km/h)", "{:.2f}"),
    ("delta_soc", "ΔSOC", "{:+.3f}"),
    ("engine_off_pct", "Engine off (%)", "{:.1f}"),
    ("total_distance_km", "Distance (km)", "{:.2f}"),
]


def _augment(df):
    df = df.copy()
    df["nox_mg_per_km"] = df["nox_g_per_km"] * 1000.0
    return df


def build_summary(rl, rb_metrics, cycle):
    rl = _augment(rl)
    rows = []
    for algo in ALGOS:
        sub = rl[rl["algo"] == algo]
        if sub.empty:
            continue
        row = {"controller": ALGO_STYLE[algo]["label"], "n": len(sub)}
        for key, _, fmt in TABLE_COLS:
            row[key + "_mean"] = sub[key].mean()
            row[key + "_std"] = sub[key].std()
        rows.append(row)
    if rb_metrics is not None and cycle == "wltc":
        rb = dict(rb_metrics)
        rb["nox_mg_per_km"] = rb["nox_g_per_km"] * 1000.0
        row = {"controller": RB_STYLE["label"], "n": 1}
        for key, _, fmt in TABLE_COLS:
            row[key + "_mean"] = rb.get(key, float("nan"))
            row[key + "_std"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(summary, cycle):
    csv_path = os.path.join(OUT_DIR, f"summary_table_{cycle}.csv")
    summary.to_csv(csv_path, index=False)

    md = [
        f"# WLTC comparison — `{cycle}`\n",
        "| Controller (n) | " + " | ".join(lab for _, lab, _ in TABLE_COLS) + " |",
        "|" + "---|" * (len(TABLE_COLS) + 1),
    ]
    for _, r in summary.iterrows():
        cells = [f"{r['controller']} ({int(r['n'])})"]
        for key, _, fmt in TABLE_COLS:
            mean = r[key + "_mean"]
            std = r[key + "_std"]
            if np.isnan(std) or r["n"] == 1:
                cells.append(fmt.format(mean))
            else:
                cells.append(f"{fmt.format(mean)} ± {fmt.format(std).lstrip('+')}")
        md.append("| " + " | ".join(cells) + " |")
    md_path = os.path.join(OUT_DIR, f"summary_table_{cycle}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {csv_path}\nwrote {md_path}")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_speed_tracking(cycle, rb_steps):
    fig, ax = plt.subplots(figsize=(12, 5))
    target_drawn = False
    for algo in ALGOS:
        runs = load_rl_steps(cycle, algo)
        if not runs:
            continue
        T = min(len(df) for _, df in runs)
        stack = np.vstack([df["speed_actual"].to_numpy()[:T] for _, df in runs])
        t = np.arange(T) * DT
        if not target_drawn:
            tgt = runs[0][1]["speed_target"].to_numpy()[:T]
            ax.plot(
                t, tgt, color="black", lw=1.0, alpha=0.5, label="WLTC target", zorder=1
            )
            target_drawn = True
        mean, std = stack.mean(0), stack.std(0)
        c = ALGO_STYLE[algo]["color"]
        ax.plot(
            t,
            mean,
            color=c,
            lw=1.3,
            label=f"{ALGO_STYLE[algo]['label']} (mean, n={len(runs)})",
            zorder=3,
        )
        ax.fill_between(t, mean - std, mean + std, color=c, alpha=0.20, zorder=2)
    if rb_steps is not None and cycle == "wltc":
        T = len(rb_steps)
        ax.plot(
            np.arange(T) * DT,
            rb_steps["speed_actual"].to_numpy(),
            color=RB_STYLE["color"],
            lw=1.1,
            ls="--",
            label=RB_STYLE["label"],
            zorder=3,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(
        f"Speed tracking on {cycle.upper()} — RL (mean ± 1σ across seeds) vs rule-based"
    )
    ax.legend(loc="upper right", ncol=2, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"speed_tracking_{cycle}.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"wrote {p}")


def plot_pareto(cycle, rl, rb_metrics):
    rl = _augment(rl)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for algo in ALGOS:
        sub = rl[rl["algo"] == algo]
        if sub.empty:
            continue
        s = ALGO_STYLE[algo]
        ax.scatter(
            sub["rmse_speed_kmph"],
            sub["nox_mg_per_km"],
            marker=s["marker"],
            s=110,
            color=s["color"],
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
            zorder=3,
            label=f"{s['label']} (seeds)",
        )
    if rb_metrics is not None and cycle == "wltc":
        ax.scatter(
            [rb_metrics["rmse_speed_kmph"]],
            [rb_metrics["nox_g_per_km"] * 1000.0],
            marker=RB_STYLE["marker"],
            s=140,
            color=RB_STYLE["color"],
            edgecolors="black",
            linewidths=1.4,
            zorder=5,
            label=RB_STYLE["label"],
        )
    # combined Pareto front (minimise RMSE and NOx mg/km)
    x = rl["rmse_speed_kmph"].to_numpy()
    y = rl["nox_mg_per_km"].to_numpy()
    if rb_metrics is not None and cycle == "wltc":
        x = np.append(x, rb_metrics["rmse_speed_kmph"])
        y = np.append(y, rb_metrics["nox_g_per_km"] * 1000.0)
    front = pareto_front_indices(x, y)
    o = np.argsort(x[front])
    ax.step(
        x[front][o],
        y[front][o],
        where="post",
        color="0.35",
        lw=1.5,
        ls="--",
        zorder=2,
        label="Pareto front",
    )
    ax.axhline(80, color="crimson", lw=1.2, ls=":", zorder=1, label="Euro-6 (80 mg/km)")
    ax.set_xlabel("Speed RMSE (km/h)")
    ax.set_ylabel("NOx (mg/km)")
    ax.set_title(f"Speed accuracy vs NOx on {cycle.upper()}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"pareto_{cycle}.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"wrote {p}")


def plot_phase_nox(cycle, rb_steps):
    phase_names = [n for n, _ in PHASES]
    series = {}  # label -> (mean_array, std_array, color)
    for algo in ALGOS:
        runs = load_rl_steps(cycle, algo)
        if not runs:
            continue
        per_seed = [phase_nox_mgkm(df["speed_actual"], df["nox"]) for _, df in runs]
        arr = np.array([[d[n] for n in phase_names] for d in per_seed])
        series[ALGO_STYLE[algo]["label"]] = (
            arr.mean(0),
            arr.std(0),
            ALGO_STYLE[algo]["color"],
        )
    if rb_steps is not None and cycle == "wltc":
        d = phase_nox_mgkm(rb_steps["speed_actual"], rb_steps["nox"])
        series[RB_STYLE["label"]] = (
            np.array([d[n] for n in phase_names]),
            np.zeros(len(phase_names)),
            RB_STYLE["color"],
        )
    if not series:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(phase_names))
    w = 0.8 / len(series)
    for i, (lab, (mean, std, c)) in enumerate(series.items()):
        ax.bar(
            x + i * w,
            mean,
            w,
            yerr=std,
            label=lab,
            color=c,
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )
    ax.axhline(80, color="crimson", lw=1.2, ls=":", label="Euro-6 (80 mg/km)")
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([n.replace("_", "\n") for n in phase_names])
    ax.set_ylabel("NOx (mg/km)")
    ax.set_title(f"WLTC phase-wise NOx on {cycle.upper()}")
    ax.legend(framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, f"phase_nox_{cycle}.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(f"wrote {p}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycles", nargs="+", default=["wltc", "wltc_high"])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    rb_metrics, rb_steps = load_rule_based()
    if rb_metrics is None:
        print(
            "WARNING: no adapted rule-based data found; run load_rule_based.py first."
        )

    for cycle in args.cycles:
        rl = load_rl_metrics(cycle)
        if rl.empty:
            print(f"No RL runs found for cycle={cycle}; skipping.")
            continue
        print(
            f"\n=== {cycle}: loaded {len(rl)} RL seeds "
            + ", ".join(f"{a}={(rl.algo==a).sum()}" for a in ALGOS)
            + " ==="
        )
        summary = build_summary(rl, rb_metrics, cycle)
        write_summary(summary, cycle)
        plot_speed_tracking(cycle, rb_steps)
        plot_pareto(cycle, rl, rb_metrics)
        plot_phase_nox(cycle, rb_steps)


if __name__ == "__main__":
    main()
