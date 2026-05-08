"""
Plot SB3 training curves for many runs in one interactive figure.

A "run" is any directory containing one or more `*.monitor.csv` files
(SB3 Monitor wrapper output). The script walks a target directory,
discovers runs, loads their monitor CSVs, and overlays reward curves on
a single interactive matplotlib figure (zoom/pan + screenshot ready).

Examples
--------
# All trials of PPO phase 1 optuna sweep
python plot_training_curves.py logs_cluster_phase1/logs/ppo/optuna

# Only trials 050..069
python plot_training_curves.py logs_cluster_phase1/logs/ppo/optuna \
    --include 'trial_05*' 'trial_06*'

# Drop a noisy run
python plot_training_curves.py logs_cluster_phase1/logs/ppo/optuna/seeds \
    --exclude seed_3

# Smoothing window of 50 episodes
python plot_training_curves.py logs_cluster_phase1/logs/ppo/optuna --smooth 50
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------- discovery --------------------------------------


@dataclass
class Run:
    name: str
    path: str
    df: pd.DataFrame  # columns: timesteps, episode, r


def _read_monitor(path: str) -> pd.DataFrame:
    # SB3 monitor: first line is JSON header starting with '#'
    return pd.read_csv(path, skiprows=1)


def _load_run(path: str, name: str) -> Run | None:
    csvs = sorted(
        f for f in os.listdir(path) if f.endswith(".monitor.csv")
    )
    if not csvs:
        return None

    frames = []
    for csv in csvs:
        try:
            df = _read_monitor(os.path.join(path, csv))
        except Exception as e:
            print(f"  ! skip {csv}: {e}", file=sys.stderr)
            continue
        if df.empty or not {"r", "l", "t"}.issubset(df.columns):
            continue
        df = df[["r", "l", "t"]].copy()
        df["env_id"] = csv.split(".")[0]
        frames.append(df)

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True).sort_values("t", kind="stable")
    merged = merged.reset_index(drop=True)
    merged["timesteps"] = merged["l"].cumsum()
    merged["episode"] = np.arange(1, len(merged) + 1)
    return Run(name=name, path=path, df=merged)


def discover_runs(root: str) -> list[Run]:
    """Find every subdirectory under `root` that holds monitor CSVs."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Not a directory: {root}")

    # If the root itself contains monitor CSVs, treat it as a single run.
    self_csvs = [f for f in os.listdir(root) if f.endswith(".monitor.csv")]
    if self_csvs:
        run = _load_run(root, os.path.basename(os.path.normpath(root)))
        return [run] if run else []

    runs: list[Run] = []
    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        run = _load_run(sub, entry)
        if run:
            runs.append(run)
    return runs


# ----------------------------- filtering --------------------------------------


def _match_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) or p in name for p in patterns)


def filter_runs(
    runs: list[Run],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[Run]:
    out = runs
    if include:
        out = [r for r in out if _match_any(r.name, include)]
    if exclude:
        out = [r for r in out if not _match_any(r.name, exclude)]
    return out


# ----------------------------- smoothing --------------------------------------


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < 2:
        return y
    s = pd.Series(y)
    return s.rolling(window=window, min_periods=1, center=False).mean().to_numpy()


# ----------------------------- plotting ---------------------------------------


def _color_cycle(n: int):
    if n <= 10:
        cmap = plt.get_cmap("tab10")
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("turbo")
    return [cmap(i / max(n - 1, 1)) for i in range(n)] if n > 20 else [
        cmap(i % cmap.N) for i in range(n)
    ]


def plot(
    runs: list[Run],
    *,
    xaxis: str = "timesteps",
    smooth_window: int = 1,
    show_raw: bool = True,
    title: str | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (12.0, 6.5),
) -> None:
    if not runs:
        print("No runs to plot.", file=sys.stderr)
        return

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors = _color_cycle(len(runs))

    xkey = "timesteps" if xaxis == "timesteps" else "episode"
    xlabel = "Total Environment Steps" if xaxis == "timesteps" else "Episode"

    for run, color in zip(runs, colors):
        x = run.df[xkey].to_numpy()
        y = run.df["r"].to_numpy()

        if show_raw and smooth_window > 1:
            ax.plot(x, y, color=color, alpha=0.18, linewidth=0.8)
        ys = smooth(y, smooth_window)
        ax.plot(x, ys, color=color, linewidth=1.6, label=run.name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Episode Reward")
    ax.set_title(title or "Training Curves")
    ax.margins(x=0.01)

    # Legend outside on the right; scrollable column count grows for many runs.
    ncol = 1 if len(runs) <= 20 else 2
    leg = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=ncol,
        frameon=False,
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")

    # Interactive: matplotlib's toolbar already has zoom/pan/save buttons.
    plt.show()


# ----------------------------- CLI --------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "directory",
        help="Root dir holding run subdirs (each with *.monitor.csv files), "
        "or a single run dir.",
    )
    p.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Glob/substring patterns; keep only runs whose name matches any.",
    )
    p.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help="Glob/substring patterns; drop runs whose name matches any.",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Rolling-mean window over episodes (1 = no smoothing).",
    )
    p.add_argument(
        "--no-raw",
        action="store_true",
        help="When smoothing, hide the faint raw curve underneath.",
    )
    p.add_argument(
        "--xaxis",
        choices=["timesteps", "episodes"],
        default="timesteps",
        help="X axis: cumulative environment steps or episode index.",
    )
    p.add_argument("--title", default=None)
    p.add_argument(
        "--save",
        default=None,
        help="Also write the figure to this path (PNG/PDF/SVG).",
    )
    p.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=(12.0, 6.5),
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List discovered runs and exit (no plot).",
    )
    args = p.parse_args(argv)

    runs = discover_runs(args.directory)
    runs = filter_runs(runs, args.include, args.exclude)

    if not runs:
        print("No runs found after filtering.", file=sys.stderr)
        return 1

    print(f"Plotting {len(runs)} run(s) from {args.directory}:")
    for r in runs:
        print(f"  - {r.name}  ({len(r.df)} episodes, "
              f"{int(r.df['timesteps'].iloc[-1])} steps)")

    if args.list:
        return 0

    plot(
        runs,
        xaxis=args.xaxis,
        smooth_window=args.smooth,
        show_raw=not args.no_raw,
        title=args.title,
        save_path=args.save,
        figsize=tuple(args.figsize),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
