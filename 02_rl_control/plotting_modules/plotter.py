"""
Plot SB3 training curves for many runs in one interactive figure.

A "run" is any directory containing one or more `*.monitor.csv` files
(SB3 Monitor wrapper output). The script walks a target directory,
discovers runs, loads their monitor CSVs, and overlays reward curves on
a single interactive matplotlib figure (zoom/pan + screenshot ready).

Examples
--------
# All trials of PPO phase 1 optuna sweep
python plotting_modules/plotter.py logs_cluster_phase1/logs/ppo/optuna

# Only trials 050..069
python plotting_modules/plotter.py logs_cluster_phase1/logs/ppo/optuna \
    --include 'trial_05*' 'trial_06*'

# Drop a noisy run
python plotting_modules/plotter.py logs_cluster_phase1/logs/ppo/optuna/seeds \
    --exclude seed_3

# Smoothing window of 50 episodes
python plotting_modules/plotter.py logs_cluster_phase1/logs/ppo/optuna --smooth 50

# Compare mean ± std across algorithms (each dir holds seed_* subdirs)
python plotting_modules/plotter.py --compare \
    logs_cluster_phase1/logs/ppo/optuna/seeds \
    logs_cluster_phase1/logs/sac/optuna/seeds \
    logs_cluster_phase1/logs/td3/optuna/seeds \
    --labels PPO SAC TD3 --smooth 30

# Compare action distributions aggregated across all seeds per algorithm
python plotting_modules/plotter.py --compare-actions \
    logs_cluster_phase1/logs/ppo/optuna/seeds \
    logs_cluster_phase1/logs/sac/optuna/seeds \
    logs_cluster_phase1/logs/td3/optuna/seeds \
    --labels PPO SAC TD3

# Time-series + distribution side-by-side for one run (5 rows x 2 cols)
python plotting_modules/plotter.py --actions-panel \
    logs_cluster_phase1/logs/ppo/optuna/seeds/seed_0
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
    csvs = sorted(f for f in os.listdir(path) if f.endswith(".monitor.csv"))
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


# ----------------------------- aggregation ------------------------------------


def aggregate_seeds(
    runs: list[Run],
    *,
    xaxis: str = "timesteps",
    smooth_window: int = 1,
    n_grid: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Mean ± std reward across seeds on a common x-grid.

    Each run's smoothed curve is linearly interpolated onto a shared grid
    spanning [0, min(max_x across seeds)]. Returns (x, mean, std, n_seeds).
    """
    if not runs:
        raise ValueError("No runs to aggregate.")

    xkey = "timesteps" if xaxis == "timesteps" else "episode"
    x_max = min(r.df[xkey].iloc[-1] for r in runs)
    x_min = max(r.df[xkey].iloc[0] for r in runs)
    grid = np.linspace(x_min, x_max, n_grid)

    interp = []
    for r in runs:
        x = r.df[xkey].to_numpy()
        y = smooth(r.df["r"].to_numpy(), smooth_window)
        interp.append(np.interp(grid, x, y))
    stack = np.vstack(interp)
    return grid, stack.mean(axis=0), stack.std(axis=0), len(runs)


# ----------------------------- plotting ---------------------------------------


def _color_cycle(n: int):
    if n <= 10:
        cmap = plt.get_cmap("tab10")
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("turbo")
    return (
        [cmap(i / max(n - 1, 1)) for i in range(n)]
        if n > 20
        else [cmap(i % cmap.N) for i in range(n)]
    )


# Fixed per-algorithm colors, used only on multi-algorithm comparison plots.
ALGO_COLORS: dict[str, str] = {
    "ppo": "#1f77b4",
    "td3": "#d62728",
    "sac": "#2ca02c",
}


def _algo_key(label: str) -> str | None:
    low = label.lower()
    for key in ALGO_COLORS:
        if key in low:
            return key
    return None


def _group_colors(labels: list[str]):
    """Return algo-fixed colors when the plot spans >=2 algorithms; otherwise
    fall back to the default cycle. Unrecognised labels keep cycle colors."""
    cycle = _color_cycle(len(labels))
    if len(labels) < 2:
        return cycle
    keys = [_algo_key(lbl) for lbl in labels]
    if sum(k is not None for k in keys) < 2:
        return cycle
    return [ALGO_COLORS[k] if k else cycle[i] for i, k in enumerate(keys)]


def plot(
    runs: list[Run],
    *,
    xaxis: str = "timesteps",
    smooth_window: int = 1,
    show_raw: bool = True,
    title: str | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (12.0, 6.5),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    legend_cols: int | None = None,
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
        ax.plot(x, ys, color=color, linewidth=1.1, alpha=0.9, label=run.name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Episode Reward")
    ax.set_title(title or "Training Curves")
    ax.margins(x=0.01)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Legend outside on the right; scrollable column count grows for many runs.
    ncol = legend_cols if legend_cols else (1 if len(runs) <= 20 else 2)
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


ACTION_CONFIGS: list[tuple[str, str, tuple[float, float]]] = [
    ("engine_on", "Engine On (0/1)", (-0.1, 1.1)),
    ("ice_speed_rpm", "ICE Speed (RPM)", (0, 4500)),
    ("em2_torque_nm", "EM2 Torque (Nm)", (-450, 450)),
    ("fuel", "Fuel (mg)", (0, 80)),
    ("brake_perc", "Brake (%)", (0, 105)),
]

# Per-variable colors matching plotting.plot_actions().
ACTION_PANEL_COLORS: dict[str, str] = {
    "engine_on": "purple",
    "ice_speed_rpm": "blue",
    "em2_torque_nm": "green",
    "fuel": "orange",
    "brake_perc": "red",
}


def _find_eval_csvs(root: str) -> list[str]:
    """Find all evaluation_data.csv files under `root` (recursive, one level
    of seed_*/trial_* typically)."""
    paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "evaluation_data.csv" in filenames:
            paths.append(os.path.join(dirpath, "evaluation_data.csv"))
    return sorted(paths)


def _seed_name(root: str, csv_path: str) -> str:
    """Get run-folder name relative to `root` (e.g. 'seed_0')."""
    rel = os.path.relpath(os.path.dirname(csv_path), root)
    return rel if rel != "." else os.path.basename(os.path.normpath(root))


def load_actions_for_group(
    root: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """Concat each action column across all seed CSVs in `root`.

    Returns ({action_key: 1D array}, n_seeds).
    """
    csvs = _find_eval_csvs(root)
    kept: list[str] = []
    for c in csvs:
        name = _seed_name(root, c)
        if include and not _match_any(name, include):
            continue
        if exclude and _match_any(name, exclude):
            continue
        kept.append(c)

    data: dict[str, list[np.ndarray]] = {k: [] for k, _, _ in ACTION_CONFIGS}
    for c in kept:
        try:
            df = pd.read_csv(c)
        except Exception as e:
            print(f"  ! skip {c}: {e}", file=sys.stderr)
            continue
        for key, _, _ in ACTION_CONFIGS:
            if key in df.columns:
                data[key].append(df[key].to_numpy(dtype=float))

    out = {
        k: (np.concatenate(v) if v else np.array([], dtype=float))
        for k, v in data.items()
    }
    return out, len(kept)


def plot_compare_actions(
    groups: list[tuple[str, dict[str, np.ndarray], int]],
    *,
    bins: int = 100,
    title: str | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (16.0, 9.0),
    kde: bool = True,
) -> None:
    """5-panel grid (one per action). Histogram (density) + KDE overlay,
    one color per algorithm group."""
    if not groups:
        print("No groups to compare.", file=sys.stderr)
        return

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    ncols = 3
    nrows = (len(ACTION_CONFIGS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    colors = _color_cycle(len(groups))

    try:
        from scipy.stats import gaussian_kde  # noqa: WPS433

        have_scipy = True
    except Exception:
        have_scipy = False
        if kde:
            print(
                "scipy not available — falling back to histogram only.", file=sys.stderr
            )

    for ax_idx, (key, xlabel, xlim) in enumerate(ACTION_CONFIGS):
        ax = axes[ax_idx]
        is_binary = key == "engine_on"

        n_groups = sum(
            1 for _, data, _ in groups if data.get(key, np.array([])).size > 0
        )
        bar_width = 0.8 / max(n_groups, 1) if is_binary else None
        g_idx = 0

        for (label, data, n), color in zip(groups, colors):
            arr = data.get(key, np.array([]))
            if arr.size == 0:
                continue
            arr = arr[(arr >= xlim[0]) & (arr <= xlim[1])]
            if arr.size == 0:
                continue

            if is_binary:
                arr_int = np.round(arr).astype(int)
                probs = np.array([np.mean(arr_int == 0), np.mean(arr_int == 1)])
                centers = np.array([0.0, 1.0])
                offset = (g_idx - (n_groups - 1) / 2.0) * bar_width
                ax.bar(
                    centers + offset,
                    probs,
                    width=bar_width,
                    color=color,
                    alpha=0.85,
                    edgecolor="none",
                    label=f"{label} (n={n})" if ax_idx == 0 else None,
                )
                g_idx += 1
            else:
                ax.hist(
                    arr,
                    bins=bins,
                    range=xlim,
                    density=True,
                    alpha=0.35,
                    color=color,
                    edgecolor="none",
                    label=f"{label} (n={n})" if ax_idx == 0 else None,
                )

                if kde and have_scipy and np.std(arr) > 1e-6:
                    try:
                        kfn = gaussian_kde(arr)
                        xs = np.linspace(xlim[0], xlim[1], 400)
                        ax.plot(xs, kfn(xs), color=color, linewidth=1.8, alpha=0.95)
                    except Exception as e:
                        print(f"  ! KDE fail for {label}/{key}: {e}", file=sys.stderr)

        ax.set_xlabel(xlabel)
        ax.set_title(f"π({key})")
        if is_binary:
            ax.set_ylabel("Probability")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Off (0)", "On (1)"])
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylabel("Density")
            ax.set_xlim(xlim)

    for ax_idx in range(len(ACTION_CONFIGS), len(axes)):
        axes[ax_idx].set_visible(False)

    # Single legend on figure (collect handles from first axis).
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.05),
            frameon=False,
            title="Algorithm",
        )

    fig.suptitle(
        title or "Action Distribution π(a) — Aggregated across Seeds", fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def _resolve_eval_csv(path: str) -> str:
    """Return the evaluation_data.csv for a single run dir.

    Accepts the CSV path itself, a run dir containing it, or a parent dir
    holding exactly one run (first match wins, with a note).
    """
    if os.path.isfile(path) and path.endswith(".csv"):
        return path
    direct = os.path.join(path, "evaluation_data.csv")
    if os.path.isfile(direct):
        return direct
    found = _find_eval_csvs(path)
    if not found:
        raise FileNotFoundError(f"No evaluation_data.csv under {path}")
    if len(found) > 1:
        print(f"  ! {len(found)} eval CSVs found; using {found[0]}", file=sys.stderr)
    return found[0]


def plot_actions_panel(
    df: pd.DataFrame,
    run_name: str,
    *,
    bins: int = 100,
    window_start: int | None = None,
    window_size: int | None = None,
    title: str | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (15.0, 16.0),
    kde: bool = True,
    manual_layout: bool = False,
) -> None:
    """5 rows x 2 cols: left = action time series, right = its distribution.

    One row per action variable, sharing a color. engine_on shows Off/On
    probability bars instead of a density.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    try:
        from scipy.stats import gaussian_kde  # noqa: WPS433

        have_scipy = True
    except Exception:
        have_scipy = False
        if kde:
            print("scipy not available — histogram only.", file=sys.stderr)

    total = len(df)
    s = 0 if window_start is None else max(0, window_start)
    e = total if window_size is None else min(total, s + window_size)
    t = np.arange(s, e)

    nrows = len(ACTION_CONFIGS)
    fig, axes = plt.subplots(
        nrows, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.2, 1.0]}
    )

    for row, (key, label, xlim) in enumerate(ACTION_CONFIGS):
        ax_ts, ax_hist = axes[row, 0], axes[row, 1]
        color = ACTION_PANEL_COLORS.get(key, "gray")
        is_binary = key == "engine_on"

        full = df[key].to_numpy(dtype=float) if key in df.columns else np.array([])
        seg = full[s:e] if full.size else full

        # --- left: time series ---
        if seg.size:
            if is_binary:
                ax_ts.step(t, seg, where="mid", color=color, linewidth=1.2)
            else:
                ax_ts.plot(t, seg, color=color, linewidth=1.0, alpha=0.9)
        ax_ts.set_ylabel(label)
        if is_binary:
            ax_ts.set_ylim(-0.2, 1.2)
            ax_ts.set_yticks([0, 1])
            ax_ts.set_yticklabels(["Off", "On"])
        if row == 0:
            ax_ts.set_title(f"Time Series — {run_name}")
        if row == nrows - 1:
            ax_ts.set_xlabel("Time Step")

        # --- right: distribution ---
        dist = full[(full >= xlim[0]) & (full <= xlim[1])] if full.size else full
        if dist.size:
            if is_binary:
                arr_int = np.round(dist).astype(int)
                probs = [np.mean(arr_int == 0), np.mean(arr_int == 1)]
                ax_hist.bar(
                    [0, 1],
                    probs,
                    width=0.6,
                    color=color,
                    alpha=0.85,
                    edgecolor="none",
                )
            else:
                ax_hist.hist(
                    dist,
                    bins=bins,
                    range=xlim,
                    density=True,
                    alpha=0.35,
                    color=color,
                    edgecolor="none",
                )
                if kde and have_scipy and np.std(dist) > 1e-6:
                    try:
                        kfn = gaussian_kde(dist)
                        xs = np.linspace(xlim[0], xlim[1], 400)
                        ax_hist.plot(xs, kfn(xs), color=color, linewidth=1.8)
                    except Exception as ex:
                        print(f"  ! KDE fail {key}: {ex}", file=sys.stderr)

        if is_binary:
            ax_hist.set_xticks([0, 1])
            ax_hist.set_xticklabels(["Off (0)", "On (1)"])
            ax_hist.set_xlim(-0.6, 1.6)
            ax_hist.set_ylim(0, 1.0)
            ax_hist.set_ylabel("Probability")
        else:
            ax_hist.set_xlim(xlim)
            ax_hist.set_ylabel("Density")
        if row == 0:
            ax_hist.set_title("Distribution")
        ax_hist.set_xlabel(label)

    fig.suptitle(
        title or f"Action Time Series & Distribution — {run_name}", fontsize=14
    )
    if manual_layout:
        # Hand-tuned values that fit the full overview without clipping.
        fig.subplots_adjust(
            left=0.06,
            bottom=0.063,
            right=0.983,
            top=0.913,
            wspace=0.127,
            hspace=0.331,
        )
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_compare(
    groups: list[tuple[str, list[Run]]],
    *,
    xaxis: str = "timesteps",
    smooth_window: int = 1,
    show_band: bool = True,
    band: str = "std",  # "std" or "minmax"
    title: str | None = None,
    save_path: str | None = None,
    figsize: tuple[float, float] = (12.0, 6.5),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Overlay mean (± band) reward curve for each algorithm group."""
    if not groups:
        print("No groups to compare.", file=sys.stderr)
        return

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors = _group_colors([lbl for lbl, _ in groups])
    xlabel = "Total Environment Steps" if xaxis == "timesteps" else "Episode"

    for (label, runs), color in zip(groups, colors):
        if not runs:
            print(f"  ! {label}: no runs, skipping", file=sys.stderr)
            continue
        x, mean, std, n = aggregate_seeds(
            runs, xaxis=xaxis, smooth_window=smooth_window
        )
        if band == "minmax":
            xkey = "timesteps" if xaxis == "timesteps" else "episode"
            stack = np.vstack(
                [
                    np.interp(
                        x,
                        r.df[xkey].to_numpy(),
                        smooth(r.df["r"].to_numpy(), smooth_window),
                    )
                    for r in runs
                ]
            )
            lo, hi = stack.min(axis=0), stack.max(axis=0)
        else:
            lo, hi = mean - std, mean + std

        ax.plot(x, mean, color=color, linewidth=1.1, label=f"{label} (n={n})")
        if show_band:
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Episode Reward")
    ax.set_title(title or "Algorithm Training Speed Comparison — Mean across Seeds")
    ax.margins(x=0.01)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


# ----------------------------- CLI --------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="Root dir holding run subdirs (each with *.monitor.csv files), "
        "or a single run dir. Omit when using --compare.",
    )
    p.add_argument(
        "--compare",
        nargs="+",
        default=None,
        metavar="DIR",
        help="Compare multiple algorithm groups: pass one parent dir per "
        "algorithm (each holds seed_* / trial_* subdirs). Plots mean ± band.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display labels matching --compare dirs (default: dir basenames).",
    )
    p.add_argument(
        "--band",
        choices=["std", "minmax", "none"],
        default="std",
        help="Uncertainty band in --compare mode.",
    )
    p.add_argument(
        "--compare-actions",
        nargs="+",
        default=None,
        metavar="DIR",
        help="Aggregate evaluation_data.csv from all seed subdirs and overlay "
        "5-panel action distribution histograms + KDE across algorithms.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Histogram bins for --compare-actions (default: 100).",
    )
    p.add_argument(
        "--no-kde",
        action="store_true",
        help="Disable KDE overlay in --compare-actions / --actions-panel.",
    )
    p.add_argument(
        "--actions-panel",
        default=None,
        metavar="DIR",
        help="Single run dir (e.g. .../seeds/seed_0) with evaluation_data.csv. "
        "Plots 5 action time series (left) beside their distributions (right).",
    )
    p.add_argument(
        "--window-start",
        type=int,
        default=None,
        help="Time-series start index for --actions-panel (default: 0).",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Time-series length for --actions-panel (default: full episode). "
        "Distributions always use the full episode.",
    )
    p.add_argument(
        "--manual-layout",
        action="store_true",
        help="Use hand-tuned subplot spacing for --actions-panel instead of "
        "tight_layout (fixes clipped overview without the config tool).",
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
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Axis x limits for training-curve plots (default / --compare).",
    )
    p.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="Axis y limits for training-curve plots (default / --compare).",
    )
    p.add_argument(
        "--legend-cols",
        type=int,
        default=None,
        help="Force legend column count for default plot() mode (default: "
        "1 for ≤20 runs, 2 otherwise).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List discovered runs and exit (no plot).",
    )
    args = p.parse_args(argv)

    if args.actions_panel:
        csv = _resolve_eval_csv(args.actions_panel)
        run_name = os.path.basename(os.path.dirname(csv))
        df = pd.read_csv(csv)
        print(f"Actions panel: {run_name} ({csv}), {len(df)} steps")
        if args.list:
            return 0
        plot_actions_panel(
            df,
            run_name,
            bins=args.bins,
            window_start=args.window_start,
            window_size=args.window_size,
            title=args.title,
            save_path=args.save,
            figsize=(
                tuple(args.figsize) if args.figsize != (12.0, 6.5) else (15.0, 16.0)
            ),
            kde=not args.no_kde,
            manual_layout=args.manual_layout,
        )
        return 0

    if args.compare_actions:
        labels = args.labels or [
            os.path.basename(os.path.normpath(d)) for d in args.compare_actions
        ]
        if len(labels) != len(args.compare_actions):
            print(
                "--labels count must match --compare-actions dir count.",
                file=sys.stderr,
            )
            return 2

        action_groups: list[tuple[str, dict[str, np.ndarray], int]] = []
        for label, d in zip(labels, args.compare_actions):
            data, n = load_actions_for_group(d, args.include, args.exclude)
            total = sum(int(v.size) for v in data.values())
            print(f"{label}  ({d}): {n} seed(s), {total} action samples total")
            action_groups.append((label, data, n))

        if args.list:
            return 0
        if not any(n for _, _, n in action_groups):
            print("No evaluation_data.csv files found.", file=sys.stderr)
            return 1

        plot_compare_actions(
            action_groups,
            bins=args.bins,
            title=args.title,
            save_path=args.save,
            figsize=tuple(args.figsize) if args.figsize != (12.0, 6.5) else (16.0, 9.0),
            kde=not args.no_kde,
        )
        return 0

    if args.compare:
        labels = args.labels or [
            os.path.basename(os.path.normpath(d)) for d in args.compare
        ]
        if len(labels) != len(args.compare):
            print("--labels count must match --compare dir count.", file=sys.stderr)
            return 2

        groups: list[tuple[str, list[Run]]] = []
        for label, d in zip(labels, args.compare):
            runs = filter_runs(discover_runs(d), args.include, args.exclude)
            print(f"{label}  ({d}): {len(runs)} seed run(s)")
            for r in runs:
                print(
                    f"  - {r.name}  ({len(r.df)} episodes, "
                    f"{int(r.df['timesteps'].iloc[-1])} steps)"
                )
            groups.append((label, runs))

        if args.list:
            return 0
        if not any(rs for _, rs in groups):
            print("No runs found in any compare dir.", file=sys.stderr)
            return 1

        plot_compare(
            groups,
            xaxis=args.xaxis,
            smooth_window=args.smooth,
            show_band=(args.band != "none"),
            band=args.band if args.band != "none" else "std",
            title=args.title,
            save_path=args.save,
            figsize=tuple(args.figsize),
            xlim=tuple(args.xlim) if args.xlim else None,
            ylim=tuple(args.ylim) if args.ylim else None,
        )
        return 0

    if not args.directory:
        print("Provide a directory or use --compare.", file=sys.stderr)
        return 2

    runs = discover_runs(args.directory)
    runs = filter_runs(runs, args.include, args.exclude)

    if not runs:
        print("No runs found after filtering.", file=sys.stderr)
        return 1

    print(f"Plotting {len(runs)} run(s) from {args.directory}:")
    for r in runs:
        print(
            f"  - {r.name}  ({len(r.df)} episodes, "
            f"{int(r.df['timesteps'].iloc[-1])} steps)"
        )

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
        xlim=tuple(args.xlim) if args.xlim else None,
        ylim=tuple(args.ylim) if args.ylim else None,
        legend_cols=args.legend_cols,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
