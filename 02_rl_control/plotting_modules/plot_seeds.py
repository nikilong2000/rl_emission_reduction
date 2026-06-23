"""
Aggregate results from multi-seed validation runs and produce publication plots.

Outputs three figures:
  1. training_progress.png   —  tensorboard-style mean reward curve across seeds
  2. eval_rmse_tracking.png  —  speed tracking during eval episode (mean ± std)
  3. final_metrics_bar.png   —  bar chart of mean evaluation metrics across seeds

Usage:
    python plot_seeds.py --results_dir logs/ppo/optuna/seeds/
    python plot_seeds.py --results_dir logs/sac/optuna/seeds/ --algorithm SAC
"""
import os
import re
import sys
import json
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from stable_baselines3.common.results_plotter import load_results

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
MEAN_COLOR   = "#1565C0"   # strong blue
STD_COLOR    = "#BDBDBD"   # grey-400
TARGET_COLOR = "#E53935"   # red
AXES_BG      = "#f5f5f5"   # tensorboard-like card background
GRID_COLOR   = "#cccccc"
GRID_ALPHA   = 0.35
FIG_DPI      = 200


def _apply_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=GRID_ALPHA, linewidth=0.5)
    ax.tick_params(labelsize=9)


def _ema(values, weight=0.6):
    """TensorBoard-style exponential moving average."""
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


# ---------------------------------------------------------------------------
# Plot 1 — Training progress (tensorboard-style mean across seeds)
# ---------------------------------------------------------------------------
def plot_training_progress(seed_dirs, output_dir, algo_label="RL",
                           ema_weight=0.6, n_grid=500):
    """Average training reward across seeds, plot single EMA-smoothed mean line."""
    raw_curves = []
    max_ts = []
    for sd in seed_dirs:
        try:
            df = load_results(sd)
        except Exception as e:
            print(f"  Warning: could not load monitor from {sd}: {e}")
            continue
        df = df.sort_values("t").reset_index(drop=True)
        timesteps = df["l"].cumsum().values  # cumulative episode lengths = global step count
        raw_curves.append((timesteps, df["r"].values))
        max_ts.append(timesteps[-1])

    if not raw_curves:
        print("No training data found — skipping training_progress plot.")
        return

    grid = np.linspace(0, min(max_ts), n_grid)
    interp = np.array([np.interp(grid, t, r) for t, r in raw_curves])
    mean_raw = interp.mean(axis=0)
    mean_smooth = _ema(mean_raw, weight=ema_weight)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(AXES_BG)
    ax.plot(grid, mean_smooth, color=MEAN_COLOR, linewidth=1.6, zorder=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_title(f"{algo_label} Training Progress ({len(raw_curves)} Seeds, mean)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Timesteps", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(output_dir, "training_progress.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Plot 2 — Eval RMSE / speed tracking
# ---------------------------------------------------------------------------
def plot_eval_rmse_tracking(seed_dirs, output_dir, algo_label="RL"):
    """Read evaluation_data.csv from each seed and plot speed tracking ± std."""
    all_actual = []
    all_target = []
    for sd in seed_dirs:
        csv_path = os.path.join(sd, "evaluation_data.csv")
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found — skipping seed.")
            continue
        df = pd.read_csv(csv_path)
        all_actual.append(df["speed_actual"].values)
        all_target.append(df["speed_target"].values)

    if not all_actual:
        print("No evaluation data found — skipping eval_rmse_tracking plot.")
        return

    min_len = min(len(a) for a in all_actual)
    actual_mat = np.array([a[:min_len] for a in all_actual])
    target_mat = np.array([t[:min_len] for t in all_target])
    actual_mean = actual_mat.mean(axis=0)
    actual_std  = actual_mat.std(axis=0)
    target_mean = target_mat.mean(axis=0)
    abs_error   = np.abs(actual_mat - target_mat)
    error_mean  = abs_error.mean(axis=0)
    error_std   = abs_error.std(axis=0)

    dt = 0.5  # seconds per step
    time_s = np.arange(min_len) * dt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Top — actual vs target speed
    ax = axes[0]
    ax.plot(time_s, target_mean, color=TARGET_COLOR, linewidth=1.5,
            label="Target", linestyle="--")
    ax.plot(time_s, actual_mean, color=MEAN_COLOR, linewidth=1.2,
            label="Actual (mean)")
    ax.fill_between(time_s, actual_mean - actual_std, actual_mean + actual_std,
                    color=STD_COLOR, alpha=0.5, label="± 1 Std")
    _apply_style(ax, f"{algo_label} Evaluation — Speed Tracking ({len(all_actual)} Seeds)",
                 "", "Speed (km/h)")
    ax.legend(fontsize=9, loc="upper right")
    # Bottom — absolute speed error
    ax = axes[1]
    ax.plot(time_s, error_mean, color=MEAN_COLOR, linewidth=1.2, label="Mean |error|")
    ax.fill_between(time_s, np.maximum(error_mean - error_std, 0),
                    error_mean + error_std,
                    color=STD_COLOR, alpha=0.5, label="± 1 Std")
    _apply_style(ax, "", "Time (s)", "|Speed Error| (km/h)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    path = os.path.join(output_dir, "eval_rmse_tracking.png")
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Plot 3 — Final evaluation metrics bar chart
# ---------------------------------------------------------------------------
METRIC_KEYS = [
    ("total_reward",    "Total Reward",      "#1565C0"),
    ("total_fuel_g",    "Total Fuel (g)",    "#2E7D32"),
    ("total_nox_g",     "Total NOx (g)",     "#E65100"),
    ("mae_speed_kmph",  "MAE Speed (km/h)",  "#6A1B9A"),
    ("rmse_speed_kmph", "RMSE Speed (km/h)", "#C62828"),
    ("delta_soc",       "ΔSOC",              "#00838F"),
]


def plot_final_metrics_bar(seed_dirs, output_dir, algo_label="RL"):
    """Read evaluation_metrics.json from each seed. One bar per metric, mean ± std across seeds."""
    records = []
    for sd in seed_dirs:
        json_path = os.path.join(sd, "evaluation_metrics.json")
        if not os.path.exists(json_path):
            print(f"  Warning: {json_path} not found — skipping seed.")
            continue
        with open(json_path) as f:
            records.append(json.load(f))

    if not records:
        print("No metrics JSON found — skipping final_metrics_bar plot.")
        return

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 7))
    axes = axes.flatten()

    for idx, (key, label, color) in enumerate(METRIC_KEYS):
        ax = axes[idx]
        values = [r.get(key, float("nan")) for r in records]
        values = [v for v in values if v is not None and not np.isnan(v)]
        if not values:
            ax.set_visible(False)
            continue
        mean_val = float(np.mean(values))
        std_val  = float(np.std(values))

        ax.bar(0, mean_val, width=0.5, color=color, alpha=0.85,
               yerr=std_val, capsize=8, ecolor="black",
               error_kw={"linewidth": 1.3})
        ax.set_xticks([])
        ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
        ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.5)
        ax.tick_params(labelsize=9)
        ax.margins(y=0.25)
        ax.axhline(0, color="black", linewidth=0.6)

        if mean_val >= 0:
            y_text, va = mean_val + std_val * 1.15, "bottom"
        else:
            y_text, va = mean_val - std_val * 1.15, "top"
        ax.text(0, y_text, f"{mean_val:.3f}±{std_val:.3f}",
                ha="center", va=va, fontsize=9, fontweight="bold")

    for idx in range(len(METRIC_KEYS), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"{algo_label} Final Evaluation Metrics ({len(records)} Seeds)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "final_metrics_bar.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory.")
        sys.exit(1)

    algo_label = args.algorithm.upper() if args.algorithm else "RL"

    seed_dirs = sorted(glob.glob(os.path.join(results_dir, "seed_*")))
    # Only canonical "seed_<int>" dirs; skip auxiliary copies like
    # "seed_7_wltc" or "seed_7 copy" that would contaminate the aggregates.
    seed_dirs = [
        d for d in seed_dirs
        if os.path.isdir(d) and re.fullmatch(r"seed_\d+", os.path.basename(d))
    ]
    print(f"Found {len(seed_dirs)} seed directories in {results_dir}")
    if not seed_dirs:
        print("No seed directories found. Nothing to plot.")
        return

    print("\n1/3 Training progress...")
    plot_training_progress(seed_dirs, results_dir, algo_label)

    print("\n2/3 Eval RMSE tracking...")
    plot_eval_rmse_tracking(seed_dirs, results_dir, algo_label)

    print("\n3/3 Final metrics bar chart...")
    plot_final_metrics_bar(seed_dirs, results_dir, algo_label)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed results into publication plots."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing seed_0/, seed_1/, ... subdirectories.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Algorithm name for plot titles (e.g. PPO, SAC, TD3).",
    )
    main(parser.parse_args())
