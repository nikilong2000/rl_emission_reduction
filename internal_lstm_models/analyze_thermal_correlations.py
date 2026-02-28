#!/usr/bin/env python3
"""
Thermal Correlation Analysis for ICE LSTM Model Outputs
========================================================
Analyses correlations between the 7 thermal output variables of the ICE LSTM
(T_gas_eo_K, T_Wall_SCR1_K, T_Wall_DOC_K, T_Sub_DPF_K, T_Wall_SCR2_K,
T_Wall_SCR3_K, T_gas_tp_K) across all training data in Data_ICE/.

Purpose: Inform which thermal variables to include in the RL agent's
observation space by identifying redundancies and key independent dimensions.

Outputs are saved to thermal_analysis_results/.
"""

import glob
import os
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data_ICE")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "thermal_analysis_results")

# Mapping: raw Data_ICE column name → clean config.txt output name
THERMAL_COL_MAP = {
    "T_avg_SCR1;K": "T_gas_eo_K",  # Engine-out gas temperature
    "T_Wall_SCR1;K": "T_Wall_SCR1_K",
    "T_Wall_DOC;K": "T_Wall_DOC_K",
    "T_Sub_DPF;K": "T_Sub_DPF_K",
    "T_Wall_SCR2;K": "T_Wall_SCR2_K",
    "T_Wall_SCR3;K": "T_Wall_SCR3_K",
    "T_Tailpipe;K": "T_gas_tp_K",  # Tailpipe gas temperature
}

EMISSION_COL_MAP = {
    "NOx_in_m;gps": "NOx_eo_gps",  # Engine-out NOx
    "NOx_out_m;gps": "NOx_tp_gps",  # Tailpipe NOx
}

OPERATIONAL_COL_MAP = {
    "ICE_Speed;rpm": "ICE_Speed_rpm",
    "fuel_soll;mg": "fuel_mg",
}

THERMAL_COLS = list(THERMAL_COL_MAP.values())
EMISSION_COLS = list(EMISSION_COL_MAP.values())
OPERATIONAL_COLS = list(OPERATIONAL_COL_MAP.values())

ALL_RENAME_MAP = {**THERMAL_COL_MAP, **EMISSION_COL_MAP, **OPERATIONAL_COL_MAP}

# Exhaust path order (physical flow direction)
EXHAUST_PATH_ORDER = [
    "T_gas_eo_K",  # Engine-out gas
    "T_Wall_DOC_K",  # Diesel Oxidation Catalyst
    "T_Sub_DPF_K",  # Diesel Particulate Filter substrate
    "T_Wall_SCR1_K",  # SCR stage 1
    "T_Wall_SCR2_K",  # SCR stage 2
    "T_Wall_SCR3_K",  # SCR stage 3
    "T_gas_tp_K",  # Tailpipe
]

# Scenario categories extracted from filename patterns
SCENARIO_PATTERNS = {
    "cooling": r"imp_cooling",
    "high_dynamic": r"imp_high_dynamic",
    "low_dynamic": r"imp_low_dynamic",
    "low_load": r"imp_low_load",
    "const_speed": r"imp_Speed_const",
    "normal": r"normal_Speed_fuel",
    "timeseries": r"timeseries_batch",
    "EA189": r"EA189",
    "ramp": r"ramp",
}

# Plot style
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.facecolor": "white",
    }
)

SCATTER_SAMPLE_SIZE = 50_000  # Max points for scatter plots


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────


def classify_scenario(filename: str) -> str:
    """Classify a filename into a scenario category."""
    for category, pattern in SCENARIO_PATTERNS.items():
        if re.search(pattern, filename):
            return category
    return "other"


def load_all_data(data_dir: str) -> pd.DataFrame:
    """Load and concatenate all CSV files from Data_ICE/."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"ERROR: No CSV files found in {data_dir}")
        sys.exit(1)

    print(f"Loading {len(csv_files)} CSV files from {data_dir} ...")
    frames = []
    skipped = 0

    for i, fpath in enumerate(csv_files):
        fname = os.path.basename(fpath)
        try:
            # Line 1 = comment, Line 2 = blank, Line 3 = headers, Line 4 = units, Lines 5+ = data
            df = pd.read_csv(fpath, sep="\t", skiprows=[0, 1, 3], low_memory=False)

            # Keep only columns we need (some files might have slightly different cols)
            available = {c for c in df.columns if c in ALL_RENAME_MAP}
            if not available:
                skipped += 1
                continue

            df = df[[c for c in df.columns if c in ALL_RENAME_MAP or c == "Time;s"]]
            df = df.rename(columns={**ALL_RENAME_MAP, "Time;s": "Time_s"})

            # Convert to numeric (some files might have spurious strings)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["file_id"] = i
            df["filename"] = fname
            df["scenario"] = classify_scenario(fname)
            frames.append(df)

        except Exception as e:
            print(f"  WARN: Skipping {fname}: {e}")
            skipped += 1

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(csv_files)} files ...")

    data = pd.concat(frames, ignore_index=True)
    print(f"  Done. {len(frames)} files loaded, {skipped} skipped.")
    print(f"  Total rows: {len(data):,}")
    print(f"  Scenario distribution:\n{data['scenario'].value_counts().to_string()}\n")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ──────────────────────────────────────────────────────────────────────────────


def plot_correlation_heatmap(
    data: pd.DataFrame,
    columns: list,
    title: str,
    filename: str,
    method: str = "pearson",
):
    """Plot a correlation heatmap for the given columns."""
    corr = data[columns].corr(method=method)

    fig, ax = plt.subplots(figsize=(10, 8))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(corr.values, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(columns, fontsize=9)

    # Annotate cells
    for i in range(len(columns)):
        for j in range(len(columns)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.7 else "black"
            ax.text(
                j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color
            )

    plt.colorbar(im, ax=ax, label=f"{method.title()} correlation")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")
    return corr


def plot_scatter_matrix(
    data: pd.DataFrame, columns: list, title: str, filename: str, hue_col: str = None
):
    """Plot pairwise scatter matrix (sampled)."""
    n = min(SCATTER_SAMPLE_SIZE, len(data))
    sample = data.sample(n=n, random_state=42)

    ncols = len(columns)
    fig, axes = plt.subplots(ncols, ncols, figsize=(3 * ncols, 3 * ncols))

    if hue_col and hue_col in sample.columns:
        categories = sample[hue_col].unique()
        cmap = plt.cm.get_cmap("tab10", len(categories))
        color_map = {cat: cmap(i) for i, cat in enumerate(categories)}
        colors = sample[hue_col].map(color_map)
    else:
        colors = "steelblue"

    for i in range(ncols):
        for j in range(ncols):
            ax = axes[i, j]
            if i == j:
                ax.hist(
                    sample[columns[i]].dropna(),
                    bins=50,
                    color="steelblue",
                    alpha=0.7,
                    edgecolor="none",
                )
                if i == 0:
                    ax.set_ylabel(columns[i], fontsize=8)
            else:
                ax.scatter(
                    sample[columns[j]],
                    sample[columns[i]],
                    s=1,
                    alpha=0.15,
                    c=colors,
                    rasterized=True,
                )

            if j == 0 and i != j:
                ax.set_ylabel(columns[i], fontsize=8)
            if i == ncols - 1:
                ax.set_xlabel(columns[j], fontsize=8, rotation=45, ha="right")

            ax.tick_params(labelsize=6)

            if i != ncols - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    plt.close()
    print(f"  Saved {filename}")


def run_pca_analysis(data: pd.DataFrame, columns: list):
    """Run PCA on thermal columns and plot explained variance."""
    clean = data[columns].dropna()
    if len(clean) < 10:
        print("  WARN: Not enough data for PCA.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean)

    pca = PCA()
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Bar + cumulative line chart
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(1, len(explained) + 1)

    bars = ax1.bar(x, explained * 100, color="steelblue", alpha=0.8, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    ax2.plot(
        x, cumulative * 100, "o-", color="darkorange", linewidth=2, label="Cumulative"
    )
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_ylim(0, 105)

    # Annotate bars
    for bar, val in zip(bars, explained):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val * 100:.1f}%",
            ha="center",
            fontsize=9,
        )

    # 90% and 95% lines
    ax2.axhline(90, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(95, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax2.text(len(x) + 0.1, 90, "90%", fontsize=8, color="gray", va="center")
    ax2.text(len(x) + 0.1, 95, "95%", fontsize=8, color="gray", va="center")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("PCA on Thermal Variables — Explained Variance")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "pca_explained_variance.png"), bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved pca_explained_variance.png")

    # Loadings heatmap
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(columns))],
        index=columns,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(loadings.values, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(loadings.shape[1]))
    ax.set_yticks(range(loadings.shape[0]))
    ax.set_xticklabels(loadings.columns, fontsize=9)
    ax.set_yticklabels(loadings.index, fontsize=9)

    for i in range(loadings.shape[0]):
        for j in range(loadings.shape[1]):
            val = loadings.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color
            )

    plt.colorbar(im, ax=ax, label="Loading weight")
    ax.set_title("PCA Loadings — How Each Thermal Variable Contributes to Each PC")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_loadings.png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved pca_loadings.png")

    # Number of components for 90% and 95%
    n_90 = np.searchsorted(cumulative, 0.90) + 1
    n_95 = np.searchsorted(cumulative, 0.95) + 1

    return explained, cumulative, loadings, n_90, n_95


def plot_distributions(data: pd.DataFrame, columns: list):
    """Plot distribution histograms for all thermal variables."""
    ncols = 3
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        vals = data[col].dropna()
        ax.hist(vals, bins=100, color="steelblue", alpha=0.8, edgecolor="none")
        ax.set_title(col)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Count")
        # Stats annotation
        stats_text = (
            f"min={vals.min():.0f}  max={vals.max():.0f}\n"
            f"mean={vals.mean():.0f}  std={vals.std():.0f}"
        )
        ax.text(
            0.97,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    # Hide unused axes
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Distribution of Thermal Variables Across All Training Data",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "thermal_distributions.png"), bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved thermal_distributions.png")


def plot_lag_correlation(data: pd.DataFrame, columns: list):
    """
    Compute and plot time-lagged cross-correlation along the exhaust path
    for a representative dynamic scenario.
    """
    # Pick one file from a high-dynamic scenario for clearest signal
    dynamic_files = data[data["scenario"] == "high_dynamic"]["file_id"].unique()
    if len(dynamic_files) == 0:
        dynamic_files = data[data["scenario"] == "EA189"]["file_id"].unique()
    if len(dynamic_files) == 0:
        dynamic_files = data["file_id"].unique()[:1]

    file_id = dynamic_files[0]
    ts = data[data["file_id"] == file_id].sort_values("Time_s").reset_index(drop=True)

    # Use exhaust path order
    path_cols = [
        c for c in EXHAUST_PATH_ORDER if c in ts.columns and ts[c].notna().sum() > 10
    ]
    if len(path_cols) < 2:
        print("  WARN: Not enough columns for lag correlation.")
        return

    # Cross-correlation of each consecutive pair
    dt = 0.5  # timestep in seconds
    max_lag_s = 60  # max lag in seconds
    max_lag_samples = int(max_lag_s / dt)

    fig, axes = plt.subplots(
        len(path_cols) - 1, 1, figsize=(12, 3 * (len(path_cols) - 1)), sharex=True
    )
    if len(path_cols) - 1 == 1:
        axes = [axes]

    peak_lags = []
    for idx in range(len(path_cols) - 1):
        col_a = path_cols[idx]
        col_b = path_cols[idx + 1]
        a = ts[col_a].interpolate().bfill().values
        b = ts[col_b].interpolate().bfill().values

        # Normalise
        a = (a - a.mean()) / (a.std() + 1e-10)
        b = (b - b.mean()) / (b.std() + 1e-10)

        # Cross-correlation
        corr = signal.correlate(b, a, mode="full") / len(a)
        lags = signal.correlation_lags(len(a), len(b), mode="full")
        lag_mask = np.abs(lags) <= max_lag_samples
        corr = corr[lag_mask]
        lag_times = lags[lag_mask] * dt

        peak_idx = np.argmax(corr)
        peak_lag = lag_times[peak_idx]
        peak_lags.append((col_a, col_b, peak_lag, corr[peak_idx]))

        ax = axes[idx]
        ax.plot(lag_times, corr, color="steelblue", linewidth=1)
        ax.axvline(peak_lag, color="red", linestyle="--", alpha=0.7)
        ax.set_ylabel("Cross-corr")
        ax.set_title(f"{col_a} → {col_b}  (peak lag = {peak_lag:.1f}s)", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Lag (seconds)")
    fig.suptitle(
        "Time-Lagged Cross-Correlation Along Exhaust Path\n"
        f"(Sample: file_id={file_id})",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "lag_cross_correlation.png"), bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved lag_cross_correlation.png")
    return peak_lags


def plot_per_scenario_correlation(data: pd.DataFrame, columns: list):
    """Plot thermal correlation heatmaps per scenario type."""
    scenarios = data["scenario"].unique()
    n = len(scenarios)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    for idx, scenario in enumerate(sorted(scenarios)):
        row, col_idx = divmod(idx, ncols)
        ax = axes[row, col_idx]
        subset = data[data["scenario"] == scenario]
        corr = subset[columns].corr(method="pearson")

        im = ax.imshow(corr.values, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(columns)))
        ax.set_yticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(columns, fontsize=7)
        ax.set_title(f"{scenario}\n(n={len(subset):,})", fontsize=10)

        for i in range(len(columns)):
            for j in range(len(columns)):
                val = corr.values[i, j]
                color = "white" if abs(val) > 0.7 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )

    # Hide unused axes
    for idx in range(len(scenarios), nrows * ncols):
        row, col_idx = divmod(idx, ncols)
        axes[row, col_idx].set_visible(False)

    fig.suptitle("Thermal Correlations by Scenario Type (Pearson)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "correlation_by_scenario.png"), bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved correlation_by_scenario.png")


def plot_thermal_timeseries_sample(data: pd.DataFrame, columns: list):
    """Plot thermal time series for a representative dynamic scenario."""
    dynamic_files = data[data["scenario"] == "high_dynamic"]["file_id"].unique()
    if len(dynamic_files) == 0:
        dynamic_files = data[data["scenario"] == "EA189"]["file_id"].unique()
    if len(dynamic_files) == 0:
        dynamic_files = data["file_id"].unique()[:1]

    file_id = dynamic_files[0]
    ts = data[data["file_id"] == file_id].sort_values("Time_s").reset_index(drop=True)
    fname = ts["filename"].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    for col in EXHAUST_PATH_ORDER:
        if col in ts.columns:
            ax.plot(ts["Time_s"], ts[col], label=col, linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"Thermal Variables Over Time — {fname}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "thermal_timeseries_sample.png"), bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved thermal_timeseries_sample.png")


def plot_nox_vs_thermal(data: pd.DataFrame, thermal_cols: list, emission_cols: list):
    """Scatter plots: each thermal variable vs NOx (engine-out and tailpipe)."""
    n_thermal = len(thermal_cols)
    n_emission = len(emission_cols)
    n_sample = min(SCATTER_SAMPLE_SIZE, len(data))
    sample = data.sample(n=n_sample, random_state=42)

    fig, axes = plt.subplots(
        n_emission, n_thermal, figsize=(4 * n_thermal, 4 * n_emission)
    )
    if n_emission == 1:
        axes = axes[np.newaxis, :]

    for i, em_col in enumerate(emission_cols):
        for j, th_col in enumerate(thermal_cols):
            ax = axes[i, j]
            ax.scatter(
                sample[th_col],
                sample[em_col],
                s=1,
                alpha=0.1,
                color="steelblue",
                rasterized=True,
            )
            # Pearson r
            valid = sample[[th_col, em_col]].dropna()
            if len(valid) > 2:
                r = valid[th_col].corr(valid[em_col])
                ax.text(
                    0.03,
                    0.95,
                    f"r={r:.3f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                )

            ax.set_xlabel(th_col, fontsize=8)
            if j == 0:
                ax.set_ylabel(em_col, fontsize=9)
            ax.tick_params(labelsize=7)

    fig.suptitle("NOx vs. Thermal Variables", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "nox_vs_thermal.png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved nox_vs_thermal.png")


# ──────────────────────────────────────────────────────────────────────────────
# Summary Report
# ──────────────────────────────────────────────────────────────────────────────


def print_summary(pearson_corr, spearman_corr, pca_results, peak_lags):
    """Print a textual summary of key findings."""
    explained, cumulative, loadings, n_90, n_95 = pca_results

    sep = "=" * 72
    print(f"\n{sep}")
    print("THERMAL CORRELATION ANALYSIS — SUMMARY")
    print(sep)

    # Top correlated pairs
    print("\n1. HIGHEST PAIRWISE CORRELATIONS (Pearson)")
    print("-" * 50)
    pairs = []
    cols = pearson_corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], pearson_corr.iloc[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for a, b, r in pairs:
        print(f"  {a:20s} ↔ {b:20s}  r = {r:+.4f}")

    # PCA results
    print(f"\n2. PCA — INDEPENDENT THERMAL DIMENSIONS")
    print("-" * 50)
    for i, (ev, cum) in enumerate(zip(explained, cumulative)):
        print(f"  PC{i+1}: {ev*100:5.1f}% variance  (cumulative: {cum*100:5.1f}%)")
    print(f"\n  Components for 90% variance: {n_90}")
    print(f"  Components for 95% variance: {n_95}")

    # PCA loadings interpretation
    print(f"\n3. PCA LOADINGS (top contributors per component)")
    print("-" * 50)
    for pc in loadings.columns[:n_95]:
        top = loadings[pc].abs().sort_values(ascending=False)
        top_names = [
            f"{name} ({loadings.loc[name, pc]:+.2f})" for name in top.index[:3]
        ]
        print(f"  {pc}: {', '.join(top_names)}")

    # Lag correlation
    if peak_lags:
        print(f"\n4. EXHAUST PATH THERMAL PROPAGATION DELAYS")
        print("-" * 50)
        for col_a, col_b, lag, corr_val in peak_lags:
            print(
                f"  {col_a:20s} → {col_b:20s}  lag = {lag:+5.1f}s  (r = {corr_val:.3f})"
            )

    # Recommendations
    print(f"\n5. RECOMMENDATIONS FOR RL OBSERVATION SPACE")
    print("-" * 50)
    if n_90 <= 2:
        print(f"  • Only {n_90} PC(s) explain 90% of thermal variance.")
        print(f"    Consider representing thermal state with {n_90}-{n_95} variables.")
    else:
        print(
            f"  • {n_90} PCs needed for 90% variance — thermal state is multi-dimensional."
        )

    # Check for highly correlated pairs (r > 0.95)
    redundant = [(a, b, r) for a, b, r in pairs if abs(r) > 0.95]
    if redundant:
        print(f"  • Highly correlated pairs (|r| > 0.95) — candidates for merging:")
        for a, b, r in redundant:
            print(f"      {a} ↔ {b} (r={r:.4f})")

    # Check which thermal var best correlates with NOx_tp
    print(f"\n  • Best thermal predictors for downstream NOx reduction:")
    print(f"    (See nox_vs_thermal.png for scatter plots)")

    print(f"\n{sep}\n")

    # Save summary to text file
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    import io

    buf = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = buf

    print(f"{sep}")
    print("THERMAL CORRELATION ANALYSIS — SUMMARY")
    print(sep)
    print(f"\n1. HIGHEST PAIRWISE CORRELATIONS (Pearson)")
    for a, b, r in pairs:
        print(f"  {a:20s} ↔ {b:20s}  r = {r:+.4f}")
    print(f"\n2. PCA — INDEPENDENT THERMAL DIMENSIONS")
    for i, (ev, cum) in enumerate(zip(explained, cumulative)):
        print(f"  PC{i+1}: {ev*100:5.1f}% variance  (cumulative: {cum*100:5.1f}%)")
    print(f"  Components for 90% variance: {n_90}")
    print(f"  Components for 95% variance: {n_95}")
    print(f"\n3. PCA LOADINGS")
    print(loadings.to_string())
    if peak_lags:
        print(f"\n4. EXHAUST PATH THERMAL PROPAGATION DELAYS")
        for col_a, col_b, lag, corr_val in peak_lags:
            print(
                f"  {col_a:20s} → {col_b:20s}  lag = {lag:+5.1f}s  (r = {corr_val:.3f})"
            )

    sys.stdout = _orig_stdout
    with open(summary_path, "w") as f:
        f.write(buf.getvalue())
    print(f"  Saved summary.txt")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    data = load_all_data(DATA_DIR)

    # Verify thermal columns are present
    available_thermal = [c for c in THERMAL_COLS if c in data.columns]
    available_emission = [c for c in EMISSION_COLS if c in data.columns]
    print(f"Available thermal columns: {available_thermal}")
    print(f"Available emission columns: {available_emission}")

    if len(available_thermal) < 2:
        print("ERROR: Fewer than 2 thermal columns found. Cannot proceed.")
        sys.exit(1)

    # Filter out rows where all thermal values are NaN
    data = data.dropna(subset=available_thermal, how="all")
    print(f"Rows after dropping all-NaN thermal: {len(data):,}\n")

    # 2. Correlation heatmaps
    print("── Correlation Heatmaps ──")
    pearson_corr = plot_correlation_heatmap(
        data,
        available_thermal,
        "Thermal Variable Correlations (Pearson)",
        "corr_heatmap_pearson.png",
        method="pearson",
    )

    spearman_corr = plot_correlation_heatmap(
        data,
        available_thermal,
        "Thermal Variable Correlations (Spearman)",
        "corr_heatmap_spearman.png",
        method="spearman",
    )

    # With emissions
    if available_emission:
        all_cols = available_thermal + available_emission
        plot_correlation_heatmap(
            data,
            all_cols,
            "Thermal + NOx Correlations (Pearson)",
            "corr_heatmap_thermal_nox.png",
            method="pearson",
        )

    # 3. Scatter matrix (thermal only)
    print("\n── Scatter Matrix ──")
    plot_scatter_matrix(
        data,
        available_thermal,
        "Pairwise Scatter — Thermal Variables",
        "scatter_matrix_thermal.png",
    )

    # 4. PCA
    print("\n── PCA Analysis ──")
    pca_results = run_pca_analysis(data, available_thermal)

    # 5. Distributions
    print("\n── Distributions ──")
    plot_distributions(data, available_thermal)

    # 6. Time-lagged cross-correlation
    print("\n── Lag Cross-Correlation ──")
    peak_lags = plot_lag_correlation(data, available_thermal)

    # 7. Per-scenario correlation
    print("\n── Per-Scenario Correlations ──")
    plot_per_scenario_correlation(data, available_thermal)

    # 8. Sample time series
    print("\n── Sample Time Series ──")
    plot_thermal_timeseries_sample(data, available_thermal)

    # 9. NOx vs. thermal scatter
    if available_emission:
        print("\n── NOx vs. Thermal ──")
        plot_nox_vs_thermal(data, available_thermal, available_emission)

    # 10. Summary
    print_summary(pearson_corr, spearman_corr, pca_results, peak_lags)


if __name__ == "__main__":
    main()
