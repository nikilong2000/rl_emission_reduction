import matplotlib.pyplot as plt
import numpy as np

# ── Style aligned with 02_rl_control/ plots ───────────────────────────────────
# Default matplotlib font (no family override), project blue + light grey.
UPC_COLOR = "#3578bb"  # strong blue (matches plot_seeds MEAN_COLOR)
OLD_COLOR = "#868A94"  #
FIG_DPI = 200
BAR_W = 0.22  # slim bars -> less visual weight
PAD = 0.55  # horizontal breathing room around bar groups
GRID_ALPHA = 0.25


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, axis="y", alpha=GRID_ALPHA, linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _labels(ax, bars, fmt):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )


# ── Scalers ───────────────────────────────────────────────────────────────────
# Raw values (µs) per scaler; summed to give total scaler overhead per vehicle.
scalers = {
    "ICE_in": (18.7, 1.95),
    "ICE_out": (18.9, 1.94),
    "ICE_inv_out": (12.4, 2.03),
    "PG_in": (19.0, 1.94),
    "PG_out": (19.0, 1.96),
    "PG_inv_out": (12.8, 2.06),
}

ice_sk = sum(v[0] for k, v in scalers.items() if k.startswith("ICE"))
ice_upc = sum(v[1] for k, v in scalers.items() if k.startswith("ICE"))
pg_sk = sum(v[0] for k, v in scalers.items() if k.startswith("PG"))
pg_upc = sum(v[1] for k, v in scalers.items() if k.startswith("PG"))

labels_s = ["ICE", "PG"]
old_vals = [ice_sk, pg_sk]
upc_vals = [ice_upc, pg_upc]
x = np.arange(len(labels_s))

fig, ax = plt.subplots(figsize=(5.5, 4))
fig.patch.set_facecolor("white")
b_old = ax.bar(
    x - BAR_W / 2, old_vals, BAR_W, label="Scikit-Learn", color=OLD_COLOR, zorder=3
)
b_upc = ax.bar(
    x + BAR_W / 2, upc_vals, BAR_W, label="CTTC-UPC", color=UPC_COLOR, zorder=3
)
_labels(ax, b_old, lambda h: f"{h:.1f}")
_labels(ax, b_upc, lambda h: f"{h:.1f}")
ax.set_xticks(x)
ax.set_xticklabels(labels_s, fontsize=10)
ax.set_xlim(-PAD, len(labels_s) - 1 + PAD)
_style(ax, "Scalers Total Inference Time", r"Total wall time [$\mu$s]")
ax.set_ylim(0, max(old_vals) * 1.18)
ax.legend(fontsize=9, framealpha=0.95)
fig.tight_layout()
fig.savefig("CTTC/scalers_speedup_clean.png", dpi=FIG_DPI, bbox_inches="tight")
print(
    f"ICE  — sklearn={ice_sk:.1f} µs  upc={ice_upc:.2f} µs  speedup×{ice_sk/ice_upc:.1f}"
)
print(f"PG   — sklearn={pg_sk:.1f} µs  upc={pg_upc:.2f} µs  speedup×{pg_sk/pg_upc:.1f}")

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "ICE init": (459, 243),
    "ICE main": (315, 147),
    "PG init": (455, 240),
    "PG main": (316, 146),
}

labels_m = list(models.keys())
old_m = [v[0] for v in models.values()]
upc_m = [v[1] for v in models.values()]
x2 = np.arange(len(labels_m))

fig2, ax2 = plt.subplots(figsize=(6.5, 4))
fig2.patch.set_facecolor("white")
b_old2 = ax2.bar(
    x2 - BAR_W / 2, old_m, BAR_W, label="TensorFlow", color=OLD_COLOR, zorder=3
)
b_upc2 = ax2.bar(
    x2 + BAR_W / 2, upc_m, BAR_W, label="CTTC-UPC", color=UPC_COLOR, zorder=3
)
_labels(ax2, b_old2, lambda h: f"{int(h)}")
_labels(ax2, b_upc2, lambda h: f"{int(h)}")
ax2.set_xticks(x2)
ax2.set_xticklabels(labels_m, fontsize=10)
ax2.set_xlim(-PAD, len(labels_m) - 1 + PAD)
_style(ax2, "Model Inference Time", r"Wall time [$\mu$s]")
ax2.set_ylim(0, max(old_m) * 1.15)
ax2.legend(fontsize=9, framealpha=0.95)
fig2.tight_layout()
fig2.savefig("CTTC/models_speedup_clean.png", dpi=FIG_DPI, bbox_inches="tight")
print("\nModels:")
for name, (tf, upc) in models.items():
    print(f"  {name}: TF={tf} µs  upc={upc} µs  speedup×{tf/upc:.2f}")

# ── WLTC full-cycle runtime ───────────────────────────────────────────────────
# Total wall time to roll out the full WLTC cycle (3601 steps): old TF 2.15
# stack vs the new TF 2.18 / CTTC stack. New stack -> blue, old -> grey.
wltc_labels = ["TF 2.15", "TF 2.18"]
wltc_vals = [104.50, 4.79]
wltc_colors = [OLD_COLOR, OLD_COLOR]
xw = np.arange(len(wltc_labels))

fig3, ax3 = plt.subplots(figsize=(5.5, 4))
fig3.patch.set_facecolor("white")
b_w = ax3.bar(xw, wltc_vals, BAR_W * 2, color=wltc_colors, zorder=3)
_labels(ax3, b_w, lambda h: f"{h:.2f}")
ax3.set_xticks(xw)
ax3.set_xticklabels(wltc_labels, fontsize=10)
ax3.set_xlim(-PAD, len(wltc_labels) - 1 + PAD)
_style(ax3, "WLTC Runtime Comparison", "Runtime over full WLTC cycle [s]")
ax3.set_ylim(0, max(wltc_vals) * 1.12)
fig3.tight_layout()
fig3.savefig("CTTC/WLTC_runtime_clean.pdf", bbox_inches="tight")
fig3.savefig("CTTC/WLTC_runtime_clean.png", dpi=FIG_DPI, bbox_inches="tight")
print(
    f"\nWLTC runtime — TF 2.15={wltc_vals[0]:.2f} s  TF 2.18={wltc_vals[1]:.2f} s  speedup×{wltc_vals[0]/wltc_vals[1]:.1f}"
)

# ── Standardised six-phase target-velocity profile ────────────────────────────
# Eval-mode target-speed schedule from env.py: six 600-step segments (3600 total).
SEGMENT_LEN = 600
schedule = [50.0, 70.0, 110.0, 140.0, 80.0, 35.0]
max_steps = SEGMENT_LEN * len(schedule)

# Build the step trajectory (each segment held flat for SEGMENT_LEN steps).
steps = np.arange(max_steps + 1)
target = np.array([schedule[min(s // SEGMENT_LEN, len(schedule) - 1)] for s in steps])

fig4, ax4 = plt.subplots(figsize=(8, 4))
fig4.patch.set_facecolor("white")
ax4.step(steps, target, where="post", color=UPC_COLOR, lw=1.6, zorder=3)
for i, spd in enumerate(schedule):
    ax4.annotate(
        f"{spd:.0f}",
        xy=(i * SEGMENT_LEN + SEGMENT_LEN / 2, spd),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
ax4.set_xlim(0, max_steps)
ax4.set_ylim(0, max(schedule) * 1.12)
_style(ax4, "Standardised Six-Phase Target-Velocity Profile", "Target velocity [km/h]")
ax4.set_xlabel("Simulation step", fontsize=8)
fig4.tight_layout()
fig4.savefig("CTTC/eval_target_profile.pdf", bbox_inches="tight")
fig4.savefig("CTTC/eval_target_profile.png", dpi=FIG_DPI, bbox_inches="tight")
print(f"\nTarget profile — {len(schedule)} phases × {SEGMENT_LEN} steps = {max_steps} steps")
