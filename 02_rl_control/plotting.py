import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class TrainingLivePlotCallback(BaseCallback):
    """
    Callback for plotting training progress (Reward) in real-time (saved to file).
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(TrainingLivePlotCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve mean reward from the monitor file or accumulated rewards
            # SB3 Monitor wrapper writes to a csv file. We can read that or just track episode rewards if we want.
            # But SB3 callbacks don't easily give "current episode reward".
            # Easiest is to read the monitor files if they exist.
            try:
                from stable_baselines3.common.results_plotter import load_results, ts2xy

                df = load_results(self.log_dir)
                if len(df) > 0:
                    # ts2xy extracts the timeseries (timesteps and rewards) from the monitor dataframes
                    x, y = ts2xy(df, "timesteps")
                    if len(y) > 0:
                        # Use a moving average for smoother plotting
                        mean_reward = np.mean(y[-100:])  # Last 100 episodes
                        self.rewards.append(mean_reward)
                        self.timesteps.append(self.num_timesteps)

                        self._plot()
            except Exception as e:
                pass  # Ignore errors during plotting

        return True

    def _plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.timesteps, self.rewards, label="Mean Reward (Last 100 Eps)")
        plt.xlabel("Timesteps")
        plt.ylabel("Reward")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, "training_progress.png"))
        plt.close()


def plot_evaluation(results, log_dir, config):
    """
    Plot evaluation results: Speed, SOC, Emissions (NOx, CO).
    """
    time_steps = np.arange(len(results["speed_actual"]))

    has_thermals = "T_gas_eo_K" in results

    fig, axes = plt.subplots(
        4 if has_thermals else 3, 1, figsize=(12, 12), sharex=False
    )

    # 1. Speed
    axes[0].plot(
        time_steps,
        results["speed_target"],
        label="Target Speed",
        color="black",
        alpha=0.9,
        linestyle="--",
    )
    axes[0].plot(
        time_steps,
        results["speed_actual"],
        label="Actual Speed",
        color="blue",
        alpha=0.7,
    )
    axes[0].set_ylabel("Speed (km/h)")
    axes[0].set_title("Speed Tracking")
    axes[0].set_ylim(-10, 160)
    axes[0].legend()
    axes[0].grid(True)

    # 2. SOC
    axes[1].plot(time_steps, results["soc"], label="SOC", color="green")
    axes[1].set_ylabel("SOC")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_title("State of Charge (1 = full, 0 = empty)")
    axes[1].axhline(y=0.2, color="r", linestyle=":", alpha=0.5)
    axes[1].axhline(y=0.9, color="r", linestyle=":", alpha=0.5)
    axes[1].grid(True)

    # 3. NOx
    nox_mg = np.array(results["nox"]) * 0.5 * 1000.0
    axes[2].plot(time_steps, nox_mg, label="NOx (Tailpipe)", color="orange", alpha=0.9)
    axes[2].set_ylabel("NOx (mg)")
    if not has_thermals:
        axes[2].set_xlabel("Time Step (0.5s)")
    axes[2].set_title("NOx Emissions per Step (mg)")
    axes[2].set_ylim(-1, 80)
    axes[2].legend()
    axes[2].grid(True)

    # 4. Thermals
    if has_thermals:
        cfg = config  # imported globally in planning module
        axes[3].plot(
            time_steps,
            results["T_gas_eo_K"],
            label="T Engine Out",
            color="red",
            linestyle="--",
        )
        axes[3].plot(
            time_steps,
            results["T_Sub_DPF_K"],
            label="T DPF Substrate",
            color="darkorange",
            linestyle="-",
        )
        axes[3].plot(
            time_steps,
            results["T_gas_tp_K"],
            label="T Tailpipe",
            color="gold",
            linestyle=":",
        )
        axes[3].axhline(
            y=cfg.SCR_LIGHTOFF_K,
            color="black",
            linestyle="--",
            label=f"SCR Light-off ({cfg.SCR_LIGHTOFF_K}K)",
            linewidth=1,
        )
        axes[3].set_ylabel("Temp (K)")
        axes[3].set_xlabel("Time Step (0.5s)")
        axes[3].set_title("Aftertreatment Thermodynamics")
        axes[3].legend()
        axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "evaluation_results.png"))
    plt.close()
    print(
        f"Evaluation plots saved to {os.path.join(log_dir, 'evaluation_results.png')}"
    )


def plot_actions(results, log_dir, window_start=300, window_size=30):
    """
    Plot control actions [ICE Speed, EM2 Torque, Fuel, Brake] for a specific time window.
    Default window: 300s to 330s.
    """
    # Slice the data
    start_idx = window_start
    end_idx = window_start + window_size

    # Ensure indices are within bounds
    total_len = len(results["speed_actual"])
    if start_idx >= total_len:
        print(
            f"Window start {start_idx} is beyond data length {total_len}. Plotting last {window_size} steps."
        )
        start_idx = max(0, total_len - window_size)
        end_idx = total_len

    end_idx = min(end_idx, total_len)

    time_steps = np.arange(start_idx, end_idx)

    # Extract data slices
    engine_on = results.get("engine_on", [False] * total_len)[start_idx:end_idx]
    ice_speed = results["ice_speed_rpm"][start_idx:end_idx]
    em2_torque = results["em2_torque_nm"][start_idx:end_idx]
    fuel = results["fuel"][start_idx:end_idx]
    brake = results["brake_perc"][start_idx:end_idx]

    fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=False)

    # 1. Engine State
    axes[0].step(time_steps, engine_on, label="Engine On", color="purple", where="mid")
    axes[0].set_ylabel("Boolean")
    axes[0].set_title("ICE Engine State")
    axes[0].set_ylim(-0.2, 1.2)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Off", "On"])
    axes[0].grid(True)
    axes[0].legend()

    # 2. ICE Speed
    axes[1].plot(time_steps, ice_speed, label="ICE Speed (RPM)", color="blue")
    axes[1].set_ylabel("RPM")
    axes[1].set_title("ICE Speed")
    axes[1].grid(True)
    axes[1].legend()

    # 3. EM2 Torque
    axes[2].plot(time_steps, em2_torque, label="EM2 Torque (Nm)", color="green")
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_title("EM2 Torque")
    axes[2].grid(True)
    axes[2].legend()

    # 4. Fuel Injection
    axes[3].plot(time_steps, fuel, label="Fuel Injection (mg)", color="orange")
    axes[3].set_ylabel("Fuel (mg)")
    axes[3].set_title("Fuel Injection per Step")
    axes[3].grid(True)
    axes[3].legend()

    # 5. Brake
    axes[4].plot(time_steps, brake, label="Brake (%)", color="red")
    axes[4].set_ylabel("Brake (%)")
    axes[4].set_xlabel("Time Step (s)")
    axes[4].set_title("Brake Command")
    axes[4].set_ylim(-5.0, 105.0)
    axes[4].grid(True)
    axes[4].legend()

    plt.tight_layout()
    save_path = os.path.join(log_dir, "action_results.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Action plots saved to {save_path}")


# ============================================================
# Exploration / Exploitation Analysis
# ============================================================


def _compute_state_entropy(obs_array: np.ndarray, bins: int = 20) -> float:
    """Mean Shannon entropy of the discretized state distribution over all dims."""
    if obs_array.ndim == 1:
        obs_array = obs_array[:, np.newaxis]
    entropies = []
    for d in range(obs_array.shape[1]):
        col = obs_array[:, d]
        counts, _ = np.histogram(col, bins=bins)
        probs = counts / (counts.sum() + 1e-12)
        probs = probs[probs > 0]
        entropies.append(-np.sum(probs * np.log(probs)))
    return float(np.mean(entropies))


class ExplorationEntropyCallback(BaseCallback):
    """
    Tracks per-episode state visitation entropy during training.

    Entropy is computed on the (normalised) observations collected within each
    episode.  A decreasing trend indicates that the policy is converging and
    exploiting a narrower region of the state space.  Calls
    plot_exploration_entropy every ``plot_freq`` completed episodes and once
    more when training ends.

    Args:
        plot_freq: episodes between plot updates.
        log_dir:   directory where plots are saved.
        bins:      bins per state dimension for entropy computation.
        verbose:   SB3 verbosity level.
    """

    def __init__(
        self,
        plot_freq: int,
        log_dir: str,
        bins: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.log_dir = log_dir
        self.bins = bins
        self._episode_obs: list = []
        self.entropy_history: list = []
        self.timestep_history: list = []

    def _on_step(self) -> bool:
        new_obs = self.locals.get("new_obs")
        if new_obs is not None:
            self._episode_obs.append(np.asarray(new_obs[0], dtype=float))

        dones = self.locals.get("dones", [False])
        if dones[0] and len(self._episode_obs) > 1:
            obs_array = np.stack(self._episode_obs)
            self.entropy_history.append(_compute_state_entropy(obs_array, self.bins))
            self.timestep_history.append(self.num_timesteps)
            self._episode_obs = []

            if len(self.entropy_history) % self.plot_freq == 0:
                plot_exploration_entropy(
                    self.entropy_history,
                    self.timestep_history,
                    self.log_dir,
                )
        return True

    def _on_training_end(self) -> None:
        plot_exploration_entropy(
            self.entropy_history,
            self.timestep_history,
            self.log_dir,
        )


def plot_state_visitation_1d(
    results_list,
    labels,
    log_dir: str,
    bins: int = 40,
) -> None:
    """
    Marginal stationary distribution d^π(s_i) as normalised histograms for
    each state dimension.  Multiple agents / runs can be overlaid for direct
    comparison.

    Args:
        results_list: list of eval_results dicts, or a single dict.
        labels:       list of agent/run names (one per dict).
        log_dir:      directory where the figure is saved.
        bins:         histogram bins per dimension.
    """
    if not isinstance(results_list, list):
        results_list = [results_list]
    if isinstance(labels, str):
        labels = [labels]
    labels = (
        list(labels) if labels else [f"Agent {i + 1}" for i in range(len(results_list))]
    )

    # Derive speed_error (raw obs dim 1 = target − actual)
    augmented = []
    for r in results_list:
        er = dict(r)
        er["speed_error"] = (
            np.array(r["speed_target"]) - np.array(r["speed_actual"])
        ).tolist()
        augmented.append(er)

    state_configs = [
        ("speed_actual", "Speed (km/h)", (0, 150)),
        ("speed_error", "Speed Error (km/h)", (-80, 80)),
        ("soc", "SOC", (0, 1)),
        ("ice_torque", "ICE Torque (Nm)", (-50, 300)),
        ("nox", "NOx (g/s)", (0, 10)),
        ("engine_on", "Engine On (0/1)", (-0.1, 1.1)),
    ]
    if "T_gas_eo_K" in results_list[0]:
        state_configs += [
            ("T_gas_eo_K", "T Eng-Out (K)", (290, 800)),
            ("T_Sub_DPF_K", "T DPF Substrate (K)", (290, 650)),
            ("T_gas_tp_K", "T Tailpipe (K)", (290, 600)),
        ]

    ncols = 3
    nrows = (len(state_configs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(results_list), 1)))

    for ax_idx, (key, xlabel, xlim) in enumerate(state_configs):
        ax = axes[ax_idx]
        for i, (r, lbl) in enumerate(zip(augmented, labels)):
            if key in r:
                data = np.array(r[key], dtype=float)
                ax.hist(
                    data,
                    bins=bins,
                    density=True,
                    alpha=0.6,
                    color=colors[i],
                    label=lbl,
                    range=(xlim[0], xlim[1]),
                )
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"d\u03c0({key})", fontsize=10)
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)

    if len(results_list) > 1:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.6)
            for i in range(len(labels))
        ]
        fig.legend(handles, labels, loc="lower right", fontsize=9, title="Agents")

    for ax_idx in range(len(state_configs), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("State Stationary Distribution  d\u03c0(s)", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(log_dir, "state_visitation_1d.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"State visitation (1D) saved to {save_path}")


def plot_state_visitation_2d(results, log_dir: str, bins: int = 40) -> None:
    """
    2D joint state occupancy heatmaps ρ^π(s_i, s_j) for key state pairs.

    Each cell shows the fraction of episode timesteps spent in that
    (s_i, s_j) bin, highlighting which regions of the joint state space are
    visited and which remain unexplored.

    Args:
        results: eval_results dict from an evaluation run.
        log_dir: directory where the figure is saved.
        bins:    bins per axis for the 2D histogram.
    """
    speed = np.array(results["speed_actual"])
    speed_err = np.array(results["speed_target"]) - speed
    soc = np.array(results["soc"])
    ice_torque = np.array(results["ice_torque"])
    nox = np.array(results["nox"])

    pairs = [
        (speed, soc, "Speed (km/h)", "SOC", (0, 150), (0, 1)),
        (speed, nox, "Speed (km/h)", "NOx (g/s)", (0, 150), (0, 10)),
        (speed_err, soc, "Speed Error (km/h)", "SOC", (-80, 80), (0, 1)),
        (ice_torque, nox, "ICE Torque (Nm)", "NOx (g/s)", (-50, 300), (0, 10)),
        (soc, ice_torque, "SOC", "ICE Torque (Nm)", (0, 1), (-50, 300)),
        (speed_err, nox, "Speed Error (km/h)", "NOx (g/s)", (-80, 80), (0, 10)),
    ]
    if "T_gas_eo_K" in results:
        T_eo = np.array(results["T_gas_eo_K"])
        T_dpf = np.array(results["T_Sub_DPF_K"])
        pairs += [
            (speed, T_eo, "Speed (km/h)", "T Eng-Out (K)", (0, 150), (290, 800)),
            (T_eo, nox, "T Eng-Out (K)", "NOx (g/s)", (290, 800), (0, 10)),
            (soc, T_dpf, "SOC", "T DPF Sub (K)", (0, 1), (290, 650)),
        ]

    ncols = 3
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for ax_idx, (x, y, xl, yl, xr, yr) in enumerate(pairs):
        ax = axes[ax_idx]
        h, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr])
        h_norm = h / (h.sum() + 1e-12)
        im = ax.imshow(
            h_norm.T,
            origin="lower",
            aspect="auto",
            extent=[xe[0], xe[-1], ye[0], ye[-1]],
            cmap="hot_r",
            vmin=0,
        )
        plt.colorbar(im, ax=ax, label="Visit Fraction")
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(
            f"\u03c1\u03c0({xl.split('(')[0].strip()}, {yl.split('(')[0].strip()})",
            fontsize=10,
        )

    for ax_idx in range(len(pairs), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("2D State Occupancy Measure  \u03c1\u03c0(s_i, s_j)", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(log_dir, "state_visitation_2d.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"State visitation (2D) saved to {save_path}")


def plot_action_distribution(
    results_list,
    labels,
    log_dir: str,
    bins: int = 40,
) -> None:
    """
    Marginal action distribution π(a_i) for every action dimension.

    Concentrated histograms indicate exploitation; broad histograms indicate
    exploratory behaviour.

    Args:
        results_list: list of eval_results dicts, or a single dict.
        labels:       list of agent/run names (one per dict).
        log_dir:      directory where the figure is saved.
        bins:         histogram bins per dimension.
    """
    if not isinstance(results_list, list):
        results_list = [results_list]
    if isinstance(labels, str):
        labels = [labels]
    labels = (
        list(labels) if labels else [f"Agent {i + 1}" for i in range(len(results_list))]
    )

    action_configs = [
        ("engine_on", "Engine On (0/1)", (-0.1, 1.1)),
        ("ice_speed_rpm", "ICE Speed (RPM)", (0, 4500)),
        ("em2_torque_nm", "EM2 Torque (Nm)", (-450, 450)),
        ("fuel", "Fuel (mg)", (0, 80)),
        ("brake_perc", "Brake (%)", (0, 105)),
    ]

    ncols = 3
    nrows = (len(action_configs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(results_list), 1)))

    for ax_idx, (key, xlabel, xlim) in enumerate(action_configs):
        ax = axes[ax_idx]
        for i, (r, lbl) in enumerate(zip(results_list, labels)):
            data = np.array(r.get(key, []), dtype=float)
            if len(data) > 0:
                ax.hist(
                    data,
                    bins=bins,
                    density=True,
                    alpha=0.6,
                    color=colors[i],
                    label=lbl,
                    range=(xlim[0], xlim[1]),
                )
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"\u03c0({key})", fontsize=10)
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)

    if len(results_list) > 1:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.6)
            for i in range(len(labels))
        ]
        fig.legend(handles, labels, loc="lower right", fontsize=9, title="Agents")

    for ax_idx in range(len(action_configs), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("Action Distribution  \u03c0(a)", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(log_dir, "action_distribution.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Action distribution saved to {save_path}")


def plot_state_action_occupancy(results, log_dir: str, bins: int = 40) -> None:
    """
    2D occupancy measure ρ^π(s_i, a_j) for key state-action pairs.

    Shows which (state, action) combinations are actually visited, revealing
    the conditional action strategy encoded in the learned policy.

    Args:
        results: eval_results dict from an evaluation run.
        log_dir: directory where the figure is saved.
        bins:    bins per axis for the 2D histogram.
    """
    speed = np.array(results["speed_actual"])
    speed_err = np.array(results["speed_target"]) - speed
    soc = np.array(results["soc"])
    nox = np.array(results["nox"])
    ice_torque = np.array(results["ice_torque"])
    ice_speed = np.array(results["ice_speed_rpm"])
    em2_torque = np.array(results["em2_torque_nm"])
    fuel = np.array(results["fuel"])
    brake = np.array(results["brake_perc"])

    pairs = [
        (soc, em2_torque, "SOC", "EM2 Torque (Nm)", (0, 1), (-450, 450)),
        (speed, ice_speed, "Speed (km/h)", "ICE Speed (RPM)", (0, 150), (0, 4500)),
        (speed_err, fuel, "Speed Error (km/h)", "Fuel (mg)", (-80, 80), (0, 80)),
        (nox, fuel, "NOx (g/s)", "Fuel (mg)", (0, 10), (0, 80)),
        (
            ice_torque,
            em2_torque,
            "ICE Torque (Nm)",
            "EM2 Torque (Nm)",
            (-50, 300),
            (-450, 450),
        ),
        (speed, brake, "Speed (km/h)", "Brake (%)", (0, 150), (0, 105)),
    ]

    ncols = 3
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for ax_idx, (x, y, xl, yl, xr, yr) in enumerate(pairs):
        ax = axes[ax_idx]
        h, xe, ye = np.histogram2d(x, y, bins=bins, range=[xr, yr])
        h_norm = h / (h.sum() + 1e-12)
        im = ax.imshow(
            h_norm.T,
            origin="lower",
            aspect="auto",
            extent=[xe[0], xe[-1], ye[0], ye[-1]],
            cmap="viridis",
            vmin=0,
        )
        plt.colorbar(im, ax=ax, label="Visit Fraction")
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(
            f"\u03c1\u03c0({xl.split('(')[0].strip()}, {yl.split('(')[0].strip()})",
            fontsize=10,
        )

    for ax_idx in range(len(pairs), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("State-Action Occupancy Measure  \u03c1\u03c0(s, a)", fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(log_dir, "state_action_occupancy.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"State-action occupancy saved to {save_path}")


def plot_temporal_state_heatmap(
    results,
    log_dir: str,
    n_time_bins: int = 60,
    n_val_bins: int = 40,
) -> None:
    """
    Temporal occupancy heatmap: conditional state density over episode time.

    Each column represents a time window of the episode; the colour shows the
    conditional distribution of state values within that window.  This reveals
    how the agent traverses the state space over the course of an episode —
    including stuck regions, oscillations, or systematic drifts.

    Args:
        results:      eval_results dict from an evaluation run.
        log_dir:      directory where the figure is saved.
        n_time_bins:  bins for the normalised time axis [0, 1].
        n_val_bins:   bins for the state-value axis.
    """
    speed = np.array(results["speed_actual"])
    speed_err = np.array(results["speed_target"]) - speed
    soc = np.array(results["soc"])
    ice_torque = np.array(results["ice_torque"])
    nox = np.array(results["nox"])

    T = len(speed)
    t_norm = np.linspace(0, 1, T)

    state_configs = [
        (speed, "Speed (km/h)", (0, 150)),
        (speed_err, "Speed Error (km/h)", (-80, 80)),
        (soc, "SOC", (0, 1)),
        (ice_torque, "ICE Torque (Nm)", (-50, 300)),
        (nox, "NOx (g/s)", (0, 10)),
    ]
    if "T_gas_eo_K" in results:
        T_eo = np.array(results["T_gas_eo_K"])
        T_dpf = np.array(results["T_Sub_DPF_K"])
        T_tp = np.array(results["T_gas_tp_K"])
        state_configs += [
            (T_eo, "T Eng-Out (K)", (290, 800)),
            (T_dpf, "T DPF Substrate (K)", (290, 650)),
            (T_tp, "T Tailpipe (K)", (290, 600)),
        ]

    ncols = 2
    nrows = (len(state_configs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for ax_idx, (data, ylabel, val_range) in enumerate(state_configs):
        ax = axes[ax_idx]
        h, xe, ye = np.histogram2d(
            t_norm,
            data,
            bins=[n_time_bins, n_val_bins],
            range=[[0, 1], val_range],
        )
        # Normalise each time column → conditional density p(s | t)
        col_sums = h.sum(axis=1, keepdims=True)
        h_cond = np.where(col_sums > 0, h / (col_sums + 1e-12), 0.0)
        im = ax.imshow(
            h_cond.T,
            origin="lower",
            aspect="auto",
            extent=[0, 1, ye[0], ye[-1]],
            cmap="YlOrRd",
            vmin=0,
            vmax=h_cond.max() or 1.0,
        )
        plt.colorbar(im, ax=ax, label="Cond. Density")
        ax.set_xlabel("Normalised Episode Time", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            f"Temporal Occupancy: {ylabel.split('(')[0].strip()}",
            fontsize=10,
        )

    for ax_idx in range(len(state_configs), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        "Temporal State Heatmap  (Conditional Density over Episode)",
        fontsize=13,
    )
    plt.tight_layout()
    save_path = os.path.join(log_dir, "temporal_state_heatmap.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Temporal state heatmap saved to {save_path}")


def plot_exploration_entropy(
    entropy_history,
    timestep_history,
    log_dir: str,
) -> None:
    """
    Shannon entropy of the state visitation distribution vs. training timesteps.

    High entropy signals broad exploration; a downward trend indicates that the
    policy is converging and exploiting a narrower region of state space.

    Args:
        entropy_history:  list of per-episode mean state entropy values.
        timestep_history: corresponding training timestep values.
        log_dir:          directory where the figure is saved.
    """
    if len(entropy_history) < 2:
        return

    entropy = np.array(entropy_history)
    timesteps = np.array(timestep_history)

    window = max(1, len(entropy) // 20)
    smoothed = np.convolve(entropy, np.ones(window) / window, mode="valid")
    t_smooth = timesteps[window - 1 :]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        timesteps, entropy, alpha=0.25, color="steelblue", label="Per-episode entropy"
    )
    ax.plot(
        t_smooth,
        smoothed,
        color="steelblue",
        linewidth=2,
        label=f"Moving avg (window={window})",
    )
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean State Entropy (nats)")
    ax.set_title(
        "Exploration Entropy over Training\n"
        "(Higher \u2192 more exploration · Lower \u2192 convergence / exploitation)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(log_dir, "exploration_entropy.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Exploration entropy saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate state-space exploration/exploitation plots from one or more "
            "evaluation_data.csv files produced by eval_*.py.\n\n"
            "Single agent:   python plotting.py path/to/evaluation_data.csv\n"
            "Multi-agent:    python plotting.py td3.csv sac.csv --labels TD3 SAC"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        help="Path(s) to evaluation_data.csv.  Pass multiple files to overlay agents.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Agent label(s) matching each CSV (default: Agent 1, Agent 2, …).",
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Output directory for plots (default: directory of the first CSV).",
    )
    _args = parser.parse_args()

    _log_dir = _args.log_dir or os.path.dirname(os.path.abspath(_args.csv_paths[0]))
    os.makedirs(_log_dir, exist_ok=True)

    _labels = _args.labels or [f"Agent {i + 1}" for i in range(len(_args.csv_paths))]

    _results_list = []
    for _csv in _args.csv_paths:
        _df = pd.read_csv(_csv)
        _results_list.append({col: _df[col].tolist() for col in _df.columns})
    print(f"Loaded {len(_results_list)} result set(s) → saving plots to {_log_dir}")

    plot_state_visitation_1d(_results_list, _labels, _log_dir)
    plot_state_visitation_2d(_results_list[0], _log_dir)
    plot_action_distribution(_results_list, _labels, _log_dir)
    plot_state_action_occupancy(_results_list[0], _log_dir)
    plot_temporal_state_heatmap(_results_list[0], _log_dir)
    print("Done.  All exploration/exploitation plots generated.")
