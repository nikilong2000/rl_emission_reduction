"""
Manual controller for interacting with EmissionControlEnv without an RL policy.

Features:
- 4 action sliders in normalized agent space [-1, 1]
- Manual one-step inference trigger
- Auto-run at 1 Hz (start/stop)
- Live status panel and trend plots
- CSV export for step history

Usage examples:
    python 02_rl_control/manual_controller.py
    python 02_rl_control/manual_controller.py --algorithm sac
    python 02_rl_control/manual_controller.py --dataset_path 02_rl_control/data_train/WLTC.csv
    python 02_rl_control/manual_controller.py --fixed_target_speed 100
"""

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))
models_dir = os.path.join(current_dir, "models")
sys.path.append(models_dir)

from env import EmissionControlEnv
from utils.config_utils import load_config


@dataclass
class RuntimeOptions:
    algorithm: str = "ppo"
    random_target: bool = True
    fixed_target_speed: float | None = None
    dataset_path: str | None = None


class SimulatorCore:
    """Headless simulator layer around EmissionControlEnv."""

    def __init__(self, options: RuntimeOptions):
        self.options = options
        self.history: list[dict] = []
        self.step_index = 0
        self.last_obs = None
        self.last_info = None
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False

        self.config = load_config(current_dir=models_dir, algo_key=options.algorithm)

        self.env = EmissionControlEnv(
            config_module=self.config,
            random_target=options.random_target,
            fixed_target_speed=options.fixed_target_speed,
            dataset_path=options.dataset_path,
            eval_mode=options.fixed_target_speed is not None,
        )

    def reset(self):
        self.history.clear()
        self.step_index = 0

        obs, info = self.env.reset()
        self.last_obs = obs
        self.last_info = info
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        return obs, info

    def _rescale_action(self, action_norm: np.ndarray) -> np.ndarray:
        return self.env.action_min + (action_norm + 1.0) * 0.5 * (
            self.env.action_max - self.env.action_min
        )

    def step(self, action_norm: np.ndarray):
        action_norm = np.asarray(action_norm, dtype=np.float32).reshape(-1)
        if action_norm.shape[0] != 4:
            raise ValueError(f"Action must have shape (4,), got {action_norm.shape}")

        action_norm = np.clip(action_norm, -1.0, 1.0)
        action_scaled = self._rescale_action(action_norm)

        obs, reward, terminated, truncated, info = self.env.step(action_norm)

        record = {
            "step": int(self.step_index),
            "time_s": float(info.get("time_s", self.step_index)),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "target_speed": float(info.get("target_speed", np.nan)),
            "speed_error": float(info.get("speed_error", np.nan)),
            "speed_actual": float(obs[0]),
            "soc": float(obs[2]),
            "ice_torque": float(obs[3]),
            "nox": float(info.get("nox", obs[4])),
            "fuel": float(info.get("fuel", np.nan)),
            "engine_on": bool(info.get("engine_on", False)),
            "ice_speed_rpm": float(info.get("ice_speed_rpm", np.nan)),
            "em2_torque_nm": float(info.get("em2_torque_nm", np.nan)),
            "brake_perc": float(info.get("brake_perc", np.nan)),
            "switch_timer_steps": int(info.get("steps_since_last_engine_switch", 0)),
            "switch_timer_norm": float(
                min(float(info.get("steps_since_last_engine_switch", 0)) / 6.0, 1.0)
            ),
            "action_ice_cmd_norm": float(action_norm[0]),
            "action_em2_norm": float(action_norm[1]),
            "action_fuel_norm": float(action_norm[2]),
            "action_brake_norm": float(action_norm[3]),
            "action_ice_cmd_scaled": float(action_scaled[0]),
            "action_em2_scaled": float(action_scaled[1]),
            "action_fuel_scaled": float(action_scaled[2]),
            "action_brake_scaled": float(action_scaled[3]),
        }

        self.history.append(record)
        self.last_obs = obs
        self.last_info = info
        self.last_reward = float(reward)
        self.last_terminated = bool(terminated)
        self.last_truncated = bool(truncated)
        self.step_index += 1

        return obs, reward, terminated, truncated, info, record

    def export_csv(self, path: str):
        if not self.history:
            raise ValueError(
                "No history available. Run at least one step before export."
            )

        df = pd.DataFrame(self.history)
        df.to_csv(path, index=False)

    def latest_snapshot(self) -> dict:
        if self.last_obs is None:
            return {}

        return {
            "speed_actual": float(self.last_obs[0]),
            "speed_error_obs": float(self.last_obs[1]),
            "soc": float(self.last_obs[2]),
            "ice_torque": float(self.last_obs[3]),
            "nox_obs": float(self.last_obs[4]),
            "ice_speed_obs": float(self.last_obs[5]),
            "fuel_obs": float(self.last_obs[6]),
            "soc_error": float(self.last_obs[7]),
            "switch_timer_norm_obs": (
                float(self.last_obs[8]) if len(self.last_obs) > 8 else np.nan
            ),
            "reward": float(self.last_reward),
            "terminated": bool(self.last_terminated),
            "truncated": bool(self.last_truncated),
            "target_speed": (
                float(self.last_info.get("target_speed", np.nan))
                if self.last_info
                else np.nan
            ),
            "engine_on": (
                bool(self.last_info.get("engine_on", False))
                if self.last_info
                else False
            ),
            "switch_timer_steps": (
                int(self.last_info.get("steps_since_last_engine_switch", 0))
                if self.last_info
                else 0
            ),
            "time_s": (
                float(self.last_info.get("time_s", self.step_index))
                if self.last_info
                else float(self.step_index)
            ),
        }


class ManualControllerApp:
    """Tkinter UI for manual action control and simulation playback."""

    ACTION_LABELS = [
        "ICE command (-1..1)",
        "EM2 torque command (-1..1)",
        "Fuel command (-1..1)",
        "Brake command (-1..1)",
    ]

    def __init__(self, root: tk.Tk, core: SimulatorCore):
        self.root = root
        self.core = core

        self.auto_running = False
        self.is_stepping = False
        self.auto_job_id = None

        self.slider_vars = [tk.DoubleVar(value=0.0) for _ in range(4)]
        self.slider_value_labels = []

        self.status_vars = {
            "step": tk.StringVar(value="0"),
            "time_s": tk.StringVar(value="0.0"),
            "reward": tk.StringVar(value="0.0000"),
            "speed": tk.StringVar(value="0.00"),
            "target_speed": tk.StringVar(value="0.00"),
            "speed_error": tk.StringVar(value="0.00"),
            "soc": tk.StringVar(value="0.0000"),
            "nox": tk.StringVar(value="0.0000"),
            "fuel": tk.StringVar(value="0.00"),
            "engine_on": tk.StringVar(value="False"),
            "ice_speed": tk.StringVar(value="0.0"),
            "switch_timer": tk.StringVar(value="0"),
            "episode_state": tk.StringVar(value="running"),
        }

        self._build_ui()
        self._bind_keys()

        self._init_plot()
        self._reset_env(initial=True)

    def _build_ui(self):
        self.root.title("Manual RL Plant Controller")
        self.root.geometry("1420x920")

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        right = ttk.Frame(outer)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        slider_box = ttk.LabelFrame(left, text="Actions (Normalized)", padding=10)
        slider_box.pack(fill=tk.X, pady=(0, 10))

        for idx, label in enumerate(self.ACTION_LABELS):
            row = ttk.Frame(slider_box)
            row.pack(fill=tk.X, pady=5)

            ttk.Label(row, text=label, width=29).pack(side=tk.LEFT)
            slider = ttk.Scale(
                row,
                from_=-1.0,
                to=1.0,
                orient=tk.HORIZONTAL,
                variable=self.slider_vars[idx],
                command=lambda _v, i=idx: self._on_slider_change(i),
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))

            val_label = ttk.Label(row, width=8, text="0.00")
            val_label.pack(side=tk.RIGHT)
            self.slider_value_labels.append(val_label)

        self._refresh_slider_labels()

        button_box = ttk.LabelFrame(left, text="Control", padding=10)
        button_box.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(button_box, text="Step once", command=self.step_once).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(button_box, text="Start 1 Hz", command=self.start_auto).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(button_box, text="Stop", command=self.stop_auto).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(button_box, text="Reset episode", command=self._reset_env).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(button_box, text="Export CSV", command=self.export_csv).pack(
            fill=tk.X, pady=2
        )

        presets_box = ttk.LabelFrame(left, text="Presets", padding=10)
        presets_box.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            presets_box,
            text="All zero",
            command=lambda: self._set_action_preset([0.0, 0.0, 0.0, 0.0]),
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            presets_box,
            text="Engine cruising",
            command=lambda: self._set_action_preset([0.45, 0.10, 0.20, -0.60]),
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            presets_box,
            text="Aggressive accel",
            command=lambda: self._set_action_preset([0.90, 0.75, 0.75, -1.0]),
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            presets_box,
            text="Strong brake",
            command=lambda: self._set_action_preset([-0.5, -0.2, -1.0, 0.9]),
        ).pack(fill=tk.X, pady=2)

        status_box = ttk.LabelFrame(left, text="Current state", padding=10)
        status_box.pack(fill=tk.BOTH, expand=True)

        rows = [
            ("Step", "step"),
            ("Time [s]", "time_s"),
            ("Reward", "reward"),
            ("Speed [km/h]", "speed"),
            ("Target [km/h]", "target_speed"),
            ("Speed error [km/h]", "speed_error"),
            ("SOC [-]", "soc"),
            ("NOx [g/s]", "nox"),
            ("Fuel [mg]", "fuel"),
            ("Engine on", "engine_on"),
            ("ICE speed [rpm]", "ice_speed"),
            ("Switch timer [steps]", "switch_timer"),
            ("Episode", "episode_state"),
        ]

        for label_text, key in rows:
            r = ttk.Frame(status_box)
            r.pack(fill=tk.X, pady=1)
            ttk.Label(r, text=label_text, width=19).pack(side=tk.LEFT)
            ttk.Label(r, textvariable=self.status_vars[key]).pack(side=tk.LEFT)

        plot_box = ttk.LabelFrame(right, text="Live trends", padding=6)
        plot_box.pack(fill=tk.BOTH, expand=True)
        self.plot_box = plot_box

        help_label = ttk.Label(
            left,
            text=(
                "Keyboard: L=step | Q/A ICE | W/S EM2 | E/D Fuel | R/F Brake\n"
                "Shift key gives larger increments"
            ),
            justify=tk.LEFT,
        )
        help_label.pack(fill=tk.X, pady=(10, 0))

    def _init_plot(self):
        self.figure = Figure(figsize=(10.2, 7.6), dpi=100)
        self.ax_speed = self.figure.add_subplot(221)
        self.ax_soc = self.figure.add_subplot(222)
        self.ax_emission = self.figure.add_subplot(223)
        self.ax_reward = self.figure.add_subplot(224)

        self.ax_speed.set_title("Speed")
        self.ax_speed.set_xlabel("Step")
        self.ax_speed.set_ylabel("km/h")

        self.ax_soc.set_title("SOC")
        self.ax_soc.set_xlabel("Step")
        self.ax_soc.set_ylabel("SOC [-]")

        self.ax_emission.set_title("NOx and Fuel")
        self.ax_emission.set_xlabel("Step")
        self.ax_emission.set_ylabel("NOx [g/s]")

        self.ax_reward.set_title("Reward")
        self.ax_reward.set_xlabel("Step")
        self.ax_reward.set_ylabel("Reward")

        (self.line_speed_actual,) = self.ax_speed.plot(
            [], [], label="actual", linewidth=2
        )
        (self.line_speed_target,) = self.ax_speed.plot(
            [], [], label="target", linestyle="--"
        )
        self.ax_speed.legend(loc="best")

        (self.line_soc,) = self.ax_soc.plot([], [], linewidth=2, color="tab:green")

        (self.line_nox,) = self.ax_emission.plot([], [], color="tab:red", label="NOx")
        self.ax_emission_twin = self.ax_emission.twinx()
        self.ax_emission_twin.set_ylabel("Fuel [mg]")
        (self.line_fuel,) = self.ax_emission_twin.plot(
            [], [], color="tab:blue", label="Fuel", linestyle="--"
        )

        (self.line_reward,) = self.ax_reward.plot(
            [], [], color="tab:purple", linewidth=2
        )

        handles_1, labels_1 = self.ax_emission.get_legend_handles_labels()
        handles_2, labels_2 = self.ax_emission_twin.get_legend_handles_labels()
        self.ax_emission.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")

        self.figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw_idle()

    def _bind_keys(self):
        self.root.bind("l", lambda _e: self.step_once())

        self.root.bind("q", lambda e: self._nudge_slider(0, 0.02, e))
        self.root.bind("a", lambda e: self._nudge_slider(0, -0.02, e))

        self.root.bind("w", lambda e: self._nudge_slider(1, 0.02, e))
        self.root.bind("s", lambda e: self._nudge_slider(1, -0.02, e))

        self.root.bind("e", lambda e: self._nudge_slider(2, 0.02, e))
        self.root.bind("d", lambda e: self._nudge_slider(2, -0.02, e))

        self.root.bind("r", lambda e: self._nudge_slider(3, 0.02, e))
        self.root.bind("f", lambda e: self._nudge_slider(3, -0.02, e))

    def _nudge_slider(self, idx: int, delta: float, event):
        big_step = bool(event.state & 0x0001)
        actual_delta = delta * (5.0 if big_step else 1.0)

        new_val = float(np.clip(self.slider_vars[idx].get() + actual_delta, -1.0, 1.0))
        self.slider_vars[idx].set(new_val)
        self._on_slider_change(idx)

    def _on_slider_change(self, _idx: int):
        self._refresh_slider_labels()

    def _refresh_slider_labels(self):
        for i, var in enumerate(self.slider_vars):
            self.slider_value_labels[i].config(text=f"{var.get():+.2f}")

    def _set_action_preset(self, values: list[float]):
        for i, v in enumerate(values):
            self.slider_vars[i].set(float(np.clip(v, -1.0, 1.0)))
        self._refresh_slider_labels()

    def _get_action_vector(self) -> np.ndarray:
        return np.array([v.get() for v in self.slider_vars], dtype=np.float32)

    def _reset_env(self, initial: bool = False):
        self.stop_auto()
        try:
            self.core.reset()
        except Exception as exc:
            messagebox.showerror("Reset failed", f"{exc}")
            return

        self._update_status()
        self._update_plots()

        if not initial:
            self.status_vars["episode_state"].set("running")

    def _update_status(self):
        snap = self.core.latest_snapshot()
        if not snap:
            return

        self.status_vars["step"].set(str(self.core.step_index))
        self.status_vars["time_s"].set(f"{snap['time_s']:.1f}")
        self.status_vars["reward"].set(f"{snap['reward']:+.5f}")
        self.status_vars["speed"].set(f"{snap['speed_actual']:.2f}")
        self.status_vars["target_speed"].set(f"{snap['target_speed']:.2f}")
        self.status_vars["speed_error"].set(f"{snap['speed_error_obs']:+.2f}")
        self.status_vars["soc"].set(f"{snap['soc']:.4f}")
        self.status_vars["nox"].set(f"{snap['nox_obs']:.5f}")
        self.status_vars["fuel"].set(f"{snap['fuel_obs']:.2f}")
        self.status_vars["engine_on"].set(str(snap["engine_on"]))
        self.status_vars["ice_speed"].set(f"{snap['ice_speed_obs']:.1f}")
        self.status_vars["switch_timer"].set(str(snap["switch_timer_steps"]))

        if snap["terminated"] or snap["truncated"]:
            self.status_vars["episode_state"].set("finished")
        else:
            self.status_vars["episode_state"].set("running")

    def _update_plots(self):
        if not self.core.history:
            self.line_speed_actual.set_data([], [])
            self.line_speed_target.set_data([], [])
            self.line_soc.set_data([], [])
            self.line_nox.set_data([], [])
            self.line_fuel.set_data([], [])
            self.line_reward.set_data([], [])

            for ax in [
                self.ax_speed,
                self.ax_soc,
                self.ax_emission,
                self.ax_reward,
                self.ax_emission_twin,
            ]:
                ax.relim()
                ax.autoscale_view()
            self.canvas.draw_idle()
            return

        hist = self.core.history
        x = np.array([r["step"] for r in hist], dtype=np.float32)

        speed_actual = np.array([r["speed_actual"] for r in hist], dtype=np.float32)
        speed_target = np.array([r["target_speed"] for r in hist], dtype=np.float32)
        soc = np.array([r["soc"] for r in hist], dtype=np.float32)
        nox = np.array([r["nox"] for r in hist], dtype=np.float32)
        fuel = np.array([r["fuel"] for r in hist], dtype=np.float32)
        reward = np.array([r["reward"] for r in hist], dtype=np.float32)

        self.line_speed_actual.set_data(x, speed_actual)
        self.line_speed_target.set_data(x, speed_target)
        self.line_soc.set_data(x, soc)
        self.line_nox.set_data(x, nox)
        self.line_fuel.set_data(x, fuel)
        self.line_reward.set_data(x, reward)

        for ax in [
            self.ax_speed,
            self.ax_soc,
            self.ax_emission,
            self.ax_reward,
            self.ax_emission_twin,
        ]:
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw_idle()

    def step_once(self):
        if self.is_stepping:
            return
        if self.core.last_terminated or self.core.last_truncated:
            return

        self.is_stepping = True
        try:
            action = self._get_action_vector()
            _, _, terminated, truncated, _, _ = self.core.step(action)
            self._update_status()
            self._update_plots()

            if terminated or truncated:
                self.stop_auto()
                self.status_vars["episode_state"].set("finished")
        except Exception as exc:
            self.stop_auto()
            messagebox.showerror("Simulation step failed", f"{exc}")
        finally:
            self.is_stepping = False

    def _auto_tick(self):
        if not self.auto_running:
            return

        self.step_once()

        if self.auto_running and not (
            self.core.last_terminated or self.core.last_truncated
        ):
            self.auto_job_id = self.root.after(1000, self._auto_tick)
        else:
            self.stop_auto()

    def start_auto(self):
        if self.auto_running:
            return
        if self.core.last_terminated or self.core.last_truncated:
            return

        self.auto_running = True
        self.auto_job_id = self.root.after(1000, self._auto_tick)

    def stop_auto(self):
        self.auto_running = False
        if self.auto_job_id is not None:
            self.root.after_cancel(self.auto_job_id)
            self.auto_job_id = None

    def export_csv(self):
        if not self.core.history:
            messagebox.showwarning("No data", "Run at least one step before exporting.")
            return

        default_name = datetime.now().strftime("manual_sim_%Y%m%d_%H%M%S.csv")
        output_path = filedialog.asksaveasfilename(
            title="Export step history",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if not output_path:
            return

        try:
            self.core.export_csv(output_path)
        except Exception as exc:
            messagebox.showerror("Export failed", f"{exc}")
            return

        messagebox.showinfo("Export complete", f"Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manual controller for EmissionControlEnv (base environment)."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "sac", "td3"],
        help="Config family to load from 02_rl_control/models/config_<algo>.py",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional path to a specific cycle CSV (if set, random_target is disabled).",
    )
    parser.add_argument(
        "--fixed_target_speed",
        type=float,
        default=None,
        help="Optional fixed target speed in km/h.",
    )
    parser.add_argument(
        "--random_target",
        action="store_true",
        help="Force random target mode.",
    )
    return parser.parse_args()


def resolve_runtime_options(args) -> RuntimeOptions:
    dataset_path = args.dataset_path
    if dataset_path is not None:
        dataset_path = os.path.abspath(dataset_path)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if args.fixed_target_speed is not None:
        random_target = True
    elif args.random_target:
        random_target = True
    elif dataset_path is not None:
        random_target = False
    else:
        random_target = True

    return RuntimeOptions(
        algorithm=args.algorithm.lower(),
        random_target=random_target,
        fixed_target_speed=args.fixed_target_speed,
        dataset_path=dataset_path,
    )


def main():
    args = parse_args()

    try:
        options = resolve_runtime_options(args)
        core = SimulatorCore(options)
    except Exception as exc:
        print("Failed to initialize manual controller:")
        print(exc)
        traceback.print_exc()
        return

    root = tk.Tk()
    app = ManualControllerApp(root, core)

    def _on_close():
        app.stop_auto()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
