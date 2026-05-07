"""Interactive function plotting dashboard using matplotlib sliders.

Usage with callable:
    from utils.function_dashboard import plot_function_dashboard

    def f(x, scale_factor=10.0):
        return np.exp(-0.5 * (x / scale_factor) ** 2)

    plot_function_dashboard(
        f,
        x_range=(0.0, 80.0),
        params={"scale_factor": (1.0, 30.0, 10.0)},
        title="Gaussian Speed Reward",
        x_label="speed_error",
    )

Usage with expression string:
    from utils.function_dashboard import plot_expression_dashboard

    plot_expression_dashboard(
        "np.exp(-0.5 * (x / scale_factor) ** 2)",
        x_range=(0.0, 80.0),
        params={"scale_factor": (1.0, 30.0, 10.0)},
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

ParamSpec = Mapping[str, Tuple[float, float, float]]


@dataclass
class DashboardConfig:
    x_range: Tuple[float, float] = (0.0, 100.0)
    num_points: int = 500
    title: str = "Function Dashboard"
    x_label: str = "x"
    y_label: str = "f(x)"


def _validate_params(params: ParamSpec) -> Dict[str, Tuple[float, float, float]]:
    validated: Dict[str, Tuple[float, float, float]] = {}
    for name, spec in params.items():
        if len(spec) != 3:
            raise ValueError(f"Parameter '{name}' must be (min, max, default).")
        p_min, p_max, p_default = map(float, spec)
        if not p_min < p_max:
            raise ValueError(f"Parameter '{name}' requires min < max.")
        if not (p_min <= p_default <= p_max):
            raise ValueError(f"Parameter '{name}' default must lie in [min, max].")
        validated[name] = (p_min, p_max, p_default)
    return validated


def _safe_eval_expression(
    expression: str, x: np.ndarray, **kwargs: float
) -> np.ndarray:
    allowed = {
        "np": np,
        "x": x,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "log": np.log,
        "log10": np.log10,
        "abs": np.abs,
        "pi": np.pi,
        "clip": np.clip,
        **kwargs,
    }
    return eval(expression, {"__builtins__": {}}, allowed)


def plot_function_dashboard(
    func: Callable[..., np.ndarray],
    *,
    x_range: Tuple[float, float] = (0.0, 100.0),
    params: Union[ParamSpec, None] = None,
    num_points: int = 500,
    title: str = "Function Dashboard",
    x_label: str = "x",
    y_label: str = "f(x)",
) -> None:
    """Plot a function with interactive sliders.

    Args:
        func: Callable with signature func(x, **param_values).
        x_range: Min and max x values.
        params: Dict {name: (min, max, default)} for slider-controlled params.
        num_points: Number of points on x-axis.
    """
    config = DashboardConfig(
        x_range=x_range,
        num_points=num_points,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )

    params = _validate_params(params or {})

    x_min, x_max = config.x_range
    x = np.linspace(x_min, x_max, config.num_points)

    fig_height = max(5.0, 4.0 + 0.45 * len(params))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    def current_param_values(slider_map: Mapping[str, Slider]) -> Dict[str, float]:
        return {name: slider.val for name, slider in slider_map.items()}

    initial_values = {k: v[2] for k, v in params.items()}
    y = np.asarray(func(x, **initial_values), dtype=float)
    (line,) = ax.plot(x, y, lw=2, color="tab:blue")

    ax.set_title(config.title)
    ax.set_xlabel(config.x_label)
    ax.set_ylabel(config.y_label)
    ax.grid(alpha=0.3)

    plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.18 + 0.08 * len(params))

    slider_map: Dict[str, Slider] = {}
    slider_base = 0.08
    slider_step = 0.06

    for idx, (name, (p_min, p_max, p_default)) in enumerate(params.items()):
        slider_ax = fig.add_axes([0.12, slider_base + idx * slider_step, 0.72, 0.03])
        slider = Slider(slider_ax, name, p_min, p_max, valinit=p_default)
        slider_map[name] = slider

    reset_ax = fig.add_axes([0.86, 0.02, 0.1, 0.05])
    reset_btn = Button(reset_ax, "Reset")

    def refresh(_val: float) -> None:
        vals = current_param_values(slider_map)
        try:
            y_new = np.asarray(func(x, **vals), dtype=float)
        except Exception:
            y_new = np.full_like(x, np.nan)
        line.set_ydata(y_new)
        ax.relim()
        ax.autoscale_view(scaley=True)
        fig.canvas.draw_idle()

    for slider in slider_map.values():
        slider.on_changed(refresh)

    def do_reset(_event) -> None:
        for slider in slider_map.values():
            slider.reset()

    reset_btn.on_clicked(do_reset)
    plt.show()


def plot_expression_dashboard(
    expression: str,
    *,
    x_range: Tuple[float, float] = (0.0, 100.0),
    params: Union[ParamSpec, None] = None,
    num_points: int = 500,
    title: str = "Expression Dashboard",
    x_label: str = "x",
    y_label: str = "f(x)",
) -> None:
    """Plot a string expression with sliders.

    Example expression:
        "np.exp(-0.5 * (x / scale_factor) ** 2)"
    """

    def _wrapped(x: np.ndarray, **kwargs: float) -> np.ndarray:
        return _safe_eval_expression(expression, x, **kwargs)

    plot_function_dashboard(
        _wrapped,
        x_range=x_range,
        params=params,
        num_points=num_points,
        title=title,
        x_label=x_label,
        y_label=y_label,
    )


if __name__ == "__main__":
    expression = "np.exp(-0.5 * (x / scale_factor) ** 2)"
    plot_expression_dashboard(
        expression,
        x_range=(0.0, 80.0),
        params={"scale_factor": (1.0, 30.0, 10.0)},
        title=f"Gaussian Speed-Error Reward\n{expression}",
        x_label="speed_error",
        y_label="reward",
    )
