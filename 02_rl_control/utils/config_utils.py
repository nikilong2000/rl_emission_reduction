import os
import json
import importlib


def config_check(continue_from, train_config):
    prev_metrics_path = os.path.join(
        os.path.dirname(continue_from), "evaluation_metrics.json"
    )
    if os.path.exists(prev_metrics_path):
        with open(prev_metrics_path, "r") as f:
            prev_metrics = json.load(f)
        if "configuration" in prev_metrics:
            prev_config = prev_metrics["configuration"]

            keys_to_ignore = {"total_timesteps", "continued_run", "continued_from"}
            mismatches = []
            for k, v in train_config.items():
                if k in keys_to_ignore:
                    continue
                if k in prev_config and prev_config[k] != v:
                    mismatches.append(f"{k}: current={v} != previous={prev_config[k]}")

            if mismatches:
                raise ValueError(
                    "Configuration mismatch between current run and the run being continued from. "
                    "To prevent training with inconsistent parameters, the config must match. "
                    f"Mismatches: {', '.join(mismatches)}"
                )


def load_config(current_dir, algo_key: str):
    """Dynamically import the config module for the given algorithm."""
    config_path = os.path.join(current_dir, f"config_{algo_key}.py")
    spec = importlib.util.spec_from_file_location(f"config_{algo_key}", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config
