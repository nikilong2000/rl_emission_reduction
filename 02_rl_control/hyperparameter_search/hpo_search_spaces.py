"""
Search space definitions for Optuna hyperparameter optimization.
Each function uses trial.suggest_* to sample hyperparameters for the
corresponding SB3 algorithm. resolve_params() converts Optuna-native
representations (category keys, string floats) into values ready for SB3.
"""

# Network architecture options (name -> layer sizes)
NET_ARCH_MAP = {
    "small": [64, 64],
    "medium": [128, 128],
    "large": [256, 256],
    "deep": [128, 128, 128],
}


def sample_ppo_params(trial):
    """Sample PPO hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 15),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
        "net_arch": trial.suggest_categorical("net_arch", list(NET_ARCH_MAP.keys())),
    }


def sample_sac_params(trial):
    """Sample SAC hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 4, 8]),
        "learning_starts": trial.suggest_categorical(
            "learning_starts", [1000, 5000, 10000, 20000]
        ),
        "ent_coef": trial.suggest_categorical(
            "ent_coef", ["auto", "0.01", "0.05", "0.1", "0.2"]
        ),
        "use_sde": trial.suggest_categorical("use_sde", [True, False]),
        "net_arch": trial.suggest_categorical("net_arch", list(NET_ARCH_MAP.keys())),
    }


def sample_td3_params(trial):
    """Sample TD3 hyperparameters."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        "tau": trial.suggest_float("tau", 0.001, 0.02),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 4, 8]),
        "learning_starts": trial.suggest_categorical(
            "learning_starts", [1000, 5000, 10000, 20000]
        ),
        "policy_delay": trial.suggest_categorical("policy_delay", [1, 2, 4]),
        "target_policy_noise": trial.suggest_float("target_policy_noise", 0.1, 0.4),
        "target_noise_clip": trial.suggest_float("target_noise_clip", 0.3, 0.7),
        "action_noise_sigma": trial.suggest_float("action_noise_sigma", 0.05, 0.3),
        "net_arch": trial.suggest_categorical("net_arch", list(NET_ARCH_MAP.keys())),
    }


# Dispatch table
_SAMPLERS = {
    "ppo": sample_ppo_params,
    "sac": sample_sac_params,
    "td3": sample_td3_params,
}


def sample_params(trial, algo_key):
    """Sample hyperparameters for *algo_key* using Optuna *trial*."""
    return _SAMPLERS[algo_key](trial)


def resolve_params(params):
    """Convert Optuna trial params to SB3-ready values.
    - net_arch key  -> list of layer sizes
    - ent_coef str  -> "auto" or float
    """
    resolved = dict(params)
    if "net_arch" in resolved and isinstance(resolved["net_arch"], str):
        resolved["net_arch"] = NET_ARCH_MAP[resolved["net_arch"]]
    if "ent_coef" in resolved and resolved["ent_coef"] != "auto":
        resolved["ent_coef"] = float(resolved["ent_coef"])
    return resolved


def apply_config_override(config, overrides):
    """Apply a dict of resolved hyperparameter overrides onto a config module.
    Special handling:
      - ``net_arch`` → ``config.POLICY_KWARGS = dict(net_arch=...)``
      - everything else → ``setattr(config, KEY.upper(), value)``
    """
    for key, value in overrides.items():
        if key == "net_arch":
            config.POLICY_KWARGS = dict(net_arch=value)
        else:
            setattr(config, key.upper(), value)
