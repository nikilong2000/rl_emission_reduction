import os
from stable_baselines3.common.callbacks import BaseCallback


class VecNormalizeCheckpointCallback(BaseCallback):
    """Saves VecNormalize statistics alongside each model checkpoint.

    Uses the same ``save_freq`` and ``save_path`` as ``CheckpointCallback`` so
    every checkpoint zip has a matching ``<name_prefix>_<steps>_steps_vecnormalize.pkl``
    in the same directory.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str, vec_normalize, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.vec_normalize = vec_normalize

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps_vecnormalize.pkl",
            )
            self.vec_normalize.save(path)
            if self.verbose >= 1:
                print(f"Saved VecNormalize checkpoint to {path}")
        return True
