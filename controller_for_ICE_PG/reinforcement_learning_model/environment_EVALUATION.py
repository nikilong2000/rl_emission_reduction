# environment_EVALUATION.py (CORRECTED VERSION)

import csv
import numpy as np
import os
import time

# Assuming you acturally use PyTorch somewhere, if not, you can remove it
# import torch
# import torch.nn.functional as F

import csv
import torch
import numpy as np
import torch.nn.functional as F
import math
import random
import pandas as pd
import os
import time  # <-- CORRECTION: Import added

# environment_EVALUATION.py (CORRECTED VERSION)

import csv
import numpy as np
import os
import time
import random

# environment_EVALUATION.py (CORRECTED AND VERIFIED VERSION)

import csv
import numpy as np
import os
import time
import random


class Environment:
    def __init__(
        self,
        transition_function_model,
        length_problem: int = 1200,
        alpha: float = 1.0,
        stable_velocity: float = 0.3,
        n_stable: int = 50,
    ):
        self.transition_function_model = transition_function_model
        self.length_problem = length_problem
        self.stable_velocity = -(stable_velocity**2)
        self.n_stable = n_stable
        self.alpha = alpha
        self.p_amb_bar = 1.0
        self.T_amb_K = 298.0
        self.step_count = 0
        self.stable_counter = 0

    def step(self, action, vel_target: float = 70.0, profile_step: bool = False):
        ice_timings_detailed = {}
        pg_timings_detailed = {}

        delta_mf, delta_brk, delta_ice_sp = action
        self.mf, self.brk, self.ice_sp = (
            float(delta_mf),
            float(delta_brk),
            float(delta_ice_sp),
        )

        # The tuple (outputs, timings) is unpacked here
        (outputs_ice, ice_timings) = self.transition_function_model.predict_ice(
            self.ice_sp, self.mf, self.T_amb_K, self.p_amb_bar
        )
        torque_ICE_tf, nox_tf, _, co_tf, _ = outputs_ice
        if profile_step:
            ice_timings_detailed = ice_timings

        self.torque_ICE = float(torque_ICE_tf.numpy())
        self.nox, self.co = float(nox_tf.numpy()), float(co_tf.numpy())

        self.torque_ICE = np.clip(self.torque_ICE, -50, 300.0)
        if self.ice_sp < 900.0:
            self.torque_ICE, self.mf = 0.0, 3.0

        (outputs_pg, pg_timings) = self.transition_function_model.predict_PG(
            self.ice_sp, 0.0, self.torque_ICE, self.brk
        )
        vel_out_tf, _ = outputs_pg
        if profile_step:
            pg_timings_detailed = pg_timings

        self.vel_out = float(vel_out_tf.numpy())
        self.vel = self.vel_out

        new_state = (vel_target, self.vel, self.mf, self.brk, self.ice_sp)
        reward = self.get_reward(vel_target)

        if reward >= self.stable_velocity:
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        terminated = self.stable_counter >= self.n_stable
        self.step_count += 1
        truncated = self.step_count >= self.length_problem

        return (
            new_state,
            reward,
            terminated,
            truncated,
            ice_timings_detailed,
            pg_timings_detailed,
        )

    def get_reward(self, vel_target: float) -> float:
        error = vel_target - self.vel_out
        return -self.alpha * ((error / 100.0) ** 2)

    def reset(self, vel_target=70):
        self.transition_function_model.reset_models()
        self.reset_variables()
        self.step_count = 0
        self.stable_counter = 0
        return vel_target, self.vel, self.mf, self.brk, self.ice_sp

    def reset_variables(self):
        self.sample_init_state()
        # ▼▼▼ REVERTED TO ORIGINAL VERSION ▼▼▼
        # This version is compatible with BOTH predict_PG signatures
        # (old and new) because it only needs the first value.
        outputs = self.transition_function_model.predict_PG(
            self.ice_sp, self.EM2, self.torque, self.brk
        )
        # If the function returns (outputs, timings), 'outputs' will be the list.
        # If it returns vel, soc, 'outputs' will be vel.
        if isinstance(outputs, tuple):  # Handles new signature (outputs, timings)
            vel, _ = outputs[0]
        else:  # Handles original signature (vel, soc)
            vel = outputs
        self.vel = vel.numpy() if hasattr(vel, "numpy") else vel

    def sample_init_state(self, folder_path="../src/data"):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {folder_path}")
        chosen_file = os.path.join(folder_path, random.choice(csv_files))
        with open(chosen_file, newline="") as f:
            rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError(f"File {chosen_file} has no data rows")
            chosen_row = random.choice(rows)
        self.mf = float(chosen_row["fuel"])
        self.brk = float(chosen_row["Brake"])
        self.ice_sp = float(chosen_row["ICE_Speed_soll"])
        self.EM2 = float(chosen_row["EM2_Torque"])
        self.torque = float(chosen_row["ICE_Torque_pred"])
