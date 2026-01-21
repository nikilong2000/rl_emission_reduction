import csv
import numpy as np
import random
import os
import time  # <-- CORRECTION: Import added


class Environment:
    """
    Manages the Reinforcement Learning environment for a vehicle control task.
    """

    def __init__(
        self,
        transition_function_model,
        length_problem: int = 1200,
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        stable_velocity: float = 0.3,
        n_stable: int = 50,
    ):
        self.transition_function_model = transition_function_model
        self.length_problem = length_problem
        self.stable_velocity = -(stable_velocity**2)
        self.n_stable = n_stable
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.p_amb_bar = 1.0
        self.T_amb_K = 298.0
        self.step_count = 0
        self.stable_counter = 0

    def step(self, action, vel_target: float = 70.0, profile_step: bool = False):
        """
        Executes one time step in the environment.
        """
        ice_duration = 0.0
        pg_duration = 0.0

        # 1) APPLY ACTION
        delta_mf, delta_brk, delta_ice_sp = action
        self.mf = float(delta_mf)
        self.brk = float(delta_brk)
        self.ice_sp = float(delta_ice_sp)

        # 2) ICE PREDICTION
        if profile_step:
            ice_start = time.time()
        torque_ICE_tf, nox_tf, _, co_tf, _ = self.transition_function_model.predict_ice(
            self.ice_sp, self.mf, self.T_amb_K, self.p_amb_bar
        )
        self.torque_ICE = float(torque_ICE_tf.numpy())
        self.nox = float(nox_tf.numpy())
        self.co = float(co_tf.numpy())
        if profile_step:
            ice_duration = time.time() - ice_start

        # 3) ICE CLIPPING
        self.torque_ICE = np.clip(self.torque_ICE, -50, 300.0)
        if self.ice_sp < 900.0:
            self.torque_ICE = 0.0
            self.mf = 3.0

        # 4) POWER-SPLIT PREDICTION (PG / EM2)
        if profile_step:
            pg_start = time.time()
        vel_out_tf, _ = self.transition_function_model.predict_PG(
            self.ice_sp, 0.0, self.torque_ICE, self.brk
        )
        self.vel_out = float(vel_out_tf.numpy())
        if profile_step:
            pg_duration = time.time() - pg_start

        self.vel = self.vel_out

        # 5) NEW STATE AND REWARD
        new_state = (vel_target, self.vel, self.mf, self.brk, self.ice_sp)
        reward = self.get_reward(vel_target)

        # 6) TERMINATION CONDITIONS
        if reward >= self.stable_velocity:
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        terminated = self.stable_counter >= self.n_stable
        self.step_count += 1
        truncated = self.step_count >= self.length_problem

        # Return new values
        return new_state, reward, terminated, truncated, ice_duration, pg_duration

    def get_reward(self, vel_target: float) -> float:
        error = vel_target - self.vel_out
        max_possible_error = 100.0
        normalized_error = error / max_possible_error
        reward = -self.alpha * (normalized_error**2)
        return reward

    def reset(self, vel_target=70):
        self.transition_function_model.reset_models()
        self.reset_variables()
        self.transition_function_model.reset_models()
        self.step_count = 0
        self.stable_counter = 0
        return vel_target, self.vel, self.mf, self.brk, self.ice_sp

    def reset_variables(self):
        self.sample_init_state()
        self.vel, _ = self.transition_function_model.predict_PG(
            self.ice_sp, self.EM2, self.torque, self.brk
        )

    def sample_init_state(self, folder_path="../src/data"):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {folder_path}")
        chosen_file = os.path.join(folder_path, random.choice(csv_files))
        chosen_row = None
        with open(chosen_file, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                if random.random() < 1 / i:
                    chosen_row = row
        if chosen_row is None:
            raise ValueError(f"The file {chosen_file} has no data rows")
        self.mf = float(chosen_row["fuel"])
        self.brk = float(chosen_row["Brake"])
        self.ice_sp = float(chosen_row["ICE_Speed_soll"])
        self.EM2 = float(chosen_row["EM2_Torque"])
        self.torque = float(chosen_row["ICE_Torque_pred"])
