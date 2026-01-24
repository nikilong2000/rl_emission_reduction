import csv
import torch
import numpy as np
import torch.nn.functional as F
import math
import random
import pandas as pd
import os


class Environment:  # Correcte
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
        stable_error_threshold: float = 0.5,
        n_stable: int = 50,
    ):  # Correcte
        """
        Initializes the environment.

        Args:
            transition_function_model: The external model used to predict state transitions.
            length_problem (int): The maximum number of steps per episode before it's truncated.
            alpha (float): The coefficient for the velocity error component of the reward.
            beta (float): Unused reward coefficient.
            gamma (float): Unused reward coefficient.
            stable_error_threshold (float): A threshold used to determine if the velocity is stable.
                                     The episode terminates if the reward remains above this threshold
                                     for `n_stable` consecutive steps.
            n_stable (int): The number of consecutive steps the velocity must be stable for the
                            episode to terminate.


            stable_error_threshold: float = 0.5 #(en km/h),
        """

        # Transition model (external)
        self.transition_function_model = transition_function_model

        # Maximum error  #Correct
        self.MAX_POSSIBLE_ERROR = (
            80  # -5min vel and max vel 0  = # (200.0 - (-5.0)) 205.0
        )

        # Stability [km/h needs to be converted to error range]
        ## Calculate which reward corresponds to that error threshold
        norm_stable_error = stable_error_threshold / self.MAX_POSSIBLE_ERROR

        ## This is the new reward threshold
        self.stable_reward_threshold = 1.0 - (norm_stable_error**2)

        # Episode configuration
        self.length_problem = length_problem
        self.n_stable = n_stable

        # Coefficients for the reward
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        # Ambient conditions
        self.p_amb_bar = 1.0
        self.T_amb_K = 298.0

        # State variables
        self.step_count = 0
        self.stable_counter = 0

    def step(self, action, vel_target: float = 70.0):  # Correct
        """
        Executes one time step in the environment.

        Note: All internal calculations are performed using native Python floats or NumPy
              to avoid type conflicts with framework tensors.

        Args:
            action (tuple): A tuple containing the actions to apply.
            vel_target (float): The target velocity for the current step.

        Returns:
            tuple: A tuple containing (new_state, reward, terminated, truncated).
        """
        # ------------------------------------------------------------------
        # 1) APPLY ACTION
        # ------------------------------------------------------------------
        delta_mf, delta_brk, delta_ice_sp = action
        self.mf = float(delta_mf)
        self.brk = float(delta_brk)
        self.ice_sp = float(delta_ice_sp)

        # ------------------------------------------------------------------
        # 2) ICE PREDICTION
        # ------------------------------------------------------------------
        torque_ICE_tf, nox_tf, _, co_tf, _ = self.transition_function_model.predict_ice(
            self.ice_sp, self.mf, self.T_amb_K, self.p_amb_bar
        )

        # → Convert TF tensors to float
        self.torque_ICE = float(torque_ICE_tf.numpy())
        self.nox = float(nox_tf.numpy())
        self.co = float(co_tf.numpy())

        # ------------------------------------------------------------------
        # 3) ICE CLIPPING
        # -----------------------------------------------------------------

        self.torque_ICE = np.clip(self.torque_ICE, -50, 300.0)

        # Turn off ICE if rotational speed is low
        if self.ice_sp < 900.0:
            self.torque_ICE = 0.0
            self.mf = 3.0

        # ------------------------------------------------------------------
        # 4) POWER-SPLIT PREDICTION (PG / EM2)
        # ------------------------------------------------------------------
        vel_out_tf, _ = self.transition_function_model.predict_PG(
            self.ice_sp, 0.0, self.torque_ICE, self.brk
        )

        self.vel_out = float(vel_out_tf.numpy())
        self.vel = self.vel_out  # Save current velocity

        # ------------------------------------------------------------------
        # 5) NEW STATE AND REWARD
        # ------------------------------------------------------------------
        new_state = (
            vel_target,
            self.vel,
            self.mf,
            self.brk,
            self.ice_sp,
        )  # Actually what changes is "self.vel", "self.ice_sp" and if ICE is deactivated "self.mf"

        reward = self.get_reward(vel_target)

        # ------------------------------------------------------------------
        # 6) TERMINATION CONDITIONS
        # ------------------------------------------------------------------
        if reward >= self.stable_reward_threshold:
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        terminated = self.stable_counter >= self.n_stable

        self.step_count += 1

        truncated = self.step_count >= self.length_problem

        # ------------------------------------------------------------------
        return new_state, reward, terminated, truncated

    #     # In Environment.py
    #     def get_reward(self, vel_target: float) -> float:  #Correct
    #         """
    #         Calculates the reward based on the normalized squared error.
    #         The reward is in the range [0, 1].
    #         1.0 = zero error (perfect)
    #         0.0 = max error (terrible)
    #         """

    #         # 1. Calculate absolute error
    #         error = vel_target - self.vel_out
    #         abs_error = abs(error)

    #         # 2. Normalize the error (keeps it between [0, 1])
    #         # We ensure it does not exceed 1.0 just in case
    #         normalized_error = min(abs_error / self.MAX_POSSIBLE_ERROR, 1.0)

    #         # 3. Calculate reward
    #         reward = 1.0 - (normalized_error ** 2)

    #         return reward

    def get_reward(self, vel_target: float) -> float:
        """
        Calculates the reward based on the normalized squared error.
        """
        error = vel_target - self.vel_out
        abs_error = abs(error)

        # Now, with MAX_ERROR = 80.0:
        # If error=70 -> norm_err = 0.875 -> reward = 1 - 0.765 = 0.235
        # If error=60 -> norm_err = 0.75  -> reward = 1 - 0.562 = 0.438
        # NOW THERE IS A GRADIENT!

        normalized_error = min(abs_error / self.MAX_POSSIBLE_ERROR, 1.0)
        reward = 1.0 - (normalized_error**2)

        return reward

    def reset(self, vel_target=70):  # Correct
        """
        Resets the environment to a random initial state and returns it.

        Returns:
            tuple: The initial state of the environment.
        """
        # Resets the transition function model
        self.transition_function_model.reset_models()

        # Generates a random initial state
        self.reset_variables()

        # Resets the transition function model, it is necessary because in the previous step ".reset_variables()" a PG iteration is done
        self.transition_function_model.reset_models()

        # Counters
        self.step_count = 0
        self.stable_counter = 0

        return vel_target, self.vel, self.mf, self.brk, self.ice_sp

    def reset_variables(self):  # Correct
        """
        Sets the environment's state variables to a random initial configuration.
        """
        # Generates a random initial state
        self.sample_init_state()

        # Calculates the initial velocity
        self.vel, _ = self.transition_function_model.predict_PG(
            self.ice_sp, self.EM2, self.torque, self.brk
        )

    def sample_init_state(self, folder_path="../src/data"):
        """
        Samples an initial state from a random row in a random CSV file.

        This method uses reservoir sampling to select a row in a single pass,
        which is memory-efficient for large files.
        """
        # 1) List CSVs
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {folder_path}")

        chosen_file = os.path.join(folder_path, random.choice(csv_files))

        # 2) Reservoir sampling: choose 1 random row in one pass
        chosen_row = None
        with open(chosen_file, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                # with probability 1/i replace chosen row
                if random.random() < 1 / i:
                    chosen_row = row

        if chosen_row is None:
            raise ValueError(f"File {chosen_file} has no data rows")

        # 3) Assign to attributes
        self.mf = float(chosen_row["fuel"])
        self.brk = float(chosen_row["Brake"])
        self.ice_sp = float(chosen_row["ICE_Speed_soll"])
        self.EM2 = float(chosen_row["EM2_Torque"])
        self.torque = float(chosen_row["ICE_Torque_pred"])


#         print(f"reset --> mf: {self.mf}, brk: {self.brk}, ice_sp: {self.ice_sp}, EM2: {self.EM2}, torque: {self.torque}")
