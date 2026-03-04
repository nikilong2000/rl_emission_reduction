import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import tensorflow as tf
import os

try:
    from .utils.network_utils import load_network, set_states
    from . import config
except ImportError:
    from utils.network_utils import load_network, set_states
    import config


class EmissionControlEnv(gym.Env):
    """
    Gym Environment for controlling ICE and Drivetrain to minimize emissions and track speed.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        # Load Data
        self.df = pd.read_csv(config.INPUT_DATA_PATH, delimiter=";", encoding="latin1")
        if self.df.shape[1] <= 1:
            self.df = pd.read_csv(
                config.INPUT_DATA_PATH, delimiter=",", encoding="latin1"
            )

        # Clean column names
        self.df.columns = [col.strip() for col in self.df.columns]

        self.max_steps = len(self.df)
        self.current_step = 0

        # Load Models
        print("Loading ICE Model...")
        self.ice_tuple = load_network(config.ICE_MODEL_DIR)
        (
            self.ice_main,
            self.ice_init,
            self.ice_in_scaler,
            self.ice_out_scaler,
            self.ice_predict_main,
            self.ice_predict_init,
        ) = self.ice_tuple

        print("Loading Drivetrain (PG) Model...")
        self.pg_tuple = load_network(config.PG_MODEL_DIR)
        (
            self.pg_main,
            self.pg_init,
            self.pg_in_scaler,
            self.pg_out_scaler,
            self.pg_predict_main,
            self.pg_predict_init,
        ) = self.pg_tuple

        # Define Action Space
        # [Engine_State, ICE_Speed, EM2_Torque, Fuel_Mass, Brake] — normalised to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # Physical bounds used for rescaling
        self.action_min = np.array(
            [-1.0, 900.0, -421.0, 3.0, 0.0], dtype=np.float32
        )
        self.action_max = np.array([1.0, 4000.0, 421.0, 70.0, 100.0], dtype=np.float32)

        # Define Observation Space
        # [Car_Speed, Speed_Error, SOC, ICE_Torque, NOx, Engine_On, SOC_Error]
        _obs_low  = np.array([-5.0, -155.0, 0.0, -50.0, 0.0, 0.0, -1.0], dtype=np.float32)
        _obs_high = np.array([150.0, 155.0, 1.0, 300.0, 10.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=_obs_low, high=_obs_high, dtype=np.float32
        )

        # Stored as explicit attributes so subclasses (e.g. EmissionControlEnvThermal)
        # can extend them before rebuilding observation_space.
        self.obs_low  = _obs_low.copy()
        self.obs_high = _obs_high.copy()

        # State variables
        self.last_ice_torque = 0.0
        self.last_car_speed = 0.0
        self.last_soc = 0.7
        self.initial_soc = self.last_soc
        self.last_nox = 0.0
        self.last_engine_on = False

        # Column names mapping
        self.col_map = {
            "time": "Time_s",
            "target_speed": "Car_Speed_kmph",
            "t_amb": "T_amb_K",
            "p_amb": "p_amb_bar",
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # ICE Init
        ice_init_vals = np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    298.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    298.0,
                    298.0,
                    298.0,
                    298.0,
                    298.0,
                    298.0,
                ]
            ]
        )
        ice_init_scaled = self.ice_out_scaler.transform(ice_init_vals).reshape(1, 1, -1)
        ice_states = self.ice_predict_init(ice_init_scaled)
        ice_states_dict = dict(zip(self.ice_init.output_names, ice_states))
        set_states(self.ice_main, ice_states_dict)

        # PG Init
        pg_init_vals = np.array([[0.0, 0.7]])
        pg_init_scaled = self.pg_out_scaler.transform(pg_init_vals).reshape(1, 1, -1)
        pg_states = self.pg_predict_init(pg_init_scaled)
        pg_states_dict = dict(zip(self.pg_init.output_names, pg_states))
        set_states(self.pg_main, pg_states_dict)

        # Initial State
        self.last_ice_torque = 0.0
        self.last_car_speed = 0.0
        self.last_soc = 0.7
        self.initial_soc = self.last_soc
        self.last_nox = 0.0
        self.last_engine_on = False

        target_speed = self.df.loc[0, self.target_col_name()]
        initial_speed_error = target_speed - self.last_car_speed

        obs = np.array(
            [
                self.last_car_speed,
                initial_speed_error,
                self.last_soc,
                self.last_ice_torque,
                self.last_nox,
                float(self.last_engine_on),
                0.0,
            ],
            dtype=np.float32,
        )

        info = {
            "time_s": self.df.loc[0, self.col_map["time"]],
            "raw_obs": obs,  # Exposed so subclasses can append extra observations
        }
        return obs, info

    def step(self, action):
        # 1. Rescale Action from [-1, 1] to physical range
        scaled_action = self.action_min + (action + 1.0) * 0.5 * (
            self.action_max - self.action_min
        )

        engine_state_req = scaled_action[0]
        ice_speed_rpm = scaled_action[1]
        em2_torque_nm = scaled_action[2]
        fuel_mg = scaled_action[3]
        brake_perc = scaled_action[4]

        engine_on = engine_state_req >= 0.0
        if not engine_on:
            ice_speed_rpm = 0.0
            fuel_mg = 0.0

        # 2. Ambient conditions
        t_amb = (
            self.df.loc[self.current_step, "T_amb_K"]
            if "T_amb_K" in self.df.columns
            else 298.0
        )
        p_amb = (
            self.df.loc[self.current_step, "p_amb_bar"]
            if "p_amb_bar" in self.df.columns
            else 1.005
        )

        # 3. ICE prediction
        ice_inputs = np.array([[ice_speed_rpm, fuel_mg, t_amb, p_amb]])
        ice_in_scaled = self.ice_in_scaler.transform(ice_inputs).reshape(1, 1, -1)
        ice_pred_scaled = self.ice_predict_main(ice_in_scaled)
        ice_pred = self.ice_out_scaler.inverse_transform(ice_pred_scaled.numpy()[0])

        ice_torque = ice_pred[0][0]
        nox_tp = ice_pred[0][6]

        # 4. Drivetrain prediction
        pg_inputs = np.array([[ice_speed_rpm, ice_torque, em2_torque_nm, brake_perc]])
        pg_in_scaled = self.pg_in_scaler.transform(pg_inputs).reshape(1, 1, -1)
        pg_pred_scaled = self.pg_predict_main(pg_in_scaled)
        pg_pred = self.pg_out_scaler.inverse_transform(pg_pred_scaled.numpy()[0])

        car_speed = pg_pred[0][0]
        soc = pg_pred[0][1]

        # 5. Reward
        target_speed = self.df.loc[self.current_step, self.target_col_name()]

        speed_error = abs(target_speed - car_speed)
        soc_error = abs(soc - self.initial_soc)
        soc_error_squared = (soc - self.initial_soc) ** 2

        norm_speed = 50.0
        norm_emission = 0.1
        norm_fuel = 70.0
        norm_brake = 100.0

        safe_speed_penalty = min(speed_error / norm_speed, 1.0)
        safe_emission_penalty = min(nox_tp / norm_emission, 1.0)

        reward = 0.0

        ## DO NOT EDIT REWARDS HERE; INSTEAD IN config.py!!!!
        reward -= config.W_SPEED * safe_speed_penalty
        reward -= config.W_EMISSION * safe_emission_penalty
        reward -= config.W_FUEL * (fuel_mg / norm_fuel)
        reward -= config.W_BRAKE * (brake_perc / norm_brake)
        reward -= config.W_SOC * soc_error
        reward -= config.W_SOC_SQUARED * soc_error_squared

        if engine_on and not self.last_engine_on:
            reward -= config.W_FLICKER

        # 6. Update State
        self.current_step += 1
        self.last_ice_torque = ice_torque
        self.last_car_speed = car_speed
        self.last_soc = soc
        self.last_nox = nox_tp
        self.last_engine_on = engine_on

        terminated = False
        truncated = False

        if self.current_step >= self.max_steps - 1:
            terminated = True

        next_target_speed = (
            self.df.loc[self.current_step, self.target_col_name()]
            if not terminated
            else 0.0
        )

        next_speed_error = next_target_speed - car_speed
        soc_error_signed = soc - self.initial_soc

        obs = np.array(
            [
                car_speed,
                next_speed_error,
                soc,
                ice_torque,
                nox_tp,
                float(engine_on),
                soc_error_signed,
            ],
            dtype=np.float32,
        )

        info = {
            "time_s": self.df.loc[self.current_step, self.col_map["time"]],
            "speed_error": speed_error,
            "nox": nox_tp,
            "fuel": fuel_mg,
            "engine_on": engine_on,
            "ice_torque": ice_torque,
            "ice_speed_rpm": ice_speed_rpm,
            "em2_torque_nm": em2_torque_nm,
            "brake_perc": brake_perc,
        }

        return obs, reward, terminated, truncated, info

    def target_col_name(self):
        if "Car_Speed_kmph" in self.df.columns:
            return "Car_Speed_kmph"
        return self.df.columns[1]
