import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob
import random


try:
    from .utils.network_utils import load_network, set_states
    from . import config
except ImportError:
    from utils.network_utils import load_network, set_states
    import config

# 16 dims for ICE
_ICE_COLS = [
    "ice_torque_nm",
    "fuel_tot_gps",
    "nox_eo_gps",
    "co_eo_gps",
    "thc_eo_gps",
    "t_gas_eo_k",
    "nox_tp_gps",
    "co_tp_gps",
    "co2_tp_gps",
    "thc_tp_gps",
    "t_wall_scr1_k",
    "t_wall_doc_k",
    "t_sub_dpf_k",
    "t_wall_scr2_k",
    "t_wall_scr3_k",
    "t_gas_tp_k",
]

_ICE_DEFAULTS = [
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

_PG_COLS = ["car_speed_kmph", "soc_1"]
_PG_DEFAULTS = [0.0, 0.7]


class EmissionControlEnv(gym.Env):
    """
    Gym Environment for controlling ICE and Drivetrain to minimize emissions and track speed.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, dataset_path=None):
        super().__init__()

        self.dataset_path = dataset_path

        # loading csvs with speed trajectories
        all_files = glob.glob(os.path.join(config.TRAIN_DATA_DIR, "*.csv"))
        self.data_files = [
            f
            for f in all_files
            if os.path.basename(f).startswith("drivetrain")
            or os.path.basename(f) in ("WLTC_high.csv", "WLTC_low.csv")
        ]
        if not self.data_files:
            raise ValueError(f"No valid CSV files found in {config.TRAIN_DATA_DIR}")

        self.df = None
        self.max_steps = 0
        self.current_step = 0

        # load models
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

        # define action space
        # normalise actions to [-1, 1] and rescale inside step
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )  # [Engine_State, ICE_Speed, EM2_Torque, Fuel_Mass, Brake]

        # action scaling (Min/Max values for rescaling)
        self.action_min = np.array([-1.0, 900.0, -421.0, 3.0, 0.0], dtype=np.float32)
        self.action_max = np.array([1.0, 4000.0, 421.0, 70.0, 100.0], dtype=np.float32)

        # define observation space
        self.observation_space = spaces.Box(
            low=np.array([-5.0, -155.0, 0.0, -50.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([150.0, 155.0, 1.0, 300.0, 0.4, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )  # [Car_Speed (km/h), Speed_Error (km/h), SOC (0-1), ICE_Torque (Nm), NOx (g/s), Engine_On (0-1), SOC_Error (-1 to 1)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # load random or specific driving cycle
        if self.dataset_path is not None:
            chosen_file = self.dataset_path
        else:
            chosen_file = random.choice(self.data_files)

        print("\nUsing", str(chosen_file), ".")

        self.df = pd.read_csv(chosen_file, delimiter=";", encoding="latin1")
        if self.df.shape[1] <= 1:
            self.df = pd.read_csv(chosen_file, delimiter=",", encoding="latin1")

        # clean column names
        self.df.columns = [col.strip() for col in self.df.columns]
        self.max_steps = len(self.df)

        # initialise models based on initial state in cycle
        ice_init_val_row = []
        for c, d in zip(_ICE_COLS, _ICE_DEFAULTS):
            if c in self.df.columns and not pd.isna(self.df.loc[0, c]):
                ice_init_val_row.append(float(self.df.loc[0, c]))
            else:
                ice_init_val_row.append(d)

        ice_init_vals = np.array([ice_init_val_row])
        ice_init_scaled = self.ice_out_scaler.transform(ice_init_vals).reshape(1, 1, -1)
        ice_states = self.ice_predict_init(ice_init_scaled)
        ice_states_dict = dict(zip(self.ice_init.output_names, ice_states))
        set_states(self.ice_main, ice_states_dict)

        pg_init_val_row = []
        for c, d in zip(_PG_COLS, _PG_DEFAULTS):
            if c in self.df.columns and not pd.isna(self.df.loc[0, c]):
                pg_init_val_row.append(float(self.df.loc[0, c]))
            else:
                pg_init_val_row.append(d)

        pg_init_vals = np.array([pg_init_val_row])
        pg_init_scaled = self.pg_out_scaler.transform(pg_init_vals).reshape(1, 1, -1)
        pg_states = self.pg_predict_init(pg_init_scaled)
        pg_states_dict = dict(zip(self.pg_init.output_names, pg_states))
        set_states(self.pg_main, pg_states_dict)

        # set initial state
        self.last_ice_torque = ice_init_val_row[0]  # ICE_Torque
        self.last_car_speed = pg_init_val_row[0]  # Car_Speed
        self.last_soc = pg_init_val_row[1]  # SOC
        self.initial_soc = self.last_soc
        self.last_nox = ice_init_val_row[6]  # nox_tp_gps
        self.last_engine_on = (
            True if ice_init_val_row[1] >= 0.1 else False
        )  # value by Markus to approximate initial state

        target_speed = self.df.loc[0, self.target_col_name()]

        obs = np.array(
            [
                self.last_car_speed,
                0.0,  # Initial speed error is exactly 0.0
                self.last_soc,
                self.last_ice_torque,
                self.last_nox,
                float(self.last_engine_on),
                0.0,  # Initial SOC error is exactly 0.0
            ],
            dtype=np.float32,
        )

        info = {
            "time_s": self.df.loc[0, "time_s"],
            "car_speed_kmph": target_speed,
        }
        return obs, info

    def step(self, action):
        # 1. Rescale Action
        # action is in [-1, 1], map to [min, max] (physical scale)
        scaled_action = self.action_min + (action + 1.0) * 0.5 * (
            self.action_max - self.action_min
        )

        # print("Scaled Actions:")
        # print(f"  Engine_State: {scaled_action[0]}")
        # print(f"  ICE_Speed_rpm: {scaled_action[1]}")
        # print(f"  EM2_Torque_Nm: {scaled_action[2]}")
        # print(f"  Fuel_Mass_mg: {scaled_action[3]}")
        # print(f"  Brake_Perc: {scaled_action[4]}")

        engine_state_req = scaled_action[0]
        ice_speed_rpm = scaled_action[1]
        em2_torque_nm = scaled_action[2]
        fuel_mg = scaled_action[3]
        brake_perc = scaled_action[4]

        # Enforce Engine Off bounds
        # If engine_state_req < 0, ICE is commanded OFF
        engine_on = engine_state_req >= 0.0
        if not engine_on:
            ice_speed_rpm = 0.0
            fuel_mg = 0.0

        # 2. Prepare Inputs for ICE Model
        # Inputs: "ICE_Speed_rpm", "fuel_mg", "T_amb_K", "p_amb_bar"
        t_amb = (
            self.df.loc[self.current_step, "t_amb_k"]
            if "t_amb_k" in self.df.columns
            else 298.0
        )
        p_amb = (
            self.df.loc[self.current_step, "p_amb_bar"]
            if "p_amb_bar" in self.df.columns
            else 1.005
        )

        ice_inputs = np.array([[ice_speed_rpm, fuel_mg, t_amb, p_amb]])
        ice_in_scaled = self.ice_in_scaler.transform(ice_inputs).reshape(1, 1, -1)

        # 3. Predict ICE Outputs
        ice_pred_scaled = self.ice_predict_main(ice_in_scaled)
        ice_pred = self.ice_out_scaler.inverse_transform(ice_pred_scaled.numpy()[0])

        # Index 0: ICE_Torque_Nm
        # Index 6: NOx_tp_gps (Tailpipe)
        ice_torque = ice_pred[0][0]
        nox_tp = ice_pred[0][6]

        # 4. Prepare Inputs for PG Model
        # Inputs: "ICE_Speed_rpm", "ICE: ICE_Torque_Nm", "EM2_Torque_Nm", "Brake_perc"
        pg_inputs = np.array([[ice_speed_rpm, ice_torque, em2_torque_nm, brake_perc]])
        pg_in_scaled = self.pg_in_scaler.transform(pg_inputs).reshape(1, 1, -1)

        # 5. Predict PG Outputs
        pg_pred_scaled = self.pg_predict_main(pg_in_scaled)
        pg_pred = self.pg_out_scaler.inverse_transform(pg_pred_scaled.numpy()[0])

        # Outputs: Car_Speed_kmph, SOC_1
        car_speed = pg_pred[0][0]
        soc = pg_pred[0][1]

        # 6. Calculate Reward
        target_speed = self.df.loc[self.current_step, self.target_col_name()]

        speed_error = abs(target_speed - car_speed)
        soc_error = abs(soc - self.initial_soc)
        soc_error_squared = (soc - self.initial_soc) ** 2

        # Normalization factors to bring terms roughly into [0, 1] range
        norm_speed = 50.0  # Max expected practical speed error (km/h)
        norm_emission = 0.4  # Typical high combined tailpipe emissions (g/s)
        norm_fuel = 70.0  # Max fuel injection per step from config (mg)
        norm_brake = 100.0  # Max brake percentage bounds (%)
        norm_soc = 0.3  # Typical maximum allowed SOC drift scale

        # To cap penalties in case of hallucinations
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

        # 7. Update State
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

        soc_error = soc - self.initial_soc

        obs = np.array(
            [
                car_speed,
                next_speed_error,
                soc,
                ice_torque,
                nox_tp,
                float(engine_on),
                soc_error,
            ],
            dtype=np.float32,
        )

        info = {
            "time_s": self.df.loc[self.current_step, "time_s"],
            "target_speed": target_speed,
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
        # Helper to find the speed column
        if "car_speed_kmph" in self.df.columns:
            return (
                "car_speed_kmph"  # In WLTC.csv this is usually the target speed profile
            )
        return self.df.columns[1]  # Fallback
