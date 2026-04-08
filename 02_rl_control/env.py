import os
import sys
import glob
import random
import importlib
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

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

# bounds for soc because soc == 0 or 1 are invalid values that the plant lstms are not able to handle
_SOC_LOWER_BOUND = 0.02
_SOC_UPPER_BOUND = 0.98


def _resolve_model_backend(config_module):
    use_onnx = bool(getattr(config_module, "USE_ONNX", False))

    if use_onnx:
        try:
            from ONNX_Predict.utilities import load_network, set_states

            ice_dir = config_module.ICE_MODEL_DIR_ONNX
            drivetrain_dir = config_module.PG_MODEL_DIR_ONNX
            print(20 * "-", " USING ONNX ", 20 * "-")
            return use_onnx, load_network, set_states, ice_dir, drivetrain_dir
        except ImportError:
            print("Warning: ONNX_Predict.utilities not found. Falling back to TF2.18.")
            use_onnx = False

    try:
        from .utils.network_utils import load_network, set_states
    except ImportError:
        from utils.network_utils import load_network, set_states

    ice_dir = config_module.ICE_MODEL_DIR
    drivetrain_dir = config_module.PG_MODEL_DIR
    print(20 * "-", "USING TF2.18", 20 * "-")
    return use_onnx, load_network, set_states, ice_dir, drivetrain_dir


class EmissionControlEnv(gym.Env):
    """
    Gym Environment for controlling ICE and Drivetrain to minimize emissions and track speed.
    """

    _RANDOM_TARGET_EPISODE_LENGTH = 3600
    _SEGMENT_LENGTH = 600  # steps per target-speed segment (300 s at dt=0.5)

    def __init__(
        self,
        render_mode=None,
        dataset_path=None,
        config_module=None,
        random_target=False,
        fixed_target_speed=None,
        eval_mode=False,
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.eval_mode = eval_mode
        self.random_target = random_target
        self.fixed_target_speed = fixed_target_speed

        # If eval_mode or fixed_target_speed is True, we use random_target logic for scheduling
        if self.fixed_target_speed is not None or self.eval_mode:
            self.random_target = True

        self.target_speed = (
            0.0  # current segment target; updated by _get_current_target_speed
        )
        self.target_speed_schedule = []  # list of (start_step, speed) tuples
        self.config = config_module

        (
            self.use_onnx,
            self.load_network,
            self.set_states,
            self.ice_model_dir,
            self.drivetrain_model_dir,
        ) = _resolve_model_backend(self.config)

        # loading csvs with speed trajectories
        all_files = glob.glob(os.path.join(self.config.TRAIN_DATA_DIR, "*.csv"))
        self.data_files = [
            f
            # for f in all_files
            # if os.path.basename(f) in ("WLTC_high.csv", "WLTC_low.csv")
            for f in all_files
            if os.path.basename(f).startswith("drivetrain")
            or os.path.basename(f) in ("WLTC_high.csv", "WLTC_low.csv")
        ]
        if not self.data_files:
            raise ValueError(
                f"No valid CSV files found in {self.config.TRAIN_DATA_DIR}"
            )

        self.df = None
        self.max_steps = 0
        self.current_step = 0
        self.steps_since_last_engine_switch = 0
        self.last_ice_speed = 0.0
        self.last_fuel = 0.0

        print("Loading ICE Model...")
        self.ice_tuple = self.load_network(self.ice_model_dir)
        (
            self.ice_main,
            self.ice_init,
            self.ice_in_scaler,
            self.ice_out_scaler,
            self.ice_predict_main,
            self.ice_predict_init,
        ) = self.ice_tuple

        print("Loading Drivetrain (PG) Model...")
        self.pg_tuple = self.load_network(self.drivetrain_model_dir)
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
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )  # [ICE_Command, EM2_Torque, Fuel_Mass, Brake]

        # action scaling (Min/Max values for rescaling)
        # Note: ICE_Command is custom mapped in step(), its bounds just pass through [-1.0, 1.0]
        self.action_min = np.array([-1.0, -421.0, 3.0, 0.0], dtype=np.float32)
        self.action_max = np.array([1.0, 421.0, 70.0, 100.0], dtype=np.float32)

        # define observation space
        self.observation_space = spaces.Box(
            low=np.array(
                [-5.0, -155.0, 0.0, -50.0, 0.0, 900.0, 3.0, -1.0, 0.0], dtype=np.float32
            ),
            high=np.array(
                [150.0, 155.0, 1.0, 300.0, 0.4, 4000.0, 70.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )  # [Car_Speed (km/h), Speed_Error (km/h), SOC (0-1), ICE_Torque (Nm), NOx (g/step), ICE_Speed (rpm), Fuel (mg), SOC_Error (-1 to 1), Normalized_Timer (0-1)]

    def _get_current_target_speed(self):
        """Return the target speed for the current step from the schedule."""
        target = self.target_speed_schedule[0][1]  # fallback to first segment
        for start_step, speed in self.target_speed_schedule:
            if self.current_step >= start_step:
                target = speed
            else:
                break
        return target

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # load random or specific driving cycle
        if self.dataset_path is not None:
            chosen_file = self.dataset_path
        else:
            chosen_file = random.choice(self.data_files)

        self.df = pd.read_csv(chosen_file, delimiter=";", encoding="latin1")
        if self.df.shape[1] <= 1:
            self.df = pd.read_csv(chosen_file, delimiter=",", encoding="latin1")

        # clean column names
        self.df.columns = [col.strip() for col in self.df.columns]

        # episode length: fixed for random-target mode, CSV length otherwise
        if self.eval_mode and self.fixed_target_speed is None:
            self.max_steps = 3600
            self.target_speed_schedule = [
                (0, 50.0),
                (600, 70.0),
                (1200, 110.0),
                (1800, 140.0),
                (2400, 80.0),
                (3000, 35.0),
            ]
            self.target_speed = self.target_speed_schedule[0][1]
            schedule_str = ", ".join(
                f"{spd:.0f}" for _, spd in self.target_speed_schedule
            )
            print(f"Eval Mode Target speed schedule (km/h): [{schedule_str}]")
        elif self.random_target:
            self.max_steps = self._RANDOM_TARGET_EPISODE_LENGTH
            if self.fixed_target_speed is not None:
                # Single fixed target for the entire episode (evaluation mode)
                self.target_speed_schedule = [(0, self.fixed_target_speed)]
            else:
                # Multi-segment: new random target every _SEGMENT_LENGTH steps
                self.target_speed_schedule = []
                for seg_start in range(0, self.max_steps, self._SEGMENT_LENGTH):
                    self.target_speed_schedule.append(
                        (seg_start, random.uniform(0, 150))
                    )
            self.target_speed = self.target_speed_schedule[0][1]
            schedule_str = ", ".join(
                f"{spd:.0f}" for _, spd in self.target_speed_schedule
            )
            print(f"Target speed schedule (km/h): [{schedule_str}]")
        else:
            self.max_steps = len(self.df)
            print("\nUsing", str(chosen_file), " file.")

        # initialise models based on initial state in cycle
        ice_init_val_row = []
        for c, d in zip(_ICE_COLS, _ICE_DEFAULTS):
            if (
                not self.eval_mode
                and c in self.df.columns
                and not pd.isna(self.df.loc[0, c])
            ):
                ice_init_val_row.append(float(self.df.loc[0, c]))
            else:
                ice_init_val_row.append(d)

        ice_init_vals = np.array([ice_init_val_row], dtype=np.float32)
        ice_init_scaled = self.ice_out_scaler.transform(ice_init_vals).reshape(1, 1, -1)
        ice_states = self.ice_predict_init(ice_init_scaled)
        ice_states_dict = dict(zip(self.ice_init.output_names, ice_states))
        self.set_states(self.ice_main, ice_states_dict)

        pg_init_val_row = []
        for c, d in zip(_PG_COLS, _PG_DEFAULTS):
            if (
                not self.eval_mode
                and c in self.df.columns
                and not pd.isna(self.df.loc[0, c])
            ):
                pg_init_val_row.append(float(self.df.loc[0, c]))
            else:
                pg_init_val_row.append(d)

        pg_init_vals = np.array([pg_init_val_row], dtype=np.float32)
        pg_init_scaled = self.pg_out_scaler.transform(pg_init_vals).reshape(1, 1, -1)
        pg_states = self.pg_predict_init(pg_init_scaled)
        pg_states_dict = dict(zip(self.pg_init.output_names, pg_states))
        self.set_states(self.pg_main, pg_states_dict)

        # set initial state
        self.last_ice_torque = ice_init_val_row[0]  # ICE_Torque
        self.last_car_speed = pg_init_val_row[0]  # Car_Speed
        self.last_soc = pg_init_val_row[1]  # SOC
        self.initial_soc = self.last_soc
        self.last_nox = ice_init_val_row[6]  # nox_tp_gps
        self.last_engine_on = (
            True if ice_init_val_row[1] >= 0.1 else False
        )  # value by Markus to approximate initial state
        self.steps_since_last_engine_switch = 6  # Allow immediate switch on first step

        if self.random_target:
            target_speed = self.target_speed
        else:
            target_speed = self.df.loc[0, self.target_col_name()]

        obs = np.array(
            [
                self.last_car_speed,
                0.0,  # Initial speed error is exactly 0.0
                self.last_soc,
                self.last_ice_torque,
                self.last_nox,
                self.last_ice_speed,
                self.last_fuel,
                0.0,  # Initial SOC error is exactly 0.0
                min(float(self.steps_since_last_engine_switch) / 6.0, 1.0),
            ],
            dtype=np.float32,
        )

        info = {
            "time_s": 0 if self.random_target else self.df.loc[0, "time_s"],
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

        ice_command = scaled_action[0]
        em2_torque_nm = scaled_action[1]
        fuel_mg = scaled_action[2]
        brake_perc = scaled_action[3]

        if ice_command <= 0.0:
            engine_on = False
            ice_speed_rpm = 0.0
            fuel_mg = 0.0
        else:
            engine_on = True
            # Map top half (0.0, 1.0] to [900, 4000]
            ice_speed_rpm = 900.0 + (ice_command * (4000.0 - 900.0))

        # Prevent battery overcharging / overdischarging via Action Safety Filter
        # If battery is almost full, block charging (negative EM2 torque)
        if self.last_soc >= 0.98 and em2_torque_nm < 0.0:
            em2_torque_nm = 0.0
        # If battery is almost empty, block discharging (positive EM2 torque)
        elif self.last_soc <= 0.02 and em2_torque_nm > 0.0:
            em2_torque_nm = 0.0

        # Enforce minimum time in current engine state (on/off)
        if engine_on != self.last_engine_on and self.steps_since_last_engine_switch < 6:
            engine_on = self.last_engine_on
            if engine_on:
                ice_speed_rpm = 900.0
                fuel_mg = 3.0

        if engine_on == self.last_engine_on:
            self.steps_since_last_engine_switch += 1
        else:
            self.steps_since_last_engine_switch = 1

        if not engine_on:
            ice_speed_rpm = 0.0
            fuel_mg = 0.0

        # 2. Prepare Inputs for ICE Model
        # Inputs: "ICE_Speed_rpm", "fuel_mg", "T_amb_K", "p_amb_bar"
        if self.random_target:
            t_amb = 298.0
            p_amb = 1.005
        else:
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

        ice_inputs = np.array(
            [[ice_speed_rpm, fuel_mg, t_amb, p_amb]], dtype=np.float32
        )
        ice_in_scaled = self.ice_in_scaler.transform(ice_inputs).reshape(1, 1, -1)

        # 3. Predict ICE Outputs
        ice_pred_scaled = self.ice_predict_main(ice_in_scaled)
        if self.use_onnx:
            ice_pred = self.ice_out_scaler.inverse_transform(ice_pred_scaled[0])
        else:
            ice_pred = self.ice_out_scaler.inverse_transform(ice_pred_scaled.numpy()[0])

        # Index 0: ICE_Torque_Nm
        # Index 6: NOx_tp_gps (Tailpipe)
        ice_torque = ice_pred[0][0]
        nox_tp = ice_pred[0][6]

        # 4. Prepare Inputs for PG Model
        # Inputs: "ICE_Speed_rpm", "ICE: ICE_Torque_Nm", "EM2_Torque_Nm", "Brake_perc"
        pg_inputs = np.array(
            [[ice_speed_rpm, ice_torque, em2_torque_nm, brake_perc]], dtype=np.float32
        )
        pg_in_scaled = self.pg_in_scaler.transform(pg_inputs).reshape(1, 1, -1)

        # 5. Predict PG Outputs
        pg_pred_scaled = self.pg_predict_main(pg_in_scaled)
        if self.use_onnx:
            pg_pred = self.pg_out_scaler.inverse_transform(pg_pred_scaled[0])
        else:
            pg_pred = self.pg_out_scaler.inverse_transform(pg_pred_scaled.numpy()[0])

        # Outputs: Car_Speed_kmph, SOC_1
        car_speed = pg_pred[0][0]
        soc = pg_pred[0][1]

        # Strictly clip SOC mathematically to keep LSTM in known valid domain
        soc = np.clip(soc, 0.0001, 0.9999)

        # 6. Calculate Reward

        if self.random_target:
            target_speed = self._get_current_target_speed()
        else:
            target_speed = self.df.loc[self.current_step, self.target_col_name()]

        speed_error = abs(target_speed - car_speed)
        soc_error = abs(soc - self.initial_soc)
        soc_error_squared = (soc - self.initial_soc) ** 2

        # Normalization factors to bring terms roughly into [0, 1] range
        norm_speed = 155.0  # Max expected practical speed error (km/h)
        norm_emission = 0.4  # Typical high combined tailpipe emissions (g/s)
        norm_fuel = 70.0  # Max fuel injection per step from config (mg)
        norm_brake = 100.0  # Max brake percentage bounds (%)

        # To cap penalties in case of hallucinations
        safe_speed_penalty = min(speed_error / norm_speed, 1.0)
        safe_emission_penalty = min(nox_tp / norm_emission, 1.0)

        reward = 0.0

        ## DO NOT EDIT REWARDS HERE; INSTEAD IN config.py!!!!

        # Scale factor dictates how wide the "bell" is.
        # A scale of 10.0 means at 10 km/h error, the reward drops significantly.
        scale_factor = 10.0

        reward += self.config.W_SPEED * np.exp(
            -0.5 * (speed_error / scale_factor) ** 2
        )  # gaussian reward
        reward -= self.config.W_EMISSION * safe_emission_penalty
        reward -= self.config.W_FUEL * (fuel_mg / norm_fuel)
        reward -= self.config.W_BRAKE * (brake_perc / norm_brake)
        reward -= self.config.W_SOC * soc_error
        reward -= self.config.W_SOC_SQUARED * soc_error_squared

        if engine_on and not self.last_engine_on:
            reward -= self.config.W_FLICKER

        # 7. Update State
        self.current_step += 1
        self.last_ice_torque = ice_torque
        self.last_car_speed = car_speed
        self.last_soc = soc
        self.last_nox = nox_tp
        self.last_engine_on = engine_on
        self.last_ice_speed = ice_speed_rpm
        self.last_fuel = fuel_mg

        terminated = False
        truncated = False

        if self.current_step >= self.max_steps - 1:
            terminated = True

        if self.random_target:
            next_target_speed = (
                self._get_current_target_speed() if not terminated else 0.0
            )
        else:
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
                ice_speed_rpm,
                fuel_mg,
                soc_error,
                min(float(self.steps_since_last_engine_switch) / 6.0, 1.0),
            ],
            dtype=np.float32,
        )

        info = {
            "time_s": (
                self.current_step
                if self.random_target
                else self.df.loc[self.current_step, "time_s"]
            ),
            "target_speed": target_speed,
            "speed_error": speed_error,
            "nox": nox_tp,
            "fuel": fuel_mg,
            "engine_on": engine_on,
            "steps_since_last_engine_switch": self.steps_since_last_engine_switch,
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
