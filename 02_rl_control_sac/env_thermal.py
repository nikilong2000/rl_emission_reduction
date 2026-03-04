"""
env_thermal.py — EmissionControlEnv extended with 3 thermal observation variables.

Thermal variable selection is based on PCA of aftertreatment temperatures
(see internal_lstm_models/thermal_analysis_results/summary.txt):

  PC1 (85.9% variance): general thermal level — represented by T_gas_eo_K
  PC2 ( 9.0% variance): DPF substrate dynamics — represented by T_Sub_DPF_K
  PC3 ( 3.6% variance): engine-out vs tailpipe contrast — represented by T_gas_tp_K

Together these 3 variables cover ~95% of the total thermal state variance.

ICE model output indices (from NN_Application/config.txt):
  5  → T_gas_eo_K   (exhaust gas temperature, engine-out)
  12 → T_Sub_DPF_K  (DPF substrate wall temperature)
  15 → T_gas_tp_K   (exhaust gas temperature, tailpipe)
"""

import numpy as np

try:
    from .env import EmissionControlEnv
except ImportError:
    from env import EmissionControlEnv


# Physical temperature bounds [K] for normalization
_T_GAS_EO_LOW  = 298.0
_T_GAS_EO_HIGH = 1200.0
_T_SUB_DPF_LOW  = 298.0
_T_SUB_DPF_HIGH = 1000.0
_T_GAS_TP_LOW  = 298.0
_T_GAS_TP_HIGH = 900.0

# ICE model output indices for the three selected thermal variables
_IDX_T_GAS_EO  = 5
_IDX_T_SUB_DPF = 12
_IDX_T_GAS_TP  = 15


class EmissionControlEnvThermal(EmissionControlEnv):
    """
    Extends EmissionControlEnv with three thermal observation variables that
    together capture ~95% of the aftertreatment thermal state (2 PCA components
    reach 90%, 3 reach 95%).

    Observation space (10 dimensions):
      [0] Car_Speed_kmph   — vehicle speed
      [1] Speed_Error      — target − current speed
      [2] SOC              — battery state-of-charge
      [3] ICE_Torque_Nm    — engine torque
      [4] NOx_tp_gps       — tailpipe NOx emission rate
      [5] Engine_On        — binary engine state
      [6] SOC_Error        — SOC drift from initial
      [7] T_gas_eo_K       — exhaust gas temp at engine-out  (PC1 representative)
      [8] T_Sub_DPF_K      — DPF substrate temperature       (PC2 representative)
      [9] T_gas_tp_K       — exhaust gas temp at tailpipe    (PC3 representative)
    """

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

        # Extend obs bounds with the three thermal variables
        self.obs_low  = np.append(self.obs_low,  [_T_GAS_EO_LOW,  _T_SUB_DPF_LOW,  _T_GAS_TP_LOW ]).astype(np.float32)
        self.obs_high = np.append(self.obs_high, [_T_GAS_EO_HIGH, _T_SUB_DPF_HIGH, _T_GAS_TP_HIGH]).astype(np.float32)

        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=self.obs_low, high=self.obs_high, dtype=np.float32
        )

        # Thermal state (K), initialised to ambient temperature
        self.last_t_gas_eo  = 298.0
        self.last_t_sub_dpf = 298.0
        self.last_t_gas_tp  = 298.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_raw_obs(self, car_speed, speed_error, soc, ice_torque,
                       nox_tp, engine_on, soc_error,
                       t_gas_eo, t_sub_dpf, t_gas_tp):
        return np.array(
            [car_speed, speed_error, soc, ice_torque, nox_tp,
             float(engine_on), soc_error,
             t_gas_eo, t_sub_dpf, t_gas_tp],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Reset thermal state
        self.last_t_gas_eo  = 298.0
        self.last_t_sub_dpf = 298.0
        self.last_t_gas_tp  = 298.0

        # Rebuild obs including thermal variables (all at ambient on reset)
        obs = np.append(info["raw_obs"],
                        [self.last_t_gas_eo,
                         self.last_t_sub_dpf,
                         self.last_t_gas_tp]).astype(np.float32)
        info["raw_obs"] = obs
        return obs, info

    def step(self, action):
        # NOTE: We do NOT call super().step() here because the ICE model is
        # stateful (LSTM). Calling the base step and then re-running the ICE
        # prediction would advance the hidden states twice. Instead we duplicate
        # the base logic so the ICE model runs exactly once per step and we can
        # read the full 16-output prediction including thermal variables.

        # 1. Rescale action from [-1, 1] to physical range
        scaled_action = self.action_min + (action + 1.0) * 0.5 * (
            self.action_max - self.action_min
        )
        engine_state_req = scaled_action[0]
        ice_speed_rpm    = scaled_action[1]
        em2_torque_nm    = scaled_action[2]
        fuel_mg          = scaled_action[3]
        brake_perc       = scaled_action[4]

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

        # 3. ICE prediction (single call — advances LSTM state once)
        ice_inputs    = np.array([[ice_speed_rpm, fuel_mg, t_amb, p_amb]])
        ice_in_scaled = self.ice_in_scaler.transform(ice_inputs).reshape(1, 1, -1)
        ice_pred_scaled = self.ice_predict_main(ice_in_scaled)
        ice_pred = self.ice_out_scaler.inverse_transform(ice_pred_scaled.numpy()[0])

        ice_torque = ice_pred[0][0]
        nox_tp     = ice_pred[0][6]

        # Extract thermal variables from the full ICE output
        if engine_on:
            t_gas_eo  = float(ice_pred[0][_IDX_T_GAS_EO])
            t_sub_dpf = float(ice_pred[0][_IDX_T_SUB_DPF])
            t_gas_tp  = float(ice_pred[0][_IDX_T_GAS_TP])
        else:
            # Engine off: exponential decay toward ambient (~0.5% per second)
            alpha = 0.995
            t_gas_eo  = alpha * self.last_t_gas_eo  + (1.0 - alpha) * t_amb
            t_sub_dpf = alpha * self.last_t_sub_dpf + (1.0 - alpha) * t_amb
            t_gas_tp  = alpha * self.last_t_gas_tp  + (1.0 - alpha) * t_amb

        # 4. Drivetrain prediction
        pg_inputs    = np.array([[ice_speed_rpm, ice_torque, em2_torque_nm, brake_perc]])
        pg_in_scaled = self.pg_in_scaler.transform(pg_inputs).reshape(1, 1, -1)
        pg_pred_scaled = self.pg_predict_main(pg_in_scaled)
        pg_pred = self.pg_out_scaler.inverse_transform(pg_pred_scaled.numpy()[0])

        car_speed = pg_pred[0][0]
        soc       = pg_pred[0][1]

        # 5. Reward (identical to base class)
        try:
            from . import config
        except ImportError:
            import config

        target_speed       = self.df.loc[self.current_step, self.target_col_name()]
        speed_error        = abs(target_speed - car_speed)
        soc_error          = abs(soc - self.initial_soc)
        soc_error_squared  = (soc - self.initial_soc) ** 2

        norm_speed    = 50.0
        norm_emission = 0.1
        norm_fuel     = 70.0
        norm_brake    = 100.0

        safe_speed_penalty    = min(speed_error / norm_speed,    1.0)
        safe_emission_penalty = min(nox_tp      / norm_emission, 1.0)

        reward  = 0.0
        reward -= config.W_SPEED    * safe_speed_penalty
        reward -= config.W_EMISSION * safe_emission_penalty
        reward -= config.W_FUEL     * (fuel_mg   / norm_fuel)
        reward -= config.W_BRAKE    * (brake_perc / norm_brake)
        reward -= config.W_SOC      * soc_error
        reward -= config.W_SOC_SQUARED * soc_error_squared

        if engine_on and not self.last_engine_on:
            reward -= config.W_FLICKER

        # 6. Update state
        self.current_step      += 1
        self.last_ice_torque    = ice_torque
        self.last_car_speed     = car_speed
        self.last_soc           = soc
        self.last_nox           = nox_tp
        self.last_engine_on     = engine_on
        self.last_t_gas_eo      = t_gas_eo
        self.last_t_sub_dpf     = t_sub_dpf
        self.last_t_gas_tp      = t_gas_tp

        terminated = False
        truncated  = False
        if self.current_step >= self.max_steps - 1:
            terminated = True

        next_target_speed = (
            self.df.loc[self.current_step, self.target_col_name()]
            if not terminated
            else 0.0
        )
        next_speed_error = next_target_speed - car_speed
        soc_error_signed = soc - self.initial_soc

        obs = self._build_raw_obs(
            car_speed, next_speed_error, soc, ice_torque, nox_tp,
            engine_on, soc_error_signed,
            t_gas_eo, t_sub_dpf, t_gas_tp,
        )

        info = {
            "time_s":       self.df.loc[self.current_step, self.col_map["time"]],
            "speed_error":  speed_error,
            "nox":          nox_tp,
            "fuel":         fuel_mg,
            "engine_on":    engine_on,
            "ice_torque":   ice_torque,
            "ice_speed_rpm": ice_speed_rpm,
            "em2_torque_nm": em2_torque_nm,
            "brake_perc":   brake_perc,
            "raw_obs":      obs,
            "t_gas_eo_K":   t_gas_eo,
            "t_sub_dpf_K":  t_sub_dpf,
            "t_gas_tp_K":   t_gas_tp,
        }

        return obs, reward, terminated, truncated, info
