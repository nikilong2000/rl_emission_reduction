import os  # Operative system
import numpy as np  # Mathematics: arrays
import pandas as pd
import math  # Mathematics-simple


class PredictiveController:
    def __init__(self):
        return

    def load_cycle(self, str_dir_cycles, file_cycle):
        self.RDE_filename = os.path.join(str_dir_cycles, file_cycle)
        self.tab_cycle = pd.read_csv(self.RDE_filename)

        self.tab_cycle.drop(self.tab_cycle.columns[[0]], axis=1, inplace=True)
        self.tab_cycle.info()

        # print(self.tab_cycle)

        self.vect_time = np.array(self.tab_cycle["T"])
        self.vect_v_now = np.array(self.tab_cycle["Speed"])
        self.vect_v_nxt = np.array(self.tab_cycle["Speed.1"])
        self.vect_GP_pc = np.array(self.tab_cycle["grade"])

        self.vect_errors = self.vect_v_nxt - self.vect_v_now
        self.vect_ac = (
            self.vect_errors / 3.6
        )  # Acceleration in SI units, timestep is 1s

        self.Ncase = len(self.vect_GP_pc)
        self.max_acc = max(self.vect_ac)
        return

    def get_vect_time_cycle(self):
        return self.vect_time

    def get_Ncase(self):
        return self.Ncase

    def set_gains(
        self, K_split, PCI, rend
    ):  # default start values for first engine run
        self.K_split = K_split
        self.K_power_ctrl = 100 / self.max_acc
        self.PCI = PCI
        self.rend = rend
        self.K_eng_ctrl = 1 / (self.PCI * self.rend)
        return

    def box_power_ctrl(self):  # Look, it is independent of GP_pc
        self.ac_ped = max(0, self.acc) * self.K_power_ctrl
        return

    def power_dynamics(self):
        # as a long function I prefer to unload and load interface from/to the class
        slope = self.GP
        v_k = self.v_k
        acc = self.acc

        g = 9.81  # Gravity, m/s^2
        mass = 1750  # Vehicle mass, Kg
        f = 0.01  # Rolling coefficient, N·m
        Cd = 0.26
        A = 2.33
        Rroll = 0.3
        theta = math.atan(slope / 100)  # Slope, from % to rad
        v_ms = v_k / 3.6  # Speed, from km/h to m/s
        w_whl = v_ms / Rroll  # Rotating speed of the axle, based on v_k, rad/s

        T_whl = Rroll * (
            mass * acc
            + mass * g * f * math.cos(theta)
            + mass * g * math.sin(theta)
            + Cd * A * (v_ms) ** 2 / 21.15
        )  # Torque, N·m
        P_req = T_whl * w_whl / 1000  # Required Power, kW

        self.w_whl = w_whl
        self.T_whl = T_whl
        self.P_req = P_req

        return

    def box_scaling_RP(self):
        self.P_ice = self.P_req * self.K_split  ## ICE Required Power, kW

        return

    def box_engine_ctrl(self):
        self.m_inj = max(
            0, self.P_ice * self.rend
        )  # Injection mass in grams per second (gps)

        return

    def if_then_ctrl(self):
        ac_ped = self.ac_ped
        v_current = self.v_k

        if ((ac_ped < 15.0) and (v_current >= 10.0)) or (
            (ac_ped > 15.0) and (v_current >= 10.0)
        ):
            n_ice_sp = 2100
        else:
            n_ice_sp = 1800

        self.n_ice_sp = n_ice_sp

        return

    def EMx_power(self):
        # as a long function I prefer to unload and load interface from/to the class

        T_whl = self.T_whl
        w_whl = self.w_whl
        n_ice_sp = self.n_ice_sp

        k1 = 1.92
        k2 = 2.6
        kf = 4.1

        w_ice = n_ice_sp * 2 * math.pi / 60  # rad/s
        T_ice = self.P_ice * 1000.0 / w_ice

        T_EM2 = (
            1 / (k1 + k2 + 1) * (T_whl * k1 / kf - (k1 + 1) * T_ice)
        )  # Formulas from the paper, steady-state eCVT2 with EM1 as speeder
        T_EM1 = 1 / k1 * T_ice + (k2 + 1) / k1 * T_EM2
        w_C1R2 = kf * w_whl
        w_EM1 = (k1 + 1) * w_C1R2 - k1 * w_ice
        w_EM2 = (k1 + k2 + 1) / k1 * w_C1R2 - (k2 + 1) / k1 * w_C1R2
        P_EM1 = w_EM1 * T_EM1 * 0.001
        P_EM2 = w_EM2 * T_EM2 * 0.001

        self.P_EM1 = P_EM1
        self.P_EM2 = P_EM2

        return

    def control_loop(self, c):
        self.GP = self.vect_GP_pc[c]
        self.v_k = self.vect_v_now[c]
        self.v_sp = self.vect_v_nxt[c]
        self.acc = (self.v_sp - self.v_k) / 3.6
        self.dt = self.vect_time[c] - self.vect_time[c - 1]

        self.box_power_ctrl()
        self.power_dynamics()
        self.if_then_ctrl()
        self.box_scaling_RP()
        self.box_engine_ctrl()
        self.EMx_power()

        return self.v_k, self.n_ice_sp, self.P_EM1, self.P_EM2, self.GP, self.m_inj

    def get_plot_data(self):
        return [self.P_req, self.ac_ped, self.P_ice]
