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

        # Modelo de transición (externo)
        self.transition_function_model = transition_function_model

        # error maximo  #Correcte
        self.MAX_POSSIBLE_ERROR = (
            80  # -5min vel y max vel 0  = # (200.0 - (-5.0)) 205.0
        )

        # estabilidad  [se tiene que pasar el km/h a rango del error]
        ## Calcular qué recompensa corresponde a ese umbral de error
        norm_stable_error = stable_error_threshold / self.MAX_POSSIBLE_ERROR

        ##Este es el nuevo umbral de recompensa (×10 porque get_reward escala por 10)
        self.stable_reward_threshold = (1.0 - (norm_stable_error**2)) * 10

        # Configuración del episodio
        self.length_problem = length_problem
        self.n_stable = n_stable

        # Coeficientes para la recompensa
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        # Condiciones ambiente
        self.p_amb_bar = 1.0
        self.T_amb_K = 298.0

        self.stable_counter = 0

    def step(self, action, vel_target: float = 70.0):  # Correcte
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
        # 1) APLICAR LA ACCIÓN
        # ------------------------------------------------------------------
        delta_mf, delta_brk, delta_ice_sp = action
        self.mf = float(delta_mf)
        self.brk = float(delta_brk)
        self.ice_sp = float(delta_ice_sp)

        # ------------------------------------------------------------------
        # 2) PREDICCIÓN DEL ICE
        # ------------------------------------------------------------------
        torque_ICE_tf, nox_tf, _, co_tf, _ = self.transition_function_model.predict_ice(
            self.ice_sp, self.mf, self.T_amb_K, self.p_amb_bar
        )

        # → pasamos los tensores de TF a float
        self.torque_ICE = float(torque_ICE_tf)
        self.nox = float(nox_tf)
        self.co = float(co_tf)

        # ------------------------------------------------------------------
        # 3) ICE CLIPPING
        # -----------------------------------------------------------------

        self.torque_ICE = np.clip(self.torque_ICE, -50, 300.0)

        # Apagar el ICE si la velocidad de giro es baja
        if self.ice_sp < 900.0:
            self.torque_ICE = 0.0
            self.mf = 3.0

        # ------------------------------------------------------------------
        # 4) PREDICCIÓN DEL POWER‑SPLIT (PG / EM2)
        # ------------------------------------------------------------------
        vel_out_tf, _ = self.transition_function_model.predict_PG(
            self.ice_sp, 0.0, self.torque_ICE, self.brk
        )

        self.vel_out = float(vel_out_tf)
        self.vel = self.vel_out  # guardamos velocidad actual

        # ------------------------------------------------------------------
        # 5) NUEVO ESTADO + ERROR NORMALIZADO
        # ------------------------------------------------------------------
        error_normalized = (
            vel_target - self.vel
        ) / 70.0  # Normalizado por vel_target máximo
        new_state = (
            vel_target,
            self.vel,
            self.mf,
            self.brk,
            self.ice_sp,
            error_normalized,
        )

        reward = self.get_reward(vel_target)

        # ------------------------------------------------------------------
        # 6) CONDICIONES DE TERMINACIÓN (Margen del 5% del target)
        # ------------------------------------------------------------------
        # Threshold dinámico: 5% de margen proporcional al target
        # Para target=70 → margen 3.5 km/h → reward >= 9.5
        # Para target=10 → margen 0.5 km/h → reward >= 9.5
        margin_percent = 0.05  # 5%
        # Reward cuando error = margin_percent * target:
        # reward = (1 - margin_percent) * 10 = 9.5
        dynamic_threshold = (1.0 - margin_percent) * 10

        if reward >= dynamic_threshold:
            self.stable_counter += 1
        else:
            self.stable_counter = 0

        terminated = self.stable_counter >= self.n_stable

        self.step_count += 1

        truncated = self.step_count >= self.length_problem

        # ------------------------------------------------------------------
        return new_state, reward, terminated, truncated

    def get_reward(self, vel_target: float) -> float:
        # 1. Error lineal directo (sin cuadrática, sin umbrales ocultos)
        error = vel_target - self.vel_out
        abs_error = abs(error)

        # Normalizamos usando el propio target.
        # Si vel=0 -> error=70 -> reward=0.0
        # Si vel=70 -> error=0 -> reward=1.0
        # Usamos max(vel_target, 1.0) para evitar división por cero
        denom = max(vel_target, 1.0)
        reward = 1.0 - (abs_error / denom)

        # 2. Penalización por Freno (Se mantiene igual)
        if self.vel_out < (vel_target * 0.95) and self.brk > 1.0:
            penalty = 0.5 * (self.brk / 100.0)
            reward -= penalty

        return reward * 10

    def reset(self, vel_target=70):  # Correcte
        """
        Resets the environment to a random initial state and returns it.

        Returns:
            tuple: The initial state of the environment.
        """
        # Resets the transition function model
        self.transition_function_model.reset_models()

        # Generates a random initial state
        self.reset_variables()

        # Resets the transition function model, it is necessary because in the previous step ".reset_variables()" a PG iteration is performed
        self.transition_function_model.reset_models()

        # Counters
        self.step_count = 0
        self.stable_counter = 0

        # CHANGE: Return 6 dimensions including error_normalized
        error_normalized = (vel_target - self.vel) / 70.0
        return vel_target, self.vel, self.mf, self.brk, self.ice_sp, error_normalized

    def reset_variables(self):  # Correcte
        """
        Sets the environment's state variables to a random initial configuration.
        """
        # Genera un estado inicial aleatorio
        self.sample_init_state()

        # Calciula la velocidad inicial
        self.vel, _ = self.transition_function_model.predict_PG(
            self.ice_sp, self.EM2, self.torque, self.brk
        )

    def sample_init_state(self, folder_path="../src/data"):
        """
        Samples an initial state from a random row in a random CSV file.

        This method uses reservoir sampling to select a row in a single pass,
        which is memory-efficient for large files.
        """
        # 1) Listar CSVs
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No se encontraron archivos CSV en {folder_path}")

        chosen_file = os.path.join(folder_path, random.choice(csv_files))

        # 2) Reservoir sampling: escoge 1 fila al azar en una pasada
        chosen_row = None
        with open(chosen_file, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                # con probabilidad 1/i reemplazo la fila elegida
                if random.random() < 1 / i:
                    chosen_row = row

        if chosen_row is None:
            raise ValueError(f"El archivo {chosen_file} no tiene filas de datos")

        # 3) Asignar a atributos
        self.mf = float(chosen_row["fuel"])
        self.brk = float(chosen_row["Brake"])
        self.ice_sp = float(chosen_row["ICE_Speed_soll"])
        self.EM2 = float(chosen_row["EM2_Torque"])
        self.torque = float(chosen_row["ICE_Torque_pred"])
