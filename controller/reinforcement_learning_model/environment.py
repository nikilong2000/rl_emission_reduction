
import csv
import torch
import numpy as np
import torch.nn.functional as F
import math
import random
import pandas as pd
import os


class Environment:
    """
    Manages the Reinforcement Learning environment for a vehicle control task.
    """

    def __init__(self, transition_function_model, length_problem: int = 1200,alpha: float = 1.0, beta: float = 0.0, gamma: float = 0.0, stable_velocity: float = 0.3, n_stable: int = 50):
        """
        Initializes the environment.

        Args:
            transition_function_model: The external model used to predict state transitions.
            length_problem (int): The maximum number of steps per episode before it's truncated.
            alpha (float): The coefficient for the velocity error component of the reward.
            beta (float): Unused reward coefficient.
            gamma (float): Unused reward coefficient.
            stable_velocity (float): A threshold used to determine if the velocity is stable.
                                     The episode terminates if the reward remains above this threshold
                                     for `n_stable` consecutive steps.
            n_stable (int): The number of consecutive steps the velocity must be stable for the
                            episode to terminate.
        """

        # Modelo de transición (externo)
        self.transition_function_model = transition_function_model

        # Configuración del episodio
        self.length_problem = length_problem
        self.stable_velocity = -stable_velocity**2
        self.n_stable = n_stable

        # Coeficientes para la recompensa
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        # Condiciones ambiente
        self.p_amb_bar = 1.0
        self.T_amb_K = 298.0

        # Variables de estado (se rellenan en reset)
        self.step_count = 0
        self.stable_counter = 0


    def step(self, action, vel_target: float = 70.0):
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
#         print(self.step_count)

        # ------------------------------------------------------------------
        # 1) APLICAR LA ACCIÓN
        # ------------------------------------------------------------------
        delta_mf, delta_brk, delta_ice_sp = action
        self.mf     = float(delta_mf)
        self.brk    = float(delta_brk)
        self.ice_sp = float(delta_ice_sp)

 
#         print(f"Mf: {self.mf}, "f"Brk: {self.brk}, "f"ICE_sp: {self.ice_sp}")


        # ------------------------------------------------------------------
        # 2) PREDICCIÓN DEL ICE
        # ------------------------------------------------------------------
        torque_ICE_tf, nox_tf, _, co_tf, _ = self.transition_function_model.predict_ice(
            self.ice_sp, self.mf, self.T_amb_K, self.p_amb_bar
        )

        # → pasamos los tensores de TF a float
        self.torque_ICE = float(torque_ICE_tf.numpy())
        self.nox    = float(nox_tf.numpy())
        self.co     = float(co_tf.numpy())
        


        # ------------------------------------------------------------------
        # 3) ICE CLIPPING 
        # -----------------------------------------------------------------
       
        self.torque_ICE = np.clip(self.torque_ICE, -50, 300.0)

    
        # Apagar el ICE si la velocidad de giro es baja
        if self.ice_sp < 900.0:
            self.torque_ICE = 0.0
            self.mf     = 3.0



        # ------------------------------------------------------------------
        # 4) PREDICCIÓN DEL POWER‑SPLIT (PG / EM2)
        # ------------------------------------------------------------------        
        vel_out_tf, _ = self.transition_function_model.predict_PG(
            self.ice_sp, 0.0, self.torque_ICE, self.brk
        )
        self.vel_out = float(vel_out_tf.numpy())
        self.vel     = self.vel_out          # guardamos velocidad actual
        
#         print(f"vel_out: {self.vel_out}")

        # ------------------------------------------------------------------
        # 5) NUEVO ESTADO Y RECOMPENSA
        # ------------------------------------------------------------------
        new_state = (vel_target, self.vel, self.mf, self.brk, self.ice_sp)
        reward    = self.get_reward(vel_target)

        # ------------------------------------------------------------------
        # 6) CONDICIONES DE TERMINACIÓN
        # ------------------------------------------------------------------
        if reward >= self.stable_velocity:
            self.stable_counter += 1
        else:
            self.stable_counter = 0
            
        terminated = self.stable_counter >= self.n_stable
        
#         if terminated:
#             print(f"self.stable_velocity: {self.stable_velocity}, reward: {reward}")
#             print(f"vel_out:{self.vel_out}    |    reward:{reward}")
#             print("terminated")

        self.step_count += 1
        
        truncated  = self.step_count >= self.length_problem
    
#         if truncated:
#             print(f"vel_out:{self.vel_out}    |    reward:{reward}")
#             print("truncated")

        # ------------------------------------------------------------------
        return new_state, reward, terminated, truncated

    
    def get_reward(self, vel_target: float) -> float:
        """
        Calculates the reward for the current state.

        The reward is a penalty based on the squared error between the target
        and current velocities. A larger error results in a more negative reward.
        """
        error = vel_target - self.vel_out            # escalar float
        reward = -self.alpha * (error ** 2)           # cuadrado del error
        return reward
    
    # En Environment.py

    def get_reward(self, vel_target: float) -> float:
        """
        Calculates the reward for the current state.
        """
        error = vel_target - self.vel_out

        max_possible_error = 100.0 
        normalized_error = error / max_possible_error
        reward = -self.alpha * (normalized_error ** 2) # Ahora la recompensa estará entre [-alpha, 0]

        # Opción B: Simplemente usar un factor de escala más agresivo
        # reward = -0.001 * (error ** 2) 

        return reward

    def reset(self, vel_target=70):
        """
        Resets the environment to a random initial state and returns it.

        Returns:
            tuple: The initial state of the environment.
        """
        # Reinicia el modelo de función de transición
        self.transition_function_model.reset_models()

        # Genera un estado inicial aleatorio
        self.reset_variables()

        # Reinicia el modelo de función de transición
        self.transition_function_model.reset_models()
        
        # Contadores
        self.step_count = 0
        self.stable_counter = 0

        return vel_target, self.vel, self.mf, self.brk, self.ice_sp


    def reset_variables(self):
        """
        Sets the environment's state variables to a random initial configuration.
        """
        # Genera un estado inicial aleatorio
        self.sample_init_state()

        # Calciula la velocidad inicial
        self.vel, _ = self.transition_function_model.predict_PG(self.ice_sp, self.EM2, self.torque, self.brk)

    def sample_init_state(self, folder_path="../src/data"):
        """
        Samples an initial state from a random row in a random CSV file.

        This method uses reservoir sampling to select a row in a single pass,
        which is memory-efficient for large files.
        """
        # 1) Listar CSVs
        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(
                f"No se encontraron archivos CSV en {folder_path}")

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
#         print(f"reset --> mf: {self.mf}, brk: {self.brk}, ice_sp: {self.ice_sp}, EM2: {self.EM2}, torque: {self.torque}")
        
        
