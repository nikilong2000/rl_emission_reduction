import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
from ActorCriticNetworks import ActorNetwork, CriticNetwork, copy_target, soft_update
from ReplayBuffer import ReplayBuffer
import numpy as np
from Noise import NormalActionNoise
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
import time  # To measure time
import glob
import json


import warnings

warnings.filterwarnings("ignore")

from environment import Environment

# MODIFICATION: Import Ray
import ray

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@ray.remote
class RolloutWorker:
    """
    This class represents an "Actor" that runs in a separate CPU process.
    Its sole responsibility is to interact with a copy of the environment to generate
    experiences (sequence windows).
    """

    def __init__(
        self,
        f_transicio,
        worker_id,
        scaler_params,
        act_dim,
        length_problem,
        vel_target,
        B,
        U,
    ):
        self._silence_tf_in_worker()
        self.worker_id = worker_id
        self.env = Environment(f_transicio, length_problem=length_problem)
        self.exploration_noise = NormalActionNoise(mean=np.zeros(act_dim), sigma=0.1)
        self.vel_target = vel_target

        # Parameters for slicing sequences
        self.B, self.U = B, U
        self.S = B + U

        # Descaling parameters
        self._action_scale = scaler_params["scale"][2:5]
        self._action_min = scaler_params["min"][2:5]

        # The actor is created here, but its weights will be updated from the Learner
        self.actor_worker = ActorNetwork(scaler_params=scaler_params).to("cpu")
        print(f"[Worker {self.worker_id}] Created and ready.")

    def get_weights(self):
        """Returns the current weights of the worker's actor."""
        return self.actor_worker.state_dict()

    def set_weights(self, weights):
        """Updates the worker's actor weights with those from the learner."""
        self.actor_worker.load_state_dict(weights)
        self.actor_worker.reset_states()

    def choose_action(self, s_raw):
        """Chooses an action using the local actor (on CPU)."""
        state = torch.tensor(s_raw, dtype=torch.float32).unsqueeze(0).to("cpu")
        self.actor_worker.eval()
        with torch.no_grad():
            # Unpack the tuple and keep only the actions.
            actions, _ = self.actor_worker(state)
            a = actions.cpu().squeeze(0).numpy()

        self.actor_worker.train()
        return a

    def descale_action(self, action_scaled):
        action_scaled_np = np.asarray(action_scaled, dtype=np.float32)
        return (action_scaled_np - self._action_min) / self._action_scale

    def _silence_tf_in_worker(self):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        try:
            import tensorflow as tf

            tf.get_logger().setLevel("ERROR")
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    def collect_windows(self):
        """
        Main function of the worker: it runs a full episode, slices it
        and returns the generated windows.
        """
        # --- Episode execution logic ---
        obs = self.env.reset()
        self.actor_worker.reset_states()
        episode_data = {
            k: []
            for k in [
                "states",
                "actions",
                "rewards",
                "next_states",
                "terminated",
                "truncated",
            ]
        }

        while True:
            action_scaled = self.choose_action(obs)
            noise_value = self.exploration_noise.sample()
            action_scaled_noisy = np.clip(action_scaled + noise_value, -1, 1)
            action_phys = self.descale_action(action_scaled_noisy)

            next_obs, reward, terminated, truncated = self.env.step(
                action_phys, self.vel_target
            )

            #             print(f"action: {action_scaled} + noise: {noise_value} = action_noisy: {action_scaled_noisy}")
            #             print(f"actions real value: {action_phys}")
            #             print(f"For a speed of {next_obs[1]} it has a Reward of = {reward}")
            #             print(f"next_obs: {next_obs}, reward: {reward}, terminated: {terminated}, truncated: {truncated}")

            #             print("==========================================")

            done = terminated or truncated

            for k, v in zip(
                episode_data.keys(),
                [obs, action_scaled_noisy, reward, next_obs, terminated, truncated],
            ):
                episode_data[k].append(v)

            obs = next_obs
            if done:
                break

        # --- Logic to slice the episode into windows ---
        windows = []
        episode_len = len(episode_data["states"])
        for i in range(0, max(0, episode_len - self.S + 1), self.U):
            s_window = np.stack(episode_data["states"][i : i + self.S]).astype(
                np.float32
            )
            a_window = np.stack(episode_data["actions"][i : i + self.S]).astype(
                np.float32
            )
            r_window = np.asarray(
                episode_data["rewards"][i : i + self.S], dtype=np.float32
            )
            ns_window = np.stack(episode_data["next_states"][i : i + self.S]).astype(
                np.float32
            )
            t_window = np.asarray(
                episode_data["terminated"][i : i + self.S], dtype=np.bool_
            )
            tr_window = np.asarray(
                episode_data["truncated"][i : i + self.S], dtype=np.bool_
            )
            windows.append(
                (s_window, a_window, r_window, ns_window, t_window, tr_window)
            )

        return windows


class TD3:
    """The TD3 Agent (Learner)."""

    def __init__(
        self,
        f_transicio,
        version="test",
        act_dim=3,
        obs_dim=5,
        length_problem=1200,
        replay_size=1000000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        early_stop=50,
        numeric_eval=False,
        scaler_params=None,
        vel_target=70,
        num_workers=2,
        U=64,
        B=32,
        reuse_warmup_buffer=False,
        buffer_dir="buffer",
    ):
        """
        Initializes the Recurrent TD3 (RTD3) agent.

        Parameters
        ----------
        f_transicio : callable
            The environment's transition function that simulates a step.
        version : str, optional
            Version name to organize results into folders, defaults to "test".
        act_dim : int, optional
            Dimension of the action space, defaults to 3.
        obs_dim : int, optional
            Dimension of the observation space, defaults to 5.
        length_problem : int, optional
            Maximum length of an episode in the environment, defaults to 1200.
        replay_size : int, optional
            Maximum size of the replay buffer (experience replay), defaults to 1000000.
        batch_size : int, optional
            Number of sequences (windows) to sample from the buffer in each learning step, defaults to 256.
        gamma : float, optional
            Discount factor for future rewards, defaults to 0.99.
        tau : float, optional
            Interpolation factor for the soft update of the target networks, defaults to 0.005.
        policy_noise : float, optional
            Standard deviation of the Gaussian noise added to the target actor's action for smoothing, defaults to 0.2.
        noise_clip : float, optional
            Maximum (absolute) value for clipping the policy noise, defaults to 0.5.
        policy_delay : int, optional
            Number of critic updates for each actor update, defaults to 2.
        early_stop : int, optional
            Number of evaluations without improvement before stopping training, defaults to 50.
        numeric_eval : bool, optional
            If True, performs a detailed numerical evaluation at the end, defaults to False.
        scaler_params : dict, optional
            Dictionary with the parameters ('scale', 'min') to normalize states and denormalize actions, defaults to None.
        vel_target : float, optional
            Target velocity that the agent should try to reach in the environment, defaults to 70.
        num_workers : int, optional
            Number of parallel 'worker' processes for experience collection, defaults to 2.
        U : int, optional
            Length of the 'unroll' (training) sub-sequence for TBPTT, defaults to 64.
        B : int, optional
            Length of the 'burn-in' (warm-up) sub-sequence for TBPTT, defaults to 32.
        reuse_warmup_buffer : bool, optional
            If True, saves the warm-up buffer to disk and reuses it in future runs, defaults to False.
        buffer_dir : str, optional
            Name of the directory to save the buffer to disk if reuse_warmup_buffer is True, defaults to "buffer".
        """

        # --- Compute Device Configuration ---
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"LEARNER compute device: {self.device}")

        # --- Class parameters (mostly unchanged) ---
        self.f_transicio = f_transicio
        self.length_problem = length_problem
        self.replay_buffer = ReplayBuffer(replay_size)
        self.vel_target = vel_target
        self.numeric_eval = numeric_eval
        self.version = version
        self.results_dir = os.path.join("results", self.version)
        os.makedirs(self.results_dir, exist_ok=True)
        self._state_scale = scaler_params["scale"][:5]
        self._state_min = scaler_params["min"][:5]
        self._action_scale = scaler_params["scale"][2:5]
        self._action_min = scaler_params["min"][2:5]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.early_stop = early_stop
        self.best_eval_reward = -float("inf")
        self.no_improve = 0
        self.total_iterations = 0

        # --- LEARNER Neural Networks (on GPU) ---
        self.Actor = ActorNetwork(scaler_params=scaler_params).to(self.device)
        self.Actor_target = ActorNetwork(scaler_params=scaler_params).to(self.device)
        self.Critic1 = CriticNetwork(scaler_params=scaler_params).to(self.device)
        self.Critic2 = CriticNetwork(scaler_params=scaler_params).to(self.device)
        self.Critic1_target = CriticNetwork(scaler_params=scaler_params).to(self.device)
        self.Critic2_target = CriticNetwork(scaler_params=scaler_params).to(self.device)
        copy_target(self.Actor_target, self.Actor)
        copy_target(self.Critic1_target, self.Critic1)
        copy_target(self.Critic2_target, self.Critic2)

        # --- LEARNER Optimizers ---
        self.optim_actor = optim.Adam(self.Actor.parameters(), lr=0.0001)
        self.optim_critic1 = optim.Adam(self.Critic1.parameters(), lr=0.001)
        self.optim_critic2 = optim.Adam(self.Critic2.parameters(), lr=0.001)

        # --- TBPTT Parameters ---
        self.N = batch_size
        self.B = B
        self.U = U
        self.S = self.B + self.U

        # --- Disk warm-up buffer (optional) ---
        self.reuse_warmup_buffer = reuse_warmup_buffer

        self.buffer_dir = os.path.join(self.results_dir, buffer_dir)
        os.makedirs(self.buffer_dir, exist_ok=True)
        self._disk_files = []
        self._disk_window_count = 0
        self._disk_counter = 0
        self._buffer_meta_path = os.path.join(self.buffer_dir, "meta.json")
        self._init_or_check_buffer_meta()

        # --- MODIFICATION: Creation of Remote Actors (Workers) ---
        self.num_workers = num_workers
        self.workers = [
            RolloutWorker.remote(
                f_transicio=f_transicio,
                worker_id=i,
                scaler_params=scaler_params,
                act_dim=act_dim,
                length_problem=length_problem,
                vel_target=vel_target,
                B=self.B,
                U=self.U,
            )
            for i in range(self.num_workers)
        ]
        print(f"✅ {self.num_workers} remote workers created.")

    # MODIFICATION: The learn method now orchestrates the asynchronous process
    def learn(
        self, total_timesteps, learning_starts=1000, train_freq=4, gradient_steps=4
    ):
        """
        Trains the agent asynchronously using the Actor-Learner architecture.
        """
        # --- Warm-up Phase ---
        print(f"🔥 Starting buffer warm-up phase. Target: {learning_starts} windows.")
        self._update_worker_weights()
        latest_actor_weights = {k: v.cpu() for k, v in self.Actor.state_dict().items()}

        # (re)index buffer on disk if used
        if self.reuse_warmup_buffer:
            self._refresh_disk_index()
            if self._disk_window_count >= learning_starts:
                # load directly to ReplayBuffer and skip collection warm-up
                loaded = self._load_windows_from_disk(learning_starts, shuffle=True)
                print(
                    f"✅ Warm-up loaded from disk: {loaded}/{learning_starts} windows."
                )

        # launch collection tasks (same as before)
        tasks_to_workers = {
            worker.collect_windows.remote(): worker for worker in self.workers
        }
        pending_tasks = list(tasks_to_workers.keys())

        steps_since_last_train = train_freq

        # --------- WARM-UP (hasta llenar el ReplayBuffer con learning_starts) ---------
        while len(self.replay_buffer) < learning_starts:
            ready_tasks_refs, pending_tasks = ray.wait(pending_tasks, num_returns=1)
            ready_ref = ready_tasks_refs[0]
            worker = tasks_to_workers.pop(ready_ref)

            result_windows = ray.get(ready_ref)  # lista de ventanas

            if self.reuse_warmup_buffer:
                # 1) guardar a disco
                n_saved = self._save_windows_to_disk(result_windows)
                # 2) si aún no hemos llenado el ReplayBuffer y ya hay suficientes en disco, cargar lo necesario
                if (
                    len(self.replay_buffer) < learning_starts
                    and self._disk_window_count >= learning_starts
                ):
                    needed = learning_starts - len(self.replay_buffer)
                    loaded = self._load_windows_from_disk(needed, shuffle=True)
            else:
                # comportamiento original: meter directamente al ReplayBuffer
                for w in result_windows:
                    self.replay_buffer.put(w)

            # actualizar pesos del worker y relanzar
            ray.get(worker.set_weights.remote(latest_actor_weights))
            new_task_ref = worker.collect_windows.remote()
            tasks_to_workers[new_task_ref] = worker
            pending_tasks.append(new_task_ref)

            # logging
            if self.reuse_warmup_buffer:
                print(
                    f"\r[WARM-UP disco] guardadas: {self._disk_window_count} | en RAM: {len(self.replay_buffer)}/{learning_starts}",
                    end="",
                )
            else:
                print(f"\rBuffer: {len(self.replay_buffer)}/{learning_starts}", end="")

        print("\n✅ Fase de calentamiento completada.")

        # --- Fase de Entrenamiento Principal ---
        print("\n🚀 Iniciando fase de entrenamiento principal asíncrona...")

        timesteps_collected = 0
        updates_done_since_eval = 0
        all_rewards_eval = []

        while timesteps_collected < total_timesteps:
            # El print de estado ahora incluye el contador de pasos para el entrenamiento
            #             print(f"\rIteraciones: {self.total_iterations} | Timesteps: ~{timesteps_collected}/{total_timesteps} | Buffer: {len(self.replay_buffer)} | Pasos para entrenar: {steps_since_last_train}/{train_freq}", end="")

            # --- PARTE 1: APRENDIZAJE (con train_freq y gradient_steps) ---
            if (
                len(self.replay_buffer) > self.N
                and steps_since_last_train >= train_freq
            ):
                print(f"\n--- Ráfaga de Entrenamiento ({gradient_steps} pasos) ---")
                print(
                    f"Iteraciones: {self.total_iterations} | Timesteps: ~{timesteps_collected}/{total_timesteps} | Buffer: {len(self.replay_buffer)}"
                )

                for _ in range(gradient_steps):
                    batch = self.replay_buffer.get(self.N)

                    # --- críticos ---
                    critic1_loss, critic2_loss = self.compute_critic_loss(batch)
                    self.optim_critic1.zero_grad(set_to_none=True)
                    critic1_loss.backward()
                    #                     torch.nn.utils.clip_grad_norm_(self.Critic1.parameters(), max_norm=1.0)
                    self.optim_critic1.step()

                    self.optim_critic2.zero_grad(set_to_none=True)
                    critic2_loss.backward()
                    #                     torch.nn.utils.clip_grad_norm_(self.Critic2.parameters(), max_norm=1.0)
                    self.optim_critic2.step()

                    self.total_iterations += 1
                    updates_done_since_eval += 1

                    if self.total_iterations % self.policy_delay == 0:
                        # Congela el crítico durante la loss del actor
                        actor_loss = self.compute_actor_loss(batch)
                        self.optim_actor.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        #                         torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), max_norm=1.0)
                        self.optim_actor.step()

                        soft_update(self.Actor_target, self.Actor, self.tau)
                        soft_update(self.Critic1_target, self.Critic1, self.tau)
                        soft_update(self.Critic2_target, self.Critic2, self.tau)

                        latest_actor_weights = {
                            k: v.cpu() for k, v in self.Actor.state_dict().items()
                        }

                steps_since_last_train = 0

            # --- PARTE 2: RECOLECCIÓN (no bloqueante y balanceada) ---
            ready_tasks_refs, pending_tasks = ray.wait(
                pending_tasks, num_returns=1, timeout=0.0
            )

            if ready_tasks_refs:
                ready_ref = ready_tasks_refs[0]

                worker_windows = ray.get(ready_ref)
                num_new_windows = len(worker_windows)
                for window in worker_windows:
                    self.replay_buffer.put(window)

                new_steps = num_new_windows * self.U
                timesteps_collected += new_steps
                # Increment the new counter ---
                steps_since_last_train += new_steps

                worker = tasks_to_workers.pop(ready_ref)

                # 1. Update worker weights and wait for it to finish.
                ray.get(worker.set_weights.remote(latest_actor_weights))

                # 2. Launch the new collection task.
                new_task_ref = worker.collect_windows.remote()

                tasks_to_workers[new_task_ref] = worker
                pending_tasks.append(new_task_ref)

            # --- PART 3: PERIODIC EVALUATION ---
            if (
                updates_done_since_eval >= gradient_steps
            ):  # Evaluate every 500 critic updates
                current_eval = self.eval_episodes(n=self.num_workers)
                all_rewards_eval.append(current_eval)
                print(
                    f"\nTimesteps: ~{timesteps_collected}/{total_timesteps}, Eval Reward: {current_eval:.2f}"
                )
                updates_done_since_eval = 0  # Reset counter

                # Early Stopping Logic
                if current_eval > self.best_eval_reward:
                    self.best_eval_reward = current_eval
                    self.no_improve = 0
                    self.save_models("actor_best.pth")
                else:
                    self.no_improve += 1

                if self.no_improve >= self.early_stop:
                    print(
                        f"\n[EARLY STOP] No improvement in {self.early_stop} evaluations."
                    )
                    break

        print("\n[Training Finished] Saving final model...")
        self.save_models(file_name="actor_final.pth")

        self.plot_trajectory()

        if self.numeric_eval:
            self.evaluate_numeric(n=30, vel_target=self.vel_target)

    def _update_worker_weights(self):
        """Sends the Learner's actor weights to all workers."""
        actor_weights = self.Actor.to("cpu").state_dict()
        for worker in self.workers:
            worker.set_weights.remote(actor_weights)
        self.Actor.to(self.device)

    def compute_critic_loss(self, batch):
        """
        Calculation of the MSE loss for the two critics with TBPTT correctly implemented.
        """
        # 1. Sequence Unpacking and Formatting (unchanged)
        state_b, action_b, reward_b, next_state_b, term_b, trunc_b = batch
        states_seq = torch.tensor(state_b, dtype=torch.float32, device=self.device)
        actions_seq = torch.tensor(action_b, dtype=torch.float32, device=self.device)
        rewards_seq = torch.tensor(
            reward_b, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        next_states_seq = torch.tensor(
            next_state_b, dtype=torch.float32, device=self.device
        )
        done_seq = torch.tensor(
            term_b | trunc_b, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        # 2. Vectorized Target Q Calculation (unchanged)
        with torch.no_grad():
            # Get the target actor action for the next state
            actions_target_seq, _ = self.Actor_target(next_states_seq)
            noise = torch.normal(
                0.0,
                self.policy_noise,
                size=actions_target_seq.shape,
                device=self.device,
            ).clamp(-self.noise_clip, self.noise_clip)
            next_actions_seq = torch.clamp(actions_target_seq + noise, -1.0, 1.0)

            # Get the target Q value from target critics
            q1_t_seq, _ = self.Critic1_target(next_states_seq, next_actions_seq)
            q2_t_seq, _ = self.Critic2_target(next_states_seq, next_actions_seq)

            q_target_seq = torch.min(q1_t_seq, q2_t_seq)
            y_i_seq = rewards_seq + (1 - done_seq) * self.gamma * q_target_seq

        # Select the target part corresponding to the 'unroll'
        y_i_unroll = y_i_seq[:, self.B :, :].detach()

        # --- INICIA LA LÓGICA DE TBPTT ---

        # 3. Burn-in: Obtener el estado oculto inicial sin registrar gradientes
        with torch.no_grad():
            states_burn_in = states_seq[:, : self.B, :]
            actions_burn_in = actions_seq[:, : self.B, :]

            # Pasamos la secuencia de "calentamiento" y guardamos solo el estado oculto final
            _, h_critic1 = self.Critic1(states_burn_in, actions_burn_in)
            _, h_critic2 = self.Critic2(states_burn_in, actions_burn_in)

        # 4. Unroll: Calcular Q actual usando el estado oculto del burn-in
        states_unroll = states_seq[:, self.B :, :]
        actions_unroll = actions_seq[:, self.B :, :]

        # Pasamos la secuencia de "entrenamiento" y el estado oculto inicial.
        # El gradiente fluirá a través de esta operación.
        q1_unroll, _ = self.Critic1(
            states_unroll, actions_unroll, hidden_state=h_critic1
        )
        q2_unroll, _ = self.Critic2(
            states_unroll, actions_unroll, hidden_state=h_critic2
        )

        # 5. Calcular la pérdida MSE sobre la ventana de 'unroll'
        loss1 = F.mse_loss(q1_unroll, y_i_unroll)
        loss2 = F.mse_loss(q2_unroll, y_i_unroll)

        return loss1, loss2

    def compute_actor_loss(self, batch):
        """
        Calculation of the actor's loss with TBPTT correctly implemented.
        """
        # 1. Desempaquetar y formatear estados Y ACCIONES
        # ANTES: state_batch, _, _, _, _, _ = batch
        # CORREGIDO:
        state_batch, action_batch, _, _, _, _ = batch
        states_seq = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        actions_seq = torch.tensor(action_batch, dtype=torch.float32).to(
            self.device
        )  # <-- LÍNEA AÑADIDA

        # --- INICIA LA LÓGICA DE TBPTT ---

        # 2. Burn-in: Obtener el estado oculto inicial para el Actor
        with torch.no_grad():
            states_burn_in = states_seq[:, : self.B, :]
            _, h_actor_burn_in = self.Actor(states_burn_in)

        # 3. Unroll: Calculate actor actions for the training window
        states_unroll = states_seq[:, self.B :, :]
        actions_pred_unroll, _ = self.Actor(states_unroll, hidden_state=h_actor_burn_in)

        # 4. Evaluate these actions with the Critic. To do this, we first calculate
        #    the critic's hidden state after its own burn-in.
        with torch.no_grad():
            # To get the correct critic hidden state, we must pass it
            # both the states and the REAL ACTIONS from the buffer during burn-in.
            actions_burn_in = actions_seq[:, : self.B, :]
            _, h_critic1_burn_in = self.Critic1(states_burn_in, actions_burn_in)

        # Now we evaluate the actor's actions in the unroll, starting from
        # correct critic hidden state.
        for p in self.Critic1.parameters():
            p.requires_grad_(False)
        q_values_unroll, _ = self.Critic1(
            states_unroll, actions_pred_unroll, hidden_state=h_critic1_burn_in
        )
        actor_loss = -q_values_unroll.mean()
        for p in self.Critic1.parameters():
            p.requires_grad_(True)
        return actor_loss

    def choose_action(self, s_raw):
        # This method is now used by the Learner for evaluation, not by the workers
        current_device = next(self.Actor.parameters()).device
        state = torch.tensor(s_raw, dtype=torch.float32).unsqueeze(0).to(current_device)
        self.Actor.eval()
        with torch.no_grad():
            actions, _ = self.Actor(state)
            a = actions.cpu().squeeze(0).numpy()

        self.Actor.train()
        return a

    def descale_action(self, action_scaled):
        action_scaled_np = np.asarray(action_scaled, dtype=np.float32)
        return (action_scaled_np - self._action_min) / self._action_scale

    def eval_episodes(self, n: int = 3):
        # Para simplificar, la evaluación se hace en el proceso principal (learner)
        # o se podría paralelizar también con los workers.
        # Aquí una versión simple secuencial en el learner.
        print(f"\nEjecutando {n} evaluaciones...")
        returns = []
        env_eval = Environment(self.f_transicio, length_problem=self.length_problem)
        for _ in range(n):
            obs = env_eval.reset()
            self.Actor.reset_states()
            episode_return = 0.0
            done = False
            while not done:
                a_scaled = self.choose_action(obs)  # Usa el choose_action del learner
                a_phys = self.descale_action(a_scaled)
                obs, reward, terminated, truncated = env_eval.step(
                    a_phys, self.vel_target
                )
                episode_return += float(reward)
                done = terminated or truncated

            returns.append(episode_return)
        return float(np.mean(returns))

    def save_models(self, file_name="actor.pth"):
        """
        Saves the Actor's state_dict for later inference.
        The file is saved in the results directory of the current version.
        """
        # 1. Define full file path using results directory
        file_path = os.path.join(self.results_dir, file_name)

        # 2. Save Actor state dictionary (parameters).
        #    We only need the Actor for inference, not the Critic.
        #    Using .state_dict() is the recommended PyTorch way.
        torch.save(self.Actor.state_dict(), file_path)

        # 3. Print confirmation to know it has been saved
        print(f"\n[Model Saved] Actor saved in: {file_path}")

    def plot_trajectory(
        self, vel_target: float = 70.0, K: int | None = None, title: str | None = None
    ):
        """
        Simulates K steps and generates an interactive plot.
        Automatically saves the result in:
            results/<version>/trag_<version>.html
        """
        env = Environment(self.f_transicio, length_problem=self.length_problem)
        K = K or env.length_problem
        title = title or f"TD3 – {self.version}"

        state = env.reset(vel_target=vel_target)
        self.Actor.reset_states()

        log = {
            k: []
            for k in ("step", "mf", "brk", "ice_sp", "torque", "vel_out", "nox", "co")
        }

        for step in range(K):
            a_scaled = self.choose_action(state)
            a_phys = self.descale_action(a_scaled)
            state, _, term, trunc = env.step(a_phys, self.vel_target)

            def as_float(x):
                return float(x) if torch.is_tensor(x) else x

            log["step"].append(step)
            log["mf"].append(as_float(env.mf))
            log["brk"].append(as_float(env.brk))
            log["ice_sp"].append(as_float(env.ice_sp))
            log["torque"].append(as_float(env.torque))
            log["vel_out"].append(as_float(env.vel_out))
            log["nox"].append(as_float(env.nox))
            log["co"].append(as_float(env.co))

            if term or trunc:
                break

        df = pd.DataFrame(log)

        # ------------- plotly express -------------------------------
        fig = px.line(
            df,
            x="step",
            y=["mf", "brk", "ice_sp", "torque", "vel_out", "nox", "co"],
            title=title,
        )

        for tr in fig.data:
            if tr.name in ("vel_out", "nox", "co"):
                tr.visible = "legendonly"

        buttons = [
            dict(
                label="Estados",
                method="update",
                args=[
                    {
                        "visible": [
                            t.name in ("mf", "brk", "ice_sp", "torque")
                            for t in fig.data
                        ]
                    },
                    {"title": f"{title} — Estados"},
                ],
            ),
            dict(
                label="Salidas",
                method="update",
                args=[
                    {"visible": [t.name in ("vel_out", "nox", "co") for t in fig.data]},
                    {"title": f"{title} — Salidas"},
                ],
            ),
        ]
        fig.update_layout(
            updatemenus=[dict(type="buttons", direction="right", buttons=buttons)]
        )

        # ------------- guardar automáticamente ----------------------
        file_path = os.path.join(self.results_dir, f"trag_{self.version}.html")
        fig.write_html(file_path)
        print(f"[Plot saved] {file_path}")
        return

    def evaluate_numeric(
        self, n: int = 30, vel_target: float = 70.0, file_name: str | None = None
    ):
        """
        Evaluates the current policy on 'n' episodes and saves two metrics
        per episode:
            • early_return  — until terminated|truncated
            • full_return   — until length_problem
        The result is saved in results/<version>/numeric_eval_<version>.txt
        """
        env = Environment(self.f_transicio, length_problem=self.length_problem)

        K = env.length_problem
        early_returns, full_returns, early_steps = [], [], []

        for ep in range(1, n + 1):
            state = env.reset(vel_target=vel_target)
            self.Actor.reset_states()
            eret, fret, cut = 0.0, 0.0, None

            for t in range(K):
                a_scaled = self.choose_action(state)
                a_phys = self.descale_action(a_scaled)
                state, reward, term, trunc = env.step(a_phys, vel_target)
                fret += float(reward)
                if cut is None:
                    eret += float(reward)
                    if term or trunc:
                        cut = t + 1
                if env.step_count >= K:
                    break

            early_returns.append(eret)
            full_returns.append(fret)
            early_steps.append(cut if cut is not None else K)

        # ----------------- estadísticas -----------------------------
        avg_early = np.mean(early_returns)
        avg_full = np.mean(full_returns)
        avg_step = np.mean(early_steps)

        # ----------------- save to txt ---------------------------
        if file_name is None:
            file_name = f"numeric_eval_{self.version}.txt"

        path = os.path.join(self.results_dir, file_name)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Numeric evaluation — TD3 version: {self.version}\n")
            f.write(f"Episodes: {n}\n\n")
            f.write("ep\tstep_cut\tearly_ret\tfull_ret\n")
            for i, (sc, er, fr) in enumerate(
                zip(early_steps, early_returns, full_returns), 1
            ):
                f.write(f"{i}\t{sc}\t{er:.4f}\t{fr:.4f}\n")
            f.write("\nAverages\n")
            f.write(f"mean_step_cut: {avg_step:.2f}\n")
            f.write(f"mean_early_return: {avg_early:.4f}\n")
            f.write(f"mean_full_return:  {avg_full:.4f}\n")

        print(f"[Numeric eval saved] {path}")
        return

    # ----------------- Buffer disco: meta -----------------
    def _init_or_check_buffer_meta(self):
        """
        Initializes or verifies the metadata file of the on-disk buffer.

        This method manages a `meta.json` file in the buffer directory.
        If the file does not exist, it creates it with the current agent's
        configuration (dimensions, sequence length, etc.). If it already exists, it reads it and
        compares its parameters with the current ones. If it detects a significant
        discrepancy, it prints a warning, as it could cause
        incompatibilities when loading previously saved experiences.
        """
        meta = dict(
            obs_dim=self.obs_dim, act_dim=self.act_dim, S=self.S, B=self.B, U=self.U
        )
        if not os.path.exists(self._buffer_meta_path):
            with open(self._buffer_meta_path, "w") as f:
                json.dump(meta, f)
        else:
            try:
                with open(self._buffer_meta_path, "r") as f:
                    old = json.load(f)
                # soft check: if something important changes, warn
                ok = (
                    old.get("obs_dim") == self.obs_dim
                    and old.get("act_dim") == self.act_dim
                    and old.get("S") == self.S
                )
                if not ok:
                    print(
                        "[WARN] The buffer on disk was created with a different configuration "
                        f"(meta={old}). Continuing, but you might have incompatibilities."
                    )
            except Exception as e:
                print(f"[WARN] Could not read buffer meta: {e}")

    # ----------------- Disk buffer: index/counter -----------------
    def _refresh_disk_index(self):
        """
        Updates the internal file index and the window counter of the on-disk buffer.

        It scans the buffer directory for data files (`part_*.npz`),
        sorts them, and updates the `self._disk_files` list. Then, it iterates over
        these files to recalculate the total number of stored experience
        windows (`self._disk_window_count`), summing the windows contained
        in each file. It also updates the counter for the next file
        to be saved.
        """
        self._disk_files = sorted(
            glob.glob(os.path.join(self.buffer_dir, "part_*.npz"))
        )
        self._disk_counter = len(self._disk_files)
        self._disk_window_count = 0
        for fp in self._disk_files:
            try:
                with np.load(fp) as data:
                    self._disk_window_count += int(data["s"].shape[0])
            except Exception as e:
                print(f"[WARN] Corrupt or unreadable buffer file: {fp} ({e})")

    def _next_buffer_filepath(self):
        """
        Generates the path for the next buffer file and updates the counter.

        Returns
        -------
        str
            A formatted file path with a sequential numerical counter
            (e.g., '.../buffer/part_000001.npz').
        """
        path = os.path.join(self.buffer_dir, f"part_{self._disk_counter:06d}.npz")
        self._disk_counter += 1
        return path

    # ----------------- Save windows to disk -----------------
    def _save_windows_to_disk(self, windows):
        """
        Saves a batch of experience windows to a new `.npz` file on disk.

        It receives a list of tuples (windows), where each tuple contains sequences
        of states, actions, rewards, etc. It stacks these sequences into NumPy
        arrays and saves them in a compressed format in a single file, using
        the `_next_buffer_filepath` method to get the file name.

        Parameters
        ----------
        windows : list[tuple]
            List of experience windows to save.

        Returns
        -------
        int
            The number of windows that were successfully saved.
        """
        if not windows:
            return 0
        # stack by field (all windows have the same S)
        s = np.stack([w[0] for w in windows], axis=0).astype(np.float32)
        a = np.stack([w[1] for w in windows], axis=0).astype(np.float32)
        r = np.stack([w[2] for w in windows], axis=0).astype(np.float32)
        ns = np.stack([w[3] for w in windows], axis=0).astype(np.float32)
        t = np.stack([w[4] for w in windows], axis=0).astype(np.bool_)
        tr = np.stack([w[5] for w in windows], axis=0).astype(np.bool_)
        fp = self._next_buffer_filepath()
        np.savez_compressed(fp, s=s, a=a, r=r, ns=ns, t=t, tr=tr)
        n = s.shape[0]
        self._disk_window_count += n
        return n

    # ----------------- Cargar N ventanas desde disco al ReplayBuffer -----------------
    def _load_windows_from_disk(self, n, shuffle=True, rng_seed=123):
        """
        Loads a specific number of windows from the files on disk to the in-memory buffer.

        First, it updates the file index. Then, it iterates over the files
        (in random order if `shuffle` is True) and loads individual windows
        until the requested amount `n` is reached. Each loaded window is added
        to the in-memory `ReplayBuffer`.

        Parameters
        ----------
        n : int
            The number of experience windows to be loaded.
        shuffle : bool, optional
            If True, loads the windows from files and from within the files
            in random order, defaults to True.
        rng_seed : int, optional
            Seed for the random number generator to ensure the
            reproducibility of the shuffling, defaults to 123.

        Returns
        -------
        int
            The number of windows that were successfully loaded into the memory buffer.
        """
        self._refresh_disk_index()
        if n <= 0 or self._disk_window_count <= 0:
            return 0

        files = list(self._disk_files)
        rng = np.random.default_rng(rng_seed)
        if shuffle:
            rng.shuffle(files)

        loaded = 0
        for fp in files:
            with np.load(fp) as data:
                m = int(data["s"].shape[0])
                idx = np.arange(m)
                if shuffle:
                    rng.shuffle(idx)
                take = min(n - loaded, m)
                sel = idx[:take]
                for i in sel:
                    window = (
                        data["s"][i],
                        data["a"][i],
                        data["r"][i],
                        data["ns"][i],
                        data["t"][i],
                        data["tr"][i],
                    )
                    self.replay_buffer.put(window)
                loaded += take
                if loaded >= n:
                    break
        return loaded
