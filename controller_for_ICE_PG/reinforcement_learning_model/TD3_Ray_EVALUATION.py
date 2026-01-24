import os

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
import time
import glob
import json
import warnings

warnings.filterwarnings("ignore")

from environment_EVALUATION import Environment
import ray

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


@ray.remote
class RolloutWorker:
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
        profile_training=False,
    ):
        self._silence_tf_in_worker()
        self.worker_id = worker_id
        self.env = Environment(f_transicio, length_problem=length_problem)
        self.exploration_noise = NormalActionNoise(mean=np.zeros(act_dim), sigma=0.1)
        self.vel_target = vel_target
        self.profile_training = profile_training
        self.B, self.U = B, U
        self.S = B + U
        self._action_scale = scaler_params["scale"][2:5]
        self._action_min = scaler_params["min"][2:5]
        self.actor_worker = ActorNetwork(scaler_params=scaler_params).to("cpu")
        print(f"[Worker {self.worker_id}] Created and ready.")

    def get_weights(self):
        return self.actor_worker.state_dict()

    def set_weights(self, weights):
        start_time = time.time()
        self.actor_worker.load_state_dict(weights)
        self.actor_worker.reset_states()
        duration = time.time() - start_time
        return duration, self.worker_id

    def choose_action(self, s_raw):
        state = torch.tensor(s_raw, dtype=torch.float32).unsqueeze(0).to("cpu")
        self.actor_worker.eval()
        with torch.no_grad():
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

    # ▼▼▼ START OF ROLLOUTWORKER MODIFICATION ▼▼▼
    def collect_windows(self):
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

        # Lists to save detailed step times
        detailed_ice_timings = []
        detailed_pg_timings = []

        while True:
            action_scaled = self.choose_action(obs)
            action_scaled_noisy = np.clip(
                action_scaled + self.exploration_noise.sample(), -1, 1
            )
            action_phys = self.descale_action(action_scaled_noisy)

            # Now we capture the dictionaries with detailed timings
            next_obs, reward, terminated, truncated, ice_timings, pg_timings = (
                self.env.step(
                    action_phys, self.vel_target, profile_step=self.profile_training
                )
            )

            if self.profile_training:
                if ice_timings:
                    detailed_ice_timings.append(ice_timings)
                if pg_timings:
                    detailed_pg_timings.append(pg_timings)

            done = terminated or truncated
            for k, v in zip(
                episode_data.keys(),
                [obs, action_scaled_noisy, reward, next_obs, terminated, truncated],
            ):
                episode_data[k].append(v)

            obs = next_obs
            if done:
                break

        windows = []
        episode_len = len(episode_data["states"])
        for i in range(0, max(0, episode_len - self.S + 1), self.U):
            # ... (logic for creating windows does not change) ...
            s, a, r, ns, t, tr = [
                np.stack(episode_data[k][i : i + self.S]) for k in episode_data.keys()
            ]
            windows.append((s, a, r, ns, t, tr))

        # Returning lists with detailed timings
        return windows, detailed_ice_timings, detailed_pg_timings

    # ▲▲▲ END OF ROLLOUTWORKER MODIFICATION ▲▲▲


class TD3:
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
        profile_training=False,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"LEARNER compute device: {self.device}")
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
        self.Actor = ActorNetwork(scaler_params=scaler_params).to(self.device)
        self.Actor_target = ActorNetwork(scaler_params=scaler_params).to(self.device)
        self.Critic1 = CriticNetwork(scaler_params=scaler_params).to(self.device)
        self.Critic2 = CriticNetwork(scaler_params=scaler_params).to(self.device)
        self.Critic1_target = CriticNetwork(scaler_params=scaler_params).to(self.device)
        self.Critic2_target = CriticNetwork(scaler_params=scaler_params).to(self.device)
        copy_target(self.Actor_target, self.Actor)
        copy_target(self.Critic1_target, self.Critic1)
        copy_target(self.Critic2_target, self.Critic2)
        self.optim_actor = optim.Adam(self.Actor.parameters(), lr=0.0001)
        self.optim_critic1 = optim.Adam(self.Critic1.parameters(), lr=0.001)
        self.optim_critic2 = optim.Adam(self.Critic2.parameters(), lr=0.001)
        self.N = batch_size
        self.B = B
        self.U = U
        self.S = self.B + self.U
        self.reuse_warmup_buffer = reuse_warmup_buffer
        self.buffer_dir = os.path.join(self.results_dir, buffer_dir)
        os.makedirs(self.buffer_dir, exist_ok=True)
        self._disk_files = []
        self._disk_window_count = 0
        self._disk_counter = 0
        self._buffer_meta_path = os.path.join(self.buffer_dir, "meta.json")
        self._init_or_check_buffer_meta()

        self.profile_training = profile_training

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
                profile_training=self.profile_training,
            )
            for i in range(self.num_workers)
        ]
        print(f"✅ {self.num_workers} remote workers created.")

        if self.profile_training:
            self.profiling_data = []
            self.wait_for_data_start_time = None
            self.is_first_learn_step = True
            self.detailed_profiling_data = []

    def learn(
        self, total_timesteps, learning_starts=1000, train_freq=4, gradient_steps=4
    ):
        print(f"🔥 Starting buffer warm-up phase. Target: {learning_starts} windows.")
        self._update_worker_weights()
        latest_actor_weights = {k: v.cpu() for k, v in self.Actor.state_dict().items()}

        if self.reuse_warmup_buffer and os.path.exists(self._buffer_meta_path):
            self._refresh_disk_index()
            if self._disk_window_count >= learning_starts:
                loaded = self._load_windows_from_disk(learning_starts, shuffle=True)
                print(
                    f"✅ Warm-up loaded from disk: {loaded}/{learning_starts} windows."
                )

        tasks_to_workers = {
            worker.collect_windows.remote(): worker for worker in self.workers
        }
        pending_tasks = list(tasks_to_workers.keys())

        # --- WARM-UP LOOP ---
        while len(self.replay_buffer) < learning_starts:
            ready_tasks_refs, pending_tasks = ray.wait(pending_tasks, num_returns=1)
            ready_ref = ready_tasks_refs[0]
            worker = tasks_to_workers.pop(ready_ref)

            worker_windows, detailed_ice_t, detailed_pg_t = ray.get(ready_ref)

            if self.profile_training:
                for timings_dict in detailed_ice_t:
                    timings_dict["model"] = "ICE_Warmup"
                    self.detailed_profiling_data.append(timings_dict)
                for timings_dict in detailed_pg_t:
                    timings_dict["model"] = "PG_Warmup"
                    self.detailed_profiling_data.append(timings_dict)

            if self.reuse_warmup_buffer:
                self._save_windows_to_disk(worker_windows)

            for w in worker_windows:
                self.replay_buffer.put(w)

            update_duration, worker_id = ray.get(
                worker.set_weights.remote(latest_actor_weights)
            )

            new_task_ref = worker.collect_windows.remote()
            tasks_to_workers[new_task_ref] = worker
            pending_tasks.append(new_task_ref)

            print(f"\rBuffer: {len(self.replay_buffer)}/{learning_starts}", end="")
        print("\n✅ Warm-up phase completed.")

        # --- MAIN TRAINING LOOP ---
        print("\n🚀 Starting main asynchronous training phase...")

        # ▼▼▼ KEY CORRECTION: Always initialize to 0 before the main loop ▼▼▼
        timesteps_collected = 0
        updates_done_since_eval = 0

        while timesteps_collected < total_timesteps:
            # Training logic
            if len(self.replay_buffer) > self.N:
                for _ in range(gradient_steps):
                    batch = self.replay_buffer.get(self.N)
                    critic1_loss, critic2_loss = self.compute_critic_loss(batch)
                    self.optim_critic1.zero_grad(set_to_none=True)
                    critic1_loss.backward()
                    self.optim_critic1.step()
                    self.optim_critic2.zero_grad(set_to_none=True)
                    critic2_loss.backward()
                    self.optim_critic2.step()
                    self.total_iterations += 1
                    if self.total_iterations % self.policy_delay == 0:
                        actor_loss = self.compute_actor_loss(batch)
                        self.optim_actor.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        self.optim_actor.step()
                        soft_update(self.Actor_target, self.Actor, self.tau)
                        soft_update(self.Critic1_target, self.Critic1, self.tau)
                        soft_update(self.Critic2_target, self.Critic2, self.tau)
                        latest_actor_weights = {
                            k: v.cpu() for k, v in self.Actor.state_dict().items()
                        }
                updates_done_since_eval += gradient_steps

            # Data collection logic
            ready_tasks_refs, pending_tasks = ray.wait(pending_tasks, num_returns=1)
            ready_ref = ready_tasks_refs[0]
            worker_windows, detailed_ice_t, detailed_pg_t = ray.get(ready_ref)

            if self.profile_training:
                for timings_dict in detailed_ice_t:
                    timings_dict["model"] = "ICE_Train"
                    self.detailed_profiling_data.append(timings_dict)
                for timings_dict in detailed_pg_t:
                    timings_dict["model"] = "PG_Train"
                    self.detailed_profiling_data.append(timings_dict)

            for window in worker_windows:
                self.replay_buffer.put(window)
            timesteps_collected += len(worker_windows) * self.U

            worker = tasks_to_workers.pop(ready_ref)
            worker.set_weights.remote(latest_actor_weights)
            new_task_ref = worker.collect_windows.remote()
            tasks_to_workers[new_task_ref] = worker
            pending_tasks.append(new_task_ref)

            # Evaluation logic
            if updates_done_since_eval >= 5000:  # Or preferred frequency
                current_eval, _, _, _, _ = self.eval_episodes(n=self.num_workers)
                print(
                    f"\nTimesteps: ~{timesteps_collected}/{total_timesteps}, Eval Reward: {current_eval:.2f}"
                )
                updates_done_since_eval = 0
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

        # --- Finalization ---
        print("\n[Training Finished] Saving final model...")
        self.save_models(file_name="actor_final.pth")
        if self.profile_training:
            self._analyze_detailed_timings()
        self.plot_trajectory()
        if self.numeric_eval:
            self.evaluate_numeric(n=30, vel_target=self.vel_target)

    def _analyze_profiling_data(self):
        """
        Processes collected timing data, prints statistics, and generates four plots
        with the requested visualization improvements.
        """
        if not self.profiling_data:
            print("\n[PROFILING] No data collected to analyze.")
            return

        print("\n--- BOTTLENECK ANALYSIS ---")
        df = pd.DataFrame(self.profiling_data).fillna(0)
        csv_path = os.path.join(self.results_dir, "profiling_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"[PROFILING] Timing data saved in: {csv_path}")

        print("\n[PROFILING] Time summary by event (in seconds):")
        summary = df.groupby("event")["duration"].describe()
        print(summary)

        # --- Boxplot with Reordered X Axis ---
        category_order = [
            "learn_step",
            "evaluation",
            "learner_wait",
            "policy_update",
            "data_collection",
        ]
        fig_box = px.box(
            df,
            x="event",
            y="duration",
            color="event",
            points="all",
            title="Time Distribution by Training Phase",
            labels={"event": "Training Phase", "duration": "Duration (seconds)"},
            category_orders={"event": category_order},
        )
        fig_box.update_layout(showlegend=False)
        plot_path_box = os.path.join(
            self.results_dir, "profiling_bottleneck_analysis.html"
        )
        fig_box.write_html(plot_path_box)
        print(f"\n[PROFILING] Bottleneck analysis plot saved in: {plot_path_box}")

        # --- Learner Activity Analysis and Pie Chart ---
        print("\n--- LEARNER ACTIVITY ANALYSIS ---")
        df_learner = df[df["worker_id"] == "learner"].copy()
        if not df_learner.empty:
            time_summary = df_learner.groupby("event")["duration"].sum()
            total_learner_time = time_summary.sum()
            if total_learner_time > 0:
                time_percentage = (time_summary / total_learner_time) * 100
                print("\n[PROFILING] Learner time % by activity:")
                for event, percentage in time_percentage.items():
                    print(f"- {event}: {percentage:.2f}%")
                pie_data = time_summary.reset_index()
                pie_data.columns = ["event", "total_duration"]
                fig_pie = px.pie(
                    pie_data,
                    values="total_duration",
                    names="event",
                    title="Learner Activity Time Distribution",
                    labels={"event": "Activity", "total_duration": "Total Time (s)"},
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                plot_path_pie = os.path.join(
                    self.results_dir, "profiling_learner_activity.html"
                )
                fig_pie.write_html(plot_path_pie)
                print(f"\n[PROFILING] Learner activity plot saved in: {plot_path_pie}")

        # --- Detailed Data Collection Analysis ---
        print("\n--- DETAILED DATA COLLECTION ANALYSIS ---")
        df_coll = df[df["event"] == "data_collection"].copy()
        if not df_coll.empty:
            df_coll["other_duration"] = (
                df_coll["duration"]
                - df_coll["env_step_duration"]
                - df_coll["action_logic_duration"]
            )
            worker_analysis = (
                df_coll.groupby("worker_id")[
                    [
                        "duration",
                        "env_step_duration",
                        "action_logic_duration",
                        "other_duration",
                    ]
                ]
                .mean()
                .reset_index()
            )
            print("\n[PROFILING] Avg Data Collection component duration by Worker (s):")
            print(worker_analysis)
            plot_data = pd.melt(
                worker_analysis,
                id_vars=["worker_id"],
                value_vars=[
                    "env_step_duration",
                    "action_logic_duration",
                    "other_duration",
                ],
                var_name="component",
                value_name="average_duration",
            )
            fig_bar = px.bar(
                plot_data,
                x="worker_id",
                y="average_duration",
                color="component",
                title="Avg Data Collection Time Breakdown by Worker",
                labels={
                    "worker_id": "Worker ID",
                    "average_duration": "Avg Duration (s)",
                    "component": "Task Component",
                },
                category_orders={
                    "component": [
                        "env_step_duration",
                        "action_logic_duration",
                        "other_duration",
                    ]
                },
            )
            new_names = {
                "env_step_duration": "Simulation (env.step)",
                "action_logic_duration": "Action Logic",
                "other_duration": "Management (Other)",
            }
            fig_bar.for_each_trace(lambda t: t.update(name=new_names[t.name]))
            plot_path_bar = os.path.join(
                self.results_dir, "profiling_datacollector_breakdown.html"
            )
            fig_bar.write_html(plot_path_bar)
            print(
                f"\n[PROFILING] Data Collector breakdown plot saved in: {plot_path_bar}"
            )

        # --- Detailed Evaluation Analysis (Average Episode) ---
        print("\n--- DETAILED EVALUATION ANALYSIS ---")
        df_eval = df[df["event"] == "evaluation"].copy()
        if not df_eval.empty and "action_logic_duration" in df_eval.columns:
            avg_total_eval_duration = df_eval["duration"].mean()
            avg_action_logic_per_episode = df_eval["action_logic_duration"].mean()
            avg_env_step_per_episode = df_eval["env_step_duration"].mean()
            avg_episode_duration = avg_total_eval_duration / self.num_workers
            avg_other_duration = (
                avg_episode_duration
                - avg_action_logic_per_episode
                - avg_env_step_per_episode
            )
            print("\n[PROFILING] Avg evaluation episode time breakdown (s):")
            print(
                f"- Action Logic: {avg_action_logic_per_episode:.4f}\n- Simulation (env.step): {avg_env_step_per_episode:.4f}\n- Management (Other): {max(0, avg_other_duration):.4f}\n--------------------------------------\n- Total episode duration: {avg_episode_duration:.4f}"
            )

            bar_data_eval = pd.DataFrame(
                {
                    "component": [
                        "Simulation (env.step)",
                        "Action Logic",
                        "Management (Other)",
                    ],
                    "duration": [
                        avg_env_step_per_episode,
                        avg_action_logic_per_episode,
                        max(0, avg_other_duration),
                    ],
                    "episode": "Avg Evaluation Episode",
                }
            )
            fig_bar_eval = px.bar(
                bar_data_eval,
                x="episode",
                y="duration",
                color="component",
                title="Time Breakdown in Avg Evaluation Episode",
                labels={
                    "episode": "",
                    "duration": "Avg Duration (s)",
                    "component": "Component",
                },
            )
            fig_bar_eval.for_each_trace(
                lambda t: t.update(
                    name=new_names.get(t.name.replace("duration", ""), t.name)
                )
            )
            plot_path_bar_eval = os.path.join(
                self.results_dir, "profiling_evaluation_breakdown.html"
            )
            fig_bar_eval.write_html(plot_path_bar_eval)
            print(
                f"\n[PROFILING] Evaluation breakdown plot saved in: {plot_path_bar_eval}"
            )

        # --- Detailed Simulation Analysis (env.step) ---
        print("\n--- DETAILED SIMULATION ANALYSIS (ENV.STEP) ---")
        df_sim = df[df["event"].isin(["data_collection", "evaluation"])].copy()
        if not df_sim.empty and "ice_duration" in df_sim.columns:
            df_sim["env_step_other_duration"] = (
                df_sim["env_step_duration"]
                - df_sim["ice_duration"]
                - df_sim["pg_duration"]
            )
            sim_analysis = (
                df_sim.groupby("event")[
                    ["ice_duration", "pg_duration", "env_step_other_duration"]
                ]
                .mean()
                .reset_index()
            )
            print("\n[PROFILING] Avg env.step component duration by event type (s):")
            print(sim_analysis)
            plot_data_sim = pd.melt(
                sim_analysis,
                id_vars=["event"],
                value_vars=["ice_duration", "pg_duration", "env_step_other_duration"],
                var_name="component",
                value_name="average_duration",
            )

            # ▼▼▼ START OF MODIFICATION ▼▼▼
            fig_bar_sim = px.bar(
                plot_data_sim,
                x="event",
                y="average_duration",
                color="component",
                barmode="group",  # Change to grouped bars
                title="Avg Simulation Time Breakdown (env.step)",
                labels={
                    "event": "Context",
                    "average_duration": "Avg Duration (s)",
                    "component": "Simulation Model",
                },
                category_orders={"event": ["data_collection", "evaluation"]},
            )
            # ▲▲▲ END OF MODIFICATION ▲▲▲

            new_names_sim = {
                "ice_duration": "ICE Model",
                "pg_duration": "PG Model",
                "env_step_other_duration": "Internal Logic (Other)",
            }
            fig_bar_sim.for_each_trace(lambda t: t.update(name=new_names_sim[t.name]))
            fig_bar_sim.update_xaxes(
                categoryarray=["data_collection", "evaluation"],
                title_text="Execution Context",
            )
            plot_path_bar_sim = os.path.join(
                self.results_dir, "profiling_simulation_breakdown.html"
            )
            fig_bar_sim.write_html(plot_path_bar_sim)
            print(
                f"\n[PROFILING] Simulation breakdown plot saved in: {plot_path_bar_sim}"
            )

    # ... (el resto de métodos de la clase TD3 no cambian) ...
    def _update_worker_weights(self):
        actor_weights = self.Actor.to("cpu").state_dict()
        for worker in self.workers:
            worker.set_weights.remote(actor_weights)
        self.Actor.to(self.device)

    def compute_critic_loss(self, batch):
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
        with torch.no_grad():
            actions_target_seq, _ = self.Actor_target(next_states_seq)
            noise = torch.normal(
                0.0,
                self.policy_noise,
                size=actions_target_seq.shape,
                device=self.device,
            ).clamp(-self.noise_clip, self.noise_clip)
            next_actions_seq = torch.clamp(actions_target_seq + noise, -1.0, 1.0)
            q1_t_seq, _ = self.Critic1_target(next_states_seq, next_actions_seq)
            q2_t_seq, _ = self.Critic2_target(next_states_seq, next_actions_seq)
            q_target_seq = torch.min(q1_t_seq, q2_t_seq)
            y_i_seq = rewards_seq + (1 - done_seq) * self.gamma * q_target_seq
        y_i_unroll = y_i_seq[:, self.B :, :].detach()
        with torch.no_grad():
            states_burn_in = states_seq[:, : self.B, :]
            actions_burn_in = actions_seq[:, : self.B, :]
            _, h_critic1 = self.Critic1(states_burn_in, actions_burn_in)
            _, h_critic2 = self.Critic2(states_burn_in, actions_burn_in)
        states_unroll = states_seq[:, self.B :, :]
        actions_unroll = actions_seq[:, self.B :, :]
        q1_unroll, _ = self.Critic1(
            states_unroll, actions_unroll, hidden_state=h_critic1
        )
        q2_unroll, _ = self.Critic2(
            states_unroll, actions_unroll, hidden_state=h_critic2
        )
        loss1 = F.mse_loss(q1_unroll, y_i_unroll)
        loss2 = F.mse_loss(q2_unroll, y_i_unroll)
        return loss1, loss2

    def compute_actor_loss(self, batch):
        state_batch, action_batch, _, _, _, _ = batch
        states_seq = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        actions_seq = torch.tensor(action_batch, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            states_burn_in = states_seq[:, : self.B, :]
            _, h_actor_burn_in = self.Actor(states_burn_in)
        states_unroll = states_seq[:, self.B :, :]
        actions_pred_unroll, _ = self.Actor(states_unroll, hidden_state=h_actor_burn_in)
        with torch.no_grad():
            actions_burn_in = actions_seq[:, : self.B, :]
            _, h_critic1_burn_in = self.Critic1(states_burn_in, actions_burn_in)
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
        print(f"\nRunning {n} evaluations...")
        returns = []
        env_eval = Environment(self.f_transicio, length_problem=self.length_problem)

        # These variables will accumulate times for the general analysis
        total_ice_duration = 0.0
        total_pg_duration = 0.0

        # We will save detailed timings of the evaluation here
        eval_detailed_timings = []

        for _ in range(n):
            obs = env_eval.reset()
            self.Actor.reset_states()
            episode_return = 0.0
            done = False
            while not done:
                a_scaled = self.choose_action(obs)
                a_phys = self.descale_action(a_scaled)

                # The call now correctly captures ice_t and pg_t
                obs, reward, terminated, truncated, ice_t, pg_t = env_eval.step(
                    a_phys, self.vel_target, profile_step=self.profile_training
                )

                if self.profile_training:
                    # Saving detailed data for final analysis
                    if ice_t:
                        ice_t["model"] = (
                            "ICE_Eval"  # Label to differentiate from collection
                        )
                        self.detailed_profiling_data.append(ice_t)
                    if pg_t:
                        pg_t["model"] = "PG_Eval"
                        self.detailed_profiling_data.append(pg_t)

                # Summing inference time for high-level analysis (the one you already had)
                if ice_t:
                    total_ice_duration += ice_t.get("3_model_inference", 0)
                if pg_t:
                    total_pg_duration += pg_t.get("3_model_inference", 0)

                episode_return += float(reward)
                done = terminated or truncated

            returns.append(episode_return)

        # Returning averages as before to not break _analyze_profiling_data logic
        avg_ice_duration = total_ice_duration / n if n > 0 else 0
        avg_pg_duration = total_pg_duration / n if n > 0 else 0

        return (float(np.mean(returns)), 0.0, 0.0, avg_ice_duration, avg_pg_duration)

    def save_models(self, file_name="actor.pth"):
        file_path = os.path.join(self.results_dir, file_name)
        torch.save(self.Actor.state_dict(), file_path)
        print(f"\n[Model Saved] Actor saved in: {file_path}")

    def plot_trajectory(
        self, vel_target: float = 70.0, K: int | None = None, title: str | None = None
    ):
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
            state, _, term, trunc, _, _ = env.step(a_phys, self.vel_target)

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
                label="States",
                method="update",
                args=[
                    {
                        "visible": [
                            t.name in ("mf", "brk", "ice_sp", "torque")
                            for t in fig.data
                        ]
                    },
                    {"title": f"{title} — States"},
                ],
            ),
            dict(
                label="Outputs",
                method="update",
                args=[
                    {"visible": [t.name in ("vel_out", "nox", "co") for t in fig.data]},
                    {"title": f"{title} — Outputs"},
                ],
            ),
        ]
        fig.update_layout(
            updatemenus=[dict(type="buttons", direction="right", buttons=buttons)]
        )
        file_path = os.path.join(self.results_dir, f"trag_{self.version}.html")
        fig.write_html(file_path)
        print(f"[Plot saved] {file_path}")
        return

    def evaluate_numeric(
        self, n: int = 30, vel_target: float = 70.0, file_name: str | None = None
    ):
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
                state, reward, term, trunc, _, _ = env.step(a_phys, vel_target)
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
        avg_early = np.mean(early_returns)
        avg_full = np.mean(full_returns)
        avg_step = np.mean(early_steps)
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

    def _init_or_check_buffer_meta(self):
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
                ok = (
                    old.get("obs_dim") == self.obs_dim
                    and old.get("act_dim") == self.act_dim
                    and old.get("S") == self.S
                )
                if not ok:
                    print(
                        f"[WARN] The disk buffer was created with another configuration (meta={old}). Continuing, but you might have incompatibilities."
                    )
            except Exception as e:
                print(f"[WARN] Could not read buffer meta: {e}")

    def _refresh_disk_index(self):
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
        path = os.path.join(self.buffer_dir, f"part_{self._disk_counter:06d}.npz")
        self._disk_counter += 1
        return path

    def _save_windows_to_disk(self, windows):
        if not windows:
            return 0
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

    def _load_windows_from_disk(self, n, shuffle=True, rng_seed=123):
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

    # ▼▼▼ ADD THIS NEW FULL METHOD TO THE TD3 CLASS ▼▼▼
    def _analyze_detailed_timings(self):
        """
        Analyzes in detail the times of each phase of predict_ice and predict_PG
        collected during the entire training.
        """
        if not self.detailed_profiling_data:
            print("\n[DETAILED PROFILING] No data collected to analyze.")
            return

        print("\n" + "=" * 60)
        print("--- DETAILED PREDICTION TIME ANALYSIS (ms) ---")
        print("=" * 60)

        df = pd.DataFrame(self.detailed_profiling_data)

        # Transform to long format for analysis and plotting
        df_long = pd.melt(
            df, id_vars=["model"], var_name="phase", value_name="duration_s"
        )
        df_long["duration_ms"] = df_long["duration_s"] * 1000

        # Print descriptive statistics
        summary = df_long.groupby(["model", "phase"])["duration_ms"].describe()
        print("\n[STATISTICS] Time summary by phase (in ms):")
        print(summary.to_string(float_format="%.4f"))

        # Create boxplot
        print("\n[PLOT] Generating time distribution visualization...")
        phase_order = ["1_package_input", "2_scale", "3_model_inference", "4_descale"]
        df_long["phase"] = pd.Categorical(
            df_long["phase"], categories=phase_order, ordered=True
        )

        fig = px.box(
            df_long,
            x="phase",
            y="duration_ms",
            color="model",
            title="Prediction Time Distribution in RL Environment",
            labels={
                "phase": "Prediction Phase",
                "duration_ms": "Duration (ms)",
                "model": "Model",
            },
        )
        fig.update_layout(
            xaxis_title="Prediction Phase",
            yaxis_title="Duration (ms)",
            legend_title="Model",
            xaxis_tickangle=-30,
        )

        plot_path = os.path.join(
            self.results_dir, "profiling_detailed_prediction_analysis.html"
        )
        fig.write_html(plot_path)
        print(f"\n[DONE] Detailed analysis plot saved in: {plot_path}")
        print("=" * 60)

    # ▲▲▲ END OF NEW METHOD ▲▲▲
