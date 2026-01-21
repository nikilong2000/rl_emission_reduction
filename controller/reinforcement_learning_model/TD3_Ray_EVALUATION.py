import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
warnings.filterwarnings('ignore')

from environment_EVALUATION import Environment
import ray

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@ray.remote
class RolloutWorker:
    def __init__(self, f_transicio, worker_id, scaler_params, act_dim, length_problem, vel_target, B, U, profile_training=False):
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
        print(f"[Worker {self.worker_id}] Creado y listo.")

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
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            tf.config.set_visible_devices([], 'GPU')
        except Exception: pass

    def collect_windows(self):
        start_time = time.time()
        action_logic_duration_total = 0.0
        env_step_duration_total = 0.0
        ice_duration_total = 0.0
        pg_duration_total = 0.0
        
        obs = self.env.reset()
        self.actor_worker.reset_states()
        episode_data = {k: [] for k in ['states', 'actions', 'rewards', 'next_states', 'terminated', 'truncated']}

        while True:
            action_logic_start = time.time()
            action_scaled = self.choose_action(obs)
            noise_value = self.exploration_noise.sample()
            action_scaled_noisy = np.clip(action_scaled + noise_value, -1, 1)
            action_phys = self.descale_action(action_scaled_noisy)
            action_logic_duration_total += time.time() - action_logic_start

            env_step_start = time.time()
            next_obs, reward, terminated, truncated, ice_t, pg_t = self.env.step(
                action_phys, self.vel_target, profile_step=self.profile_training
            )
            env_step_duration_total += time.time() - env_step_start

            if self.profile_training:
                ice_duration_total += ice_t
                pg_duration_total += pg_t

            done = terminated or truncated
            for k, v in zip(episode_data.keys(), [obs, action_scaled_noisy, reward, next_obs, terminated, truncated]):
                episode_data[k].append(v)
            obs = next_obs
            if done: break

        windows = []
        episode_len = len(episode_data['states'])
        for i in range(0, max(0, episode_len - self.S + 1), self.U):
            s_window  = np.stack(episode_data['states'][i:i+self.S]).astype(np.float32)
            a_window  = np.stack(episode_data['actions'][i:i+self.S]).astype(np.float32)
            r_window  = np.asarray(episode_data['rewards'][i:i+self.S], dtype=np.float32)
            ns_window = np.stack(episode_data['next_states'][i:i+self.S]).astype(np.float32)
            t_window  = np.asarray(episode_data['terminated'][i:i+self.S], dtype=np.bool_)
            tr_window = np.asarray(episode_data['truncated'][i:i+self.S], dtype=np.bool_)
            windows.append((s_window, a_window, r_window, ns_window, t_window, tr_window))

        total_duration = time.time() - start_time
        return (windows, total_duration, self.worker_id, 
                action_logic_duration_total, env_step_duration_total, 
                ice_duration_total, pg_duration_total)

class TD3:
    def __init__(self, f_transicio, version="test", act_dim=3, obs_dim=5, length_problem=1200, replay_size=1000000, batch_size=256, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2, early_stop=50, numeric_eval=False, scaler_params=None, vel_target=70, num_workers=2, U=64, B=32, reuse_warmup_buffer = False, buffer_dir= "buffer", profile_training = False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo de cómputo del LEARNER: {self.device}")
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
                f_transicio=f_transicio, worker_id=i, scaler_params=scaler_params,
                act_dim=act_dim, length_problem=length_problem, vel_target=vel_target,
                B=self.B, U=self.U, profile_training=self.profile_training
            ) for i in range(self.num_workers)
        ]
        print(f"✅ {self.num_workers} workers remotos creados.")

        if self.profile_training:
            self.profiling_data = []
            self.wait_for_data_start_time = None 
            self.is_first_learn_step = True

    def learn(self, total_timesteps, learning_starts=1000, train_freq=4, gradient_steps=4):
        print(f"🔥 Iniciando fase de calentamiento del buffer. Objetivo: {learning_starts} ventanas.")
        self._update_worker_weights()
        latest_actor_weights = {k: v.cpu() for k, v in self.Actor.state_dict().items()}

        if self.reuse_warmup_buffer:
            self._refresh_disk_index()
            if self._disk_window_count >= learning_starts:
                loaded = self._load_windows_from_disk(learning_starts, shuffle=True)
                print(f"✅ Warm-up cargado desde disco: {loaded}/{learning_starts} ventanas.")

        tasks_to_workers = { worker.collect_windows.remote(): worker for worker in self.workers }
        pending_tasks = list(tasks_to_workers.keys())
        steps_since_last_train = train_freq

        while len(self.replay_buffer) < learning_starts:
            ready_tasks_refs, pending_tasks = ray.wait(pending_tasks, num_returns=1)
            ready_ref = ready_tasks_refs[0]
            worker = tasks_to_workers.pop(ready_ref)
            
            result_windows, collection_duration, worker_id, action_t, step_t, ice_t, pg_t = ray.get(ready_ref)
            
            if self.profile_training:
                self.profiling_data.append({
                    'event': 'data_collection', 'duration': collection_duration, 'worker_id': worker_id, 'timestamp': time.time(),
                    'action_logic_duration': action_t, 'env_step_duration': step_t, 'ice_duration': ice_t, 'pg_duration': pg_t
                })

            if self.reuse_warmup_buffer:
                self._save_windows_to_disk(result_windows)
                if len(self.replay_buffer) < learning_starts and self._disk_window_count >= learning_starts:
                    needed = learning_starts - len(self.replay_buffer)
                    self._load_windows_from_disk(needed, shuffle=True)
            else:
                for w in result_windows: self.replay_buffer.put(w)
                
            update_duration, worker_id = ray.get(worker.set_weights.remote(latest_actor_weights))
            if self.profile_training:
                self.profiling_data.append({'event': 'policy_update','duration': update_duration,'worker_id': worker_id,'timestamp': time.time()})
                
            new_task_ref = worker.collect_windows.remote()
            tasks_to_workers[new_task_ref] = worker 
            pending_tasks.append(new_task_ref)
            
            if self.reuse_warmup_buffer:
                print(f"\r[WARM-UP disco] guardadas: {self._disk_window_count} | en RAM: {len(self.replay_buffer)}/{learning_starts}", end="")
            else:
                print(f"\rBuffer: {len(self.replay_buffer)}/{learning_starts}", end="")
        print("\n✅ Fase de calentamiento completada.")

        print("\n🚀 Iniciando fase de entrenamiento principal asíncrona...")
        timesteps_collected = 0
        updates_done_since_eval = 0
        all_rewards_eval = []
        if self.profile_training:
             self.wait_for_data_start_time = time.time()

        while timesteps_collected < total_timesteps:
            if len(self.replay_buffer) > self.N and steps_since_last_train >= train_freq:
                if self.profile_training and self.wait_for_data_start_time is not None:
                    if not self.is_first_learn_step:
                        wait_duration = time.time() - self.wait_for_data_start_time
                        self.profiling_data.append({'event': 'learner_wait', 'duration': wait_duration, 'worker_id': 'learner', 'timestamp': time.time()})
                    self.is_first_learn_step = False
                    self.wait_for_data_start_time = None

                print(f"\n--- Ráfaga de Entrenamiento ({gradient_steps} pasos) ---")
                print(f"Iteraciones de entrenamiento: {self.total_iterations} | Timesteps: ~{timesteps_collected}/{total_timesteps} | Buffer: {len(self.replay_buffer)}")
                learn_start_time = time.time()
                for _ in range(gradient_steps):
                    batch = self.replay_buffer.get(self.N)
                    critic1_loss, critic2_loss = self.compute_critic_loss(batch); self.optim_critic1.zero_grad(set_to_none=True); critic1_loss.backward(); self.optim_critic1.step()
                    self.optim_critic2.zero_grad(set_to_none=True); critic2_loss.backward(); self.optim_critic2.step()
                    self.total_iterations += 1; updates_done_since_eval +=1
                    if self.total_iterations % self.policy_delay == 0:
                        actor_loss = self.compute_actor_loss(batch); self.optim_actor.zero_grad(set_to_none=True); actor_loss.backward(); self.optim_actor.step()
                        soft_update(self.Actor_target, self.Actor, self.tau); soft_update(self.Critic1_target, self.Critic1, self.tau); soft_update(self.Critic2_target, self.Critic2, self.tau)
                        latest_actor_weights = {k: v.cpu() for k, v in self.Actor.state_dict().items()}
                if self.profile_training:
                    learn_duration = time.time() - learn_start_time
                    self.profiling_data.append({'event': 'learn_step','duration': learn_duration,'worker_id': 'learner','timestamp': time.time()})
                print(f"\n--- Ráfaga de Entrenamiento Terminada ---")
                steps_since_last_train = 0
                if self.profile_training:
                    self.wait_for_data_start_time = time.time()
            
            ready_tasks_refs, pending_tasks = ray.wait(pending_tasks, num_returns=1, timeout=0.0)
            if ready_tasks_refs:
                ready_ref = ready_tasks_refs[0]
                
                worker_windows, collection_duration, worker_id, action_t, step_t, ice_t, pg_t = ray.get(ready_ref)
                if self.profile_training:
                    self.profiling_data.append({
                        'event': 'data_collection', 'duration': collection_duration, 'worker_id': worker_id, 'timestamp': time.time(),
                        'action_logic_duration': action_t, 'env_step_duration': step_t, 'ice_duration': ice_t, 'pg_duration': pg_t
                    })

                num_new_windows = len(worker_windows)
                for window in worker_windows: self.replay_buffer.put(window)
                new_steps = num_new_windows * self.U
                timesteps_collected += new_steps
                steps_since_last_train += new_steps
                worker = tasks_to_workers.pop(ready_ref)
                update_duration, worker_id = ray.get(worker.set_weights.remote(latest_actor_weights))
                if self.profile_training:
                    self.profiling_data.append({'event': 'policy_update','duration': update_duration,'worker_id': worker_id,'timestamp': time.time()})
                new_task_ref = worker.collect_windows.remote()
                tasks_to_workers[new_task_ref] = worker
                pending_tasks.append(new_task_ref)

            if updates_done_since_eval >= gradient_steps:
                print(f"\n--- Ráfaga de Evaluación Periódica ---")
                
                eval_start_time = time.time()
                current_eval, avg_action_t, avg_step_t, avg_ice_t, avg_pg_t = self.eval_episodes(n=self.num_workers)
                
                if self.profile_training:
                    eval_duration = time.time() - eval_start_time
                    self.profiling_data.append({
                        'event': 'evaluation', 'duration': eval_duration, 'worker_id': 'learner', 'timestamp': time.time(),
                        'action_logic_duration': avg_action_t, 'env_step_duration': avg_step_t, 'ice_duration': avg_ice_t, 'pg_duration': avg_pg_t
                    })

                all_rewards_eval.append(current_eval)
                print(f'\nTimesteps: ~{timesteps_collected}/{total_timesteps}, Eval Reward: {current_eval:.2f}')
                updates_done_since_eval = 0 
                if current_eval > self.best_eval_reward:
                    self.best_eval_reward = current_eval; self.no_improve = 0; self.save_models("actor_best.pth")
                else:
                    self.no_improve += 1
                if self.no_improve >= self.early_stop:
                    print(f"\n[EARLY STOP] No improvement in {self.early_stop} evaluations."); break
                    
        print("\n[Training Finished] Guardando el modelo final...")
        self.save_models(file_name="actor_final.pth")
        if self.profile_training:
            self._analyze_profiling_data()
        self.plot_trajectory()
        if self.numeric_eval:
            self.evaluate_numeric(n=30, vel_target=self.vel_target)

    # Dins de la classe TD3, substitueix aquest mètode sencer:
    # Dins de la classe TD3, substitueix aquest mètode sencer:

    def _analyze_profiling_data(self):
        """
        Procesa los datos de tiempo recolectados, imprime estadísticas y genera cuatro gráficos
        con las mejoras de visualización solicitadas.
        """
        if not self.profiling_data:
            print("\n[PROFILING] No se recolectaron datos para analizar.")
            return

        print("\n--- ANÁLISIS DE CUELLOS DE BOTELLA ---")
        df = pd.DataFrame(self.profiling_data).fillna(0)
        csv_path = os.path.join(self.results_dir, "profiling_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"[PROFILING] Datos de tiempo guardados en: {csv_path}")

        print("\n[PROFILING] Resumen de tiempos por evento (en segundos):")
        summary = df.groupby('event')['duration'].describe()
        print(summary)

        # --- Gráfico de Cajas (boxplot) con Eje X Reordenado ---
        category_order = ['learn_step', 'evaluation', 'learner_wait', 'policy_update', 'data_collection']
        fig_box = px.box(
            df, x='event', y='duration', color='event', points='all',
            title='Distribución de Tiempos por Fase del Entrenamiento',
            labels={'event': 'Fase del Entrenamiento', 'duration': 'Duración (segundos)'},
            category_orders={'event': category_order}
        )
        fig_box.update_layout(showlegend=False)
        plot_path_box = os.path.join(self.results_dir, "profiling_bottleneck_analysis.html")
        fig_box.write_html(plot_path_box)
        print(f"\n[PROFILING] Gráfico de análisis de cuellos de botella guardado en: {plot_path_box}")

        # --- Análisis y Gráfico de Pastel para el Learner ---
        print("\n--- ANÁLISIS DE ACTIVIDAD DEL LEARNER ---")
        df_learner = df[df['worker_id'] == 'learner'].copy()
        if not df_learner.empty:
            time_summary = df_learner.groupby('event')['duration'].sum()
            total_learner_time = time_summary.sum()
            if total_learner_time > 0:
                time_percentage = (time_summary / total_learner_time) * 100
                print("\n[PROFILING] Porcentaje de tiempo del Learner por actividad:")
                for event, percentage in time_percentage.items(): print(f"- {event}: {percentage:.2f}%")
                pie_data = time_summary.reset_index(); pie_data.columns = ['event', 'total_duration']
                fig_pie = px.pie(pie_data, values='total_duration', names='event', title='Distribución del Tiempo de Actividad del Learner', labels={'event': 'Actividad', 'total_duration': 'Tiempo Total (s)'})
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                plot_path_pie = os.path.join(self.results_dir, "profiling_learner_activity.html")
                fig_pie.write_html(plot_path_pie)
                print(f"\n[PROFILING] Gráfico de actividad del Learner guardado en: {plot_path_pie}")

        # --- Análisis detallado de Data Collection ---
        print("\n--- ANÁLISIS DETALLADO DE DATA COLLECTION ---")
        df_coll = df[df['event'] == 'data_collection'].copy()
        if not df_coll.empty:
            df_coll['other_duration'] = df_coll['duration'] - df_coll['env_step_duration'] - df_coll['action_logic_duration']
            worker_analysis = df_coll.groupby('worker_id')[['duration', 'env_step_duration', 'action_logic_duration', 'other_duration']].mean().reset_index()
            print("\n[PROFILING] Duración media de componentes de Data Collection por Worker (s):"); print(worker_analysis)
            plot_data = pd.melt(worker_analysis, id_vars=['worker_id'], value_vars=['env_step_duration', 'action_logic_duration', 'other_duration'], var_name='component', value_name='average_duration')
            fig_bar = px.bar(plot_data, x='worker_id', y='average_duration', color='component', title='Desglose del Tiempo Medio en Data Collection por Worker', labels={'worker_id': 'ID del Worker', 'average_duration': 'Duración Media (s)', 'component': 'Componente de la Tarea'}, category_orders={'component': ['env_step_duration', 'action_logic_duration', 'other_duration']})
            new_names = {'env_step_duration': 'Simulación (env.step)', 'action_logic_duration': 'Lógica de Acción', 'other_duration': 'Gestión (Otros)'}
            fig_bar.for_each_trace(lambda t: t.update(name=new_names[t.name]))
            plot_path_bar = os.path.join(self.results_dir, "profiling_datacollector_breakdown.html")
            fig_bar.write_html(plot_path_bar)
            print(f"\n[PROFILING] Gráfico de desglose de Data Collector guardado en: {plot_path_bar}")

        # --- Análisis detallado de Evaluation (Episodio Promedio) ---
        print("\n--- ANÁLISIS DETALLADO DE EVALUATION ---")
        df_eval = df[df['event'] == 'evaluation'].copy()
        if not df_eval.empty and 'action_logic_duration' in df_eval.columns:
            avg_total_eval_duration = df_eval['duration'].mean()
            avg_action_logic_per_episode = df_eval['action_logic_duration'].mean()
            avg_env_step_per_episode = df_eval['env_step_duration'].mean()
            avg_episode_duration = avg_total_eval_duration / self.num_workers
            avg_other_duration = avg_episode_duration - avg_action_logic_per_episode - avg_env_step_per_episode
            print("\n[PROFILING] Desglose de tiempo de un episodio de evaluación PROMEDIO (s):")
            print(f"- Lógica de Acción: {avg_action_logic_per_episode:.4f}\n- Simulación (env.step): {avg_env_step_per_episode:.4f}\n- Gestión (Otros): {max(0, avg_other_duration):.4f}\n--------------------------------------\n- Duración total episodio: {avg_episode_duration:.4f}")

            bar_data_eval = pd.DataFrame({
                'component': ['Simulación (env.step)', 'Lógica de Acción', 'Gestión (Otros)'],
                'duration': [avg_env_step_per_episode, avg_action_logic_per_episode, max(0, avg_other_duration)],
                'episode': 'Episodio de Evaluación Promedio'
            })
            fig_bar_eval = px.bar(
                bar_data_eval, x='episode', y='duration', color='component',
                title='Desglose del Tiempo en un Episodio de Evaluación Promedio',
                labels={'episode': '', 'duration': 'Duración Media (s)', 'component': 'Componente'}
            )
            fig_bar_eval.for_each_trace(lambda t: t.update(name=new_names.get(t.name.replace("duration", ""), t.name)))
            plot_path_bar_eval = os.path.join(self.results_dir, "profiling_evaluation_breakdown.html")
            fig_bar_eval.write_html(plot_path_bar_eval)
            print(f"\n[PROFILING] Gráfico de desglose de Evaluación guardado en: {plot_path_bar_eval}")

        # --- Análisis detallado de la Simulación (env.step) ---
        print("\n--- ANÁLISIS DETALLADO DE LA SIMULACIÓN (ENV.STEP) ---")
        df_sim = df[df['event'].isin(['data_collection', 'evaluation'])].copy()
        if not df_sim.empty and 'ice_duration' in df_sim.columns:
            df_sim['env_step_other_duration'] = df_sim['env_step_duration'] - df_sim['ice_duration'] - df_sim['pg_duration']
            sim_analysis = df_sim.groupby('event')[['ice_duration', 'pg_duration', 'env_step_other_duration']].mean().reset_index()
            print("\n[PROFILING] Duración media de componentes de env.step por tipo de evento (s):"); print(sim_analysis)
            plot_data_sim = pd.melt(sim_analysis, id_vars=['event'], value_vars=['ice_duration', 'pg_duration', 'env_step_other_duration'], var_name='component', value_name='average_duration')

            # ▼▼▼ INICI DE LA MODIFICACIÓ ▼▼▼
            fig_bar_sim = px.bar(
                plot_data_sim,
                x='event',
                y='average_duration',
                color='component',
                barmode='group', # Canvi a barres agrupades
                title='Desglose del Tiempo Medio de Simulación (env.step)',
                labels={'event': 'Contexto', 'average_duration': 'Duración Media (s)', 'component': 'Modelo de Simulación'},
                category_orders={'event': ['data_collection', 'evaluation']}
            )
            # ▲▲▲ FINAL DE LA MODIFICACIÓ ▲▲▲

            new_names_sim = {'ice_duration': 'ICE Model', 'pg_duration': 'PG Model', 'env_step_other_duration': 'Lógica Interna (Otros)'}
            fig_bar_sim.for_each_trace(lambda t: t.update(name=new_names_sim[t.name]))
            fig_bar_sim.update_xaxes(categoryarray=['data_collection', 'evaluation'], title_text="Contexto de Ejecución")
            plot_path_bar_sim = os.path.join(self.results_dir, "profiling_simulation_breakdown.html")
            fig_bar_sim.write_html(plot_path_bar_sim)
            print(f"\n[PROFILING] Gráfico de desglose de simulación guardado en: {plot_path_bar_sim}")


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
        rewards_seq = torch.tensor(reward_b, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_seq = torch.tensor(next_state_b, dtype=torch.float32, device=self.device)
        done_seq = torch.tensor(term_b | trunc_b, dtype=torch.float32, device=self.device).unsqueeze(-1)
        with torch.no_grad():
            actions_target_seq, _ = self.Actor_target(next_states_seq)
            noise = torch.normal(0.0, self.policy_noise, size=actions_target_seq.shape, device=self.device).clamp(-self.noise_clip, self.noise_clip)
            next_actions_seq = torch.clamp(actions_target_seq + noise, -1.0, 1.0)
            q1_t_seq, _ = self.Critic1_target(next_states_seq, next_actions_seq)
            q2_t_seq, _ = self.Critic2_target(next_states_seq, next_actions_seq)
            q_target_seq = torch.min(q1_t_seq, q2_t_seq)
            y_i_seq = rewards_seq + (1 - done_seq) * self.gamma * q_target_seq
        y_i_unroll = y_i_seq[:, self.B:, :].detach()
        with torch.no_grad():
            states_burn_in = states_seq[:, :self.B, :]
            actions_burn_in = actions_seq[:, :self.B, :]
            _, h_critic1 = self.Critic1(states_burn_in, actions_burn_in)
            _, h_critic2 = self.Critic2(states_burn_in, actions_burn_in)
        states_unroll = states_seq[:, self.B:, :]
        actions_unroll = actions_seq[:, self.B:, :]
        q1_unroll, _ = self.Critic1(states_unroll, actions_unroll, hidden_state=h_critic1)
        q2_unroll, _ = self.Critic2(states_unroll, actions_unroll, hidden_state=h_critic2)
        loss1 = F.mse_loss(q1_unroll, y_i_unroll)
        loss2 = F.mse_loss(q2_unroll, y_i_unroll)
        return loss1, loss2

    def compute_actor_loss(self, batch):
        state_batch, action_batch, _, _, _, _ = batch
        states_seq = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        actions_seq = torch.tensor(action_batch, dtype=torch.float32).to(self.device) 
        with torch.no_grad():
            states_burn_in = states_seq[:, :self.B, :]
            _, h_actor_burn_in = self.Actor(states_burn_in)
        states_unroll = states_seq[:, self.B:, :]
        actions_pred_unroll, _ = self.Actor(states_unroll, hidden_state=h_actor_burn_in)
        with torch.no_grad():
            actions_burn_in = actions_seq[:, :self.B, :]
            _, h_critic1_burn_in = self.Critic1(states_burn_in, actions_burn_in)
        for p in self.Critic1.parameters(): p.requires_grad_(False)
        q_values_unroll, _ = self.Critic1(states_unroll, actions_pred_unroll, hidden_state=h_critic1_burn_in)
        actor_loss = -q_values_unroll.mean()
        for p in self.Critic1.parameters(): p.requires_grad_(True)
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

    # Dins de la classe TD3
    def eval_episodes(self, n: int = 3):
        print(f"\nEjecutando {n} evaluaciones...")
        returns = []
        # CORRECCIÓ: La variable es diu env_eval, no self.env_eval
        env_eval = Environment(self.f_transicio, length_problem=self.length_problem)

        total_action_logic_duration = 0.0
        total_env_step_duration = 0.0
        total_ice_duration = 0.0
        total_pg_duration = 0.0

        for _ in range(n):
            obs = env_eval.reset()
            self.Actor.reset_states()
            episode_return = 0.0
            done = False
            while not done:
                action_logic_start = time.time()
                a_scaled = self.choose_action(obs) 
                a_phys = self.descale_action(a_scaled)
                total_action_logic_duration += time.time() - action_logic_start

                env_step_start = time.time()
                # CORRECCIÓ: Utilitzar la variable local 'env_eval'
                obs, reward, terminated, truncated, ice_t, pg_t = env_eval.step(
                    a_phys, self.vel_target, profile_step=self.profile_training
                )
                total_env_step_duration += time.time() - env_step_start

                if self.profile_training:
                    total_ice_duration += ice_t
                    total_pg_duration += pg_t

                episode_return += float(reward)
                done = terminated or truncated

            returns.append(episode_return)

        avg_action_logic_duration = total_action_logic_duration / n
        avg_env_step_duration = total_env_step_duration / n
        avg_ice_duration = total_ice_duration / n
        avg_pg_duration = total_pg_duration / n

        return (float(np.mean(returns)), avg_action_logic_duration, avg_env_step_duration,
                avg_ice_duration, avg_pg_duration)


    def save_models(self, file_name="actor.pth"):
        file_path = os.path.join(self.results_dir, file_name)
        torch.save(self.Actor.state_dict(), file_path)
        print(f"\n[Model Saved] Actor guardado en: {file_path}")

    def plot_trajectory(self, vel_target: float = 70.0, K: int | None = None, title: str | None = None):
        env = Environment(self.f_transicio, length_problem=self.length_problem)
        K = K or env.length_problem
        title = title or f"TD3 – {self.version}"
        state = env.reset(vel_target=vel_target)
        self.Actor.reset_states()
        log = {k: [] for k in ("step", "mf", "brk", "ice_sp", "torque", "vel_out", "nox", "co")}
        for step in range(K):
            a_scaled = self.choose_action(state)
            a_phys = self.descale_action(a_scaled)
            state, _, term, trunc, _, _ = env.step(a_phys, self.vel_target)
            def as_float(x): return float(x) if torch.is_tensor(x) else x
            log["step"].append(step)
            log["mf"].append(as_float(env.mf))
            log["brk"].append(as_float(env.brk))
            log["ice_sp"].append(as_float(env.ice_sp))
            log["torque"].append(as_float(env.torque))
            log["vel_out"].append(as_float(env.vel_out))
            log["nox"].append(as_float(env.nox))
            log["co"].append(as_float(env.co))
            if term or trunc: break
        df = pd.DataFrame(log)
        fig = px.line(df, x="step",y=["mf", "brk", "ice_sp", "torque","vel_out", "nox", "co"],title=title)
        for tr in fig.data:
            if tr.name in ("vel_out", "nox", "co"): tr.visible = "legendonly"
        buttons = [
            dict(label="Estados", method="update", args=[{"visible": [t.name in ("mf", "brk", "ice_sp", "torque") for t in fig.data]}, {"title": f"{title} — Estados"}]),
            dict(label="Salidas", method="update", args=[{"visible": [t.name in ("vel_out", "nox", "co") for t in fig.data]}, {"title": f"{title} — Salidas"}]),
        ]
        fig.update_layout(updatemenus=[dict(type="buttons", direction="right", buttons=buttons)])
        file_path = os.path.join(self.results_dir, f"trag_{self.version}.html")
        fig.write_html(file_path)
        print(f"[Plot saved] {file_path}")
        return

    def evaluate_numeric(self, n: int = 30, vel_target: float = 70.0, file_name: str | None = None):
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
                state, reward, term, trunc, _, _ = env.step(a_phys,vel_target)
                fret += float(reward)
                if cut is None:
                    eret += float(reward)
                    if term or trunc: cut = t + 1
                if env.step_count >= K: break
            early_returns.append(eret)
            full_returns.append(fret)
            early_steps.append(cut if cut is not None else K)
        avg_early = np.mean(early_returns)
        avg_full  = np.mean(full_returns)
        avg_step  = np.mean(early_steps)
        if file_name is None: file_name = f"numeric_eval_{self.version}.txt"
        path = os.path.join(self.results_dir, file_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Numeric evaluation — TD3 version: {self.version}\n")
            f.write(f"Episodes: {n}\n\n")
            f.write("ep\tstep_cut\tearly_ret\tfull_ret\n")
            for i, (sc, er, fr) in enumerate(zip(early_steps, early_returns, full_returns), 1):
                f.write(f"{i}\t{sc}\t{er:.4f}\t{fr:.4f}\n")
            f.write("\nAverages\n")
            f.write(f"mean_step_cut: {avg_step:.2f}\n")
            f.write(f"mean_early_return: {avg_early:.4f}\n")
            f.write(f"mean_full_return:  {avg_full:.4f}\n")
        print(f"[Numeric eval saved] {path}")
        return

    def _init_or_check_buffer_meta(self):
        meta = dict(obs_dim=self.obs_dim, act_dim=self.act_dim, S=self.S, B=self.B, U=self.U)
        if not os.path.exists(self._buffer_meta_path):
            with open(self._buffer_meta_path, "w") as f: json.dump(meta, f)
        else:
            try:
                with open(self._buffer_meta_path, "r") as f: old = json.load(f)
                ok = (old.get("obs_dim")==self.obs_dim and old.get("act_dim")==self.act_dim and old.get("S")==self.S)
                if not ok: print(f"[WARN] El buffer en disco fue creado con otra configuración (meta={old}). Se seguirá, pero podrías tener incompatibilidades.")
            except Exception as e: print(f"[WARN] No pude leer meta del buffer: {e}")

    def _refresh_disk_index(self):
        self._disk_files = sorted(glob.glob(os.path.join(self.buffer_dir, "part_*.npz")))
        self._disk_counter = len(self._disk_files)
        self._disk_window_count = 0
        for fp in self._disk_files:
            try:
                with np.load(fp) as data: self._disk_window_count += int(data["s"].shape[0])
            except Exception as e: print(f"[WARN] Archivo de buffer corrupto o ilegible: {fp} ({e})")

    def _next_buffer_filepath(self):
        path = os.path.join(self.buffer_dir, f"part_{self._disk_counter:06d}.npz")
        self._disk_counter += 1
        return path

    def _save_windows_to_disk(self, windows):
        if not windows: return 0
        s  = np.stack([w[0] for w in windows], axis=0).astype(np.float32)
        a  = np.stack([w[1] for w in windows], axis=0).astype(np.float32)
        r  = np.stack([w[2] for w in windows], axis=0).astype(np.float32)
        ns = np.stack([w[3] for w in windows], axis=0).astype(np.float32)
        t  = np.stack([w[4] for w in windows], axis=0).astype(np.bool_)
        tr = np.stack([w[5] for w in windows], axis=0).astype(np.bool_)
        fp = self._next_buffer_filepath()
        np.savez_compressed(fp, s=s, a=a, r=r, ns=ns, t=t, tr=tr)
        n = s.shape[0]
        self._disk_window_count += n
        return n

    def _load_windows_from_disk(self, n, shuffle=True, rng_seed=123):
        self._refresh_disk_index()
        if n <= 0 or self._disk_window_count <= 0: return 0
        files = list(self._disk_files)
        rng = np.random.default_rng(rng_seed)
        if shuffle: rng.shuffle(files)
        loaded = 0
        for fp in files:
            with np.load(fp) as data:
                m = int(data["s"].shape[0])
                idx = np.arange(m)
                if shuffle: rng.shuffle(idx)
                take = min(n - loaded, m)
                sel = idx[:take]
                for i in sel:
                    window = (data["s"][i], data["a"][i], data["r"][i], data["ns"][i], data["t"][i], data["tr"][i])
                    self.replay_buffer.put(window)
                loaded += take
                if loaded >= n: break
        return loaded