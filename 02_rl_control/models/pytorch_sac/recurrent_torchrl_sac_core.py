import os
import pickle
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = None

try:
    from torchrl.modules.distributions import TanhNormal
except ImportError:
    TanhNormal = None

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def torchrl_available() -> bool:
    return TensorDict is not None and TanhNormal is not None


def resolve_model_path(path: str) -> str:
    if os.path.exists(path):
        return path
    if os.path.exists(path + ".zip"):
        return path + ".zip"
    raise FileNotFoundError(f"Model file '{path}' not found.")


def resolve_torch_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but no CUDA device is available.")
    return device


def configure_torch_performance(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def compute_state_entropy(obs_array: np.ndarray, bins: int = 20) -> float:
    if obs_array.ndim == 1:
        obs_array = obs_array[:, np.newaxis]
    entropies = []
    for d in range(obs_array.shape[1]):
        counts, _ = np.histogram(obs_array[:, d], bins=bins)
        probs = counts / (counts.sum() + 1e-12)
        probs = probs[probs > 0]
        entropies.append(-np.sum(probs * np.log(probs)))
    return float(np.mean(entropies))


class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == len(self.mean.shape):
            values = values[np.newaxis, ...]
        batch_mean = np.mean(values, axis=0)
        batch_var = np.var(values, axis=0)
        batch_count = values.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = total_count

    def normalize(self, values: np.ndarray, clip_obs: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        normalized = (values - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normalized, -clip_obs, clip_obs).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.mean = np.asarray(state_dict["mean"], dtype=np.float64)
        self.var = np.asarray(state_dict["var"], dtype=np.float64)
        self.count = float(state_dict["count"])

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.load_state_dict(state)


class EpisodeReplayBuffer:
    def __init__(self, capacity_steps: int):
        self.capacity_steps = capacity_steps
        self.episodes: Deque[Dict[str, np.ndarray]] = deque()
        self.current_episode: List[Tuple[np.ndarray, ...]] = []
        self.total_transitions = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.current_episode.append(
            (
                np.asarray(obs, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                np.float32(reward),
                np.asarray(next_obs, dtype=np.float32),
                np.float32(done),
            )
        )
        if done:
            self.end_episode()

    def end_episode(self) -> None:
        if not self.current_episode:
            return

        obs = np.stack([t[0] for t in self.current_episode], axis=0)
        actions = np.stack([t[1] for t in self.current_episode], axis=0)
        rewards = np.asarray([t[2] for t in self.current_episode], dtype=np.float32)
        next_obs = np.stack([t[3] for t in self.current_episode], axis=0)
        dones = np.asarray([t[4] for t in self.current_episode], dtype=np.float32)

        episode = {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
        }
        self.episodes.append(episode)
        self.total_transitions += len(rewards)
        self.current_episode = []

        while self.total_transitions > self.capacity_steps and self.episodes:
            removed = self.episodes.popleft()
            self.total_transitions -= len(removed["rewards"])

    def can_sample(self, batch_size: int, sequence_length: int) -> bool:
        return (
            len(self.episodes) > 0
            and self.total_transitions >= batch_size
            and self.total_transitions >= sequence_length
        )

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        as_tensordict: bool = False,
    ) -> Union[Dict[str, np.ndarray], "TensorDict"]:
        if not self.episodes:
            raise ValueError("Replay buffer is empty.")

        lengths = np.asarray(
            [len(ep["rewards"]) for ep in self.episodes], dtype=np.int64
        )
        probs = lengths / lengths.sum()
        chosen_episode_indices = np.random.choice(
            len(self.episodes), size=batch_size, replace=True, p=probs
        )

        obs_dim = self.episodes[0]["obs"].shape[-1]
        action_dim = self.episodes[0]["actions"].shape[-1]

        obs_batch = np.zeros((batch_size, sequence_length, obs_dim), dtype=np.float32)
        next_obs_batch = np.zeros_like(obs_batch)
        action_batch = np.zeros(
            (batch_size, sequence_length, action_dim), dtype=np.float32
        )
        reward_batch = np.zeros((batch_size, sequence_length, 1), dtype=np.float32)
        done_batch = np.zeros((batch_size, sequence_length, 1), dtype=np.float32)
        mask_batch = np.zeros((batch_size, sequence_length, 1), dtype=np.float32)

        for b, ep_idx in enumerate(chosen_episode_indices):
            ep = self.episodes[int(ep_idx)]
            ep_len = len(ep["rewards"])
            start = np.random.randint(0, ep_len)
            end = min(start + sequence_length, ep_len)
            seq_len = end - start

            obs_batch[b, :seq_len] = ep["obs"][start:end]
            next_obs_batch[b, :seq_len] = ep["next_obs"][start:end]
            action_batch[b, :seq_len] = ep["actions"][start:end]
            reward_batch[b, :seq_len, 0] = ep["rewards"][start:end]
            done_batch[b, :seq_len, 0] = ep["dones"][start:end]
            mask_batch[b, :seq_len, 0] = 1.0

        batch = {
            "obs": obs_batch,
            "next_obs": next_obs_batch,
            "actions": action_batch,
            "rewards": reward_batch,
            "dones": done_batch,
            "mask": mask_batch,
        }

        if as_tensordict and TensorDict is not None:
            return TensorDict(
                {
                    key: torch.as_tensor(value, dtype=torch.float32)
                    for key, value in batch.items()
                },
                batch_size=[batch_size, sequence_length],
            )

        return batch


class RecurrentGaussianPolicy(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_size: int, mlp_hidden_size: int
    ):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_size, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(mlp_hidden_size, action_dim)
        self.log_std_head = nn.Linear(mlp_hidden_size, action_dim)

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        lstm_out, hidden_out = self.lstm(obs_seq, hidden_state)
        features = self.mlp(lstm_out)
        mean = self.mean_head(features)
        log_std = torch.clamp(
            self.log_std_head(features), min=LOG_STD_MIN, max=LOG_STD_MAX
        )
        return mean, log_std, hidden_out

    def sample(
        self,
        obs_seq: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        mean, log_std, hidden_out = self.forward(obs_seq, hidden_state)
        if deterministic:
            action = torch.tanh(mean)
            return action, None, hidden_out

        std = log_std.exp()
        if TanhNormal is not None:
            dist = TanhNormal(loc=mean, scale=std, min=-1.0, max=1.0)
            action = dist.rsample()
            log_prob = dist.log_prob(action)
            if log_prob.ndim == action.ndim:
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            return action, log_prob, hidden_out

        dist = Normal(mean, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, hidden_out


class RecurrentQNetwork(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_size: int, mlp_hidden_size: int
    ):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim + action_dim, hidden_size, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(mlp_hidden_size, 1)

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = torch.cat([obs_seq, action_seq], dim=-1)
        lstm_out, hidden_out = self.lstm(x, hidden_state)
        features = self.mlp(lstm_out)
        q_values = self.q_head(features)
        return q_values, hidden_out


def _make_adam(params, lr: float, device: torch.device) -> torch.optim.Optimizer:
    if device.type == "cuda":
        try:
            return torch.optim.Adam(params, lr=lr, fused=True)
        except Exception:
            pass
    return torch.optim.Adam(params, lr=lr)


class TorchRLRecurrentSACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        mlp_hidden_size: int,
        learning_rate: float,
        gamma: float,
        tau: float,
        target_entropy,
        ent_coef,
        device: str,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(resolve_torch_device(device))
        configure_torch_performance(self.device)

        self.actor = RecurrentGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
        ).to(self.device)
        self.q1 = RecurrentQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
        ).to(self.device)
        self.q2 = RecurrentQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
        ).to(self.device)
        self.target_q1 = RecurrentQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
        ).to(self.device)
        self.target_q2 = RecurrentQNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
        ).to(self.device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = _make_adam(
            self.actor.parameters(), learning_rate, self.device
        )
        critic_params = list(self.q1.parameters()) + list(self.q2.parameters())
        self.critic_optimizer = _make_adam(critic_params, learning_rate, self.device)

        self.ent_coef_mode = "auto" if ent_coef == "auto" else "fixed"
        if self.ent_coef_mode == "auto":
            self.log_alpha = torch.zeros(1, device=self.device, requires_grad=True)
            self.alpha_optimizer = _make_adam(
                [self.log_alpha], learning_rate, self.device
            )
            self.fixed_alpha = None
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.fixed_alpha = float(ent_coef)

        if target_entropy == "auto":
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        self.total_updates = 0

    def get_initial_actor_state(
        self, batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        return h, c

    def _alpha_tensor(self) -> torch.Tensor:
        if self.ent_coef_mode == "auto":
            return self.log_alpha.exp()
        return torch.tensor(self.fixed_alpha, device=self.device)

    def alpha_value(self) -> float:
        return float(self._alpha_tensor().detach().cpu().item())

    def set_train_mode(self) -> None:
        self.actor.train()
        self.q1.train()
        self.q2.train()
        self.target_q1.train()
        self.target_q2.train()

    def set_eval_mode(self) -> None:
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()
        self.target_q1.eval()
        self.target_q2.eval()

    def select_action(
        self,
        normalized_obs: np.ndarray,
        actor_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
        if actor_state is None:
            actor_state = self.get_initial_actor_state(batch_size=1)

        obs_tensor = torch.as_tensor(
            normalized_obs, dtype=torch.float32, device=self.device
        ).view(1, 1, -1)
        with torch.no_grad():
            action_seq, _, next_state = self.actor.sample(
                obs_tensor, hidden_state=actor_state, deterministic=deterministic
            )

        action = action_seq[0, 0].detach().cpu().numpy().astype(np.float32)
        action = np.clip(action, -1.0, 1.0)
        return action, (next_state[0].detach(), next_state[1].detach())

    def _get_batch_tensor(
        self, batch: Union[Dict[str, np.ndarray], "TensorDict"], key: str
    ) -> torch.Tensor:
        if TensorDict is not None and isinstance(batch, TensorDict):
            value = batch.get(key)
            return value.to(self.device, non_blocking=True)
        return torch.as_tensor(batch[key], dtype=torch.float32, device=self.device)

    def update(
        self, batch: Union[Dict[str, np.ndarray], "TensorDict"]
    ) -> Dict[str, float]:
        obs = self._get_batch_tensor(batch, "obs")
        next_obs = self._get_batch_tensor(batch, "next_obs")
        actions = self._get_batch_tensor(batch, "actions")
        rewards = self._get_batch_tensor(batch, "rewards")
        dones = self._get_batch_tensor(batch, "dones")
        mask = self._get_batch_tensor(batch, "mask")

        valid_count = torch.clamp(mask.sum(), min=1.0)

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(
                next_obs, deterministic=False
            )
            target_q1, _ = self.target_q1(next_obs, next_actions)
            target_q2, _ = self.target_q2(next_obs, next_actions)
            min_target_q = (
                torch.min(target_q1, target_q2)
                - self._alpha_tensor().detach() * next_log_pi
            )
            target_q = rewards + (1.0 - dones) * self.gamma * min_target_q

        current_q1, _ = self.q1(obs, actions)
        current_q2, _ = self.q2(obs, actions)
        critic_loss = (
            (((current_q1 - target_q) ** 2) + ((current_q2 - target_q) ** 2)) * mask
        ).sum() / valid_count

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_actions, log_pi, _ = self.actor.sample(obs, deterministic=False)
        q1_pi, _ = self.q1(obs, sampled_actions)
        q2_pi, _ = self.q2(obs, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (
            (self._alpha_tensor().detach() * log_pi - min_q_pi) * mask
        ).sum() / valid_count

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss_value = 0.0
        if self.ent_coef_mode == "auto":
            alpha_loss = (
                -self.log_alpha * (log_pi + self.target_entropy).detach() * mask
            ).sum() / valid_count
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.detach().cpu().item())

        self._soft_update(self.target_q1, self.q1, self.tau)
        self._soft_update(self.target_q2, self.q2, self.tau)
        self.total_updates += 1

        mean_q = (min_q_pi.detach() * mask).sum() / valid_count
        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha_loss": alpha_loss_value,
            "alpha_value": self.alpha_value(),
            "mean_q": float(mean_q.detach().cpu().item()),
        }

    @staticmethod
    def _soft_update(target_net: nn.Module, source_net: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(
                target_net.parameters(), source_net.parameters()
            ):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * source_param.data)

    def checkpoint_dict(self, total_timesteps: int) -> Dict[str, object]:
        payload = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_size": self.hidden_size,
            "mlp_hidden_size": self.mlp_hidden_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
            "ent_coef_mode": self.ent_coef_mode,
            "fixed_alpha": self.fixed_alpha,
            "torchrl_available": torchrl_available(),
            "actor_state_dict": self.actor.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "target_q1_state_dict": self.target_q1.state_dict(),
            "target_q2_state_dict": self.target_q2.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "alpha_optimizer_state": (
                self.alpha_optimizer.state_dict()
                if self.alpha_optimizer is not None
                else None
            ),
            "log_alpha": (
                self.log_alpha.detach().cpu().numpy()
                if self.log_alpha is not None
                else None
            ),
            "total_updates": self.total_updates,
            "total_timesteps": total_timesteps,
        }
        return payload

    def save(self, model_path: str, total_timesteps: int) -> str:
        resolved_path = (
            model_path if model_path.endswith(".zip") else model_path + ".zip"
        )
        directory = os.path.dirname(resolved_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.checkpoint_dict(total_timesteps=total_timesteps), resolved_path)
        return resolved_path

    def load_from_file(
        self, model_path: str, load_optimizers: bool = True
    ) -> Dict[str, object]:
        resolved_path = resolve_model_path(model_path)
        payload = torch.load(resolved_path, map_location=self.device)
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.q1.load_state_dict(payload["q1_state_dict"])
        self.q2.load_state_dict(payload["q2_state_dict"])
        self.target_q1.load_state_dict(payload["target_q1_state_dict"])
        self.target_q2.load_state_dict(payload["target_q2_state_dict"])

        if load_optimizers:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer_state"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer_state"])
            if (
                self.ent_coef_mode == "auto"
                and payload["alpha_optimizer_state"] is not None
            ):
                self.alpha_optimizer.load_state_dict(payload["alpha_optimizer_state"])

        if self.ent_coef_mode == "auto" and payload["log_alpha"] is not None:
            self.log_alpha.data.copy_(
                torch.as_tensor(
                    payload["log_alpha"], dtype=torch.float32, device=self.device
                )
            )
        if self.ent_coef_mode == "fixed" and payload["fixed_alpha"] is not None:
            self.fixed_alpha = float(payload["fixed_alpha"])

        self.total_updates = int(payload.get("total_updates", 0))
        return payload

    @classmethod
    def load_for_evaluation(
        cls, model_path: str, device: str
    ) -> Tuple["TorchRLRecurrentSACAgent", Dict[str, object]]:
        resolved_path = resolve_model_path(model_path)
        payload = torch.load(resolved_path, map_location=resolve_torch_device(device))
        ent_coef = (
            "auto"
            if payload.get("ent_coef_mode", "auto") == "auto"
            else payload.get("fixed_alpha", 0.2)
        )
        agent = cls(
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            hidden_size=int(payload["hidden_size"]),
            mlp_hidden_size=int(payload["mlp_hidden_size"]),
            learning_rate=float(payload.get("learning_rate", 3e-4)),
            gamma=float(payload.get("gamma", 0.99)),
            tau=float(payload.get("tau", 0.005)),
            target_entropy=float(
                payload.get("target_entropy", -int(payload["action_dim"]))
            ),
            ent_coef=ent_coef,
            device=device,
        )
        agent.load_from_file(resolved_path, load_optimizers=False)
        agent.set_eval_mode()
        return agent, payload
