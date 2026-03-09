"""
Recurrent SAC for Ray RLlib (old API stack).

Provides:
  RecurrentSACTorchModel  – SACTorchModel subclass with a shared LSTM
  recurrent_actor_critic_loss – loss that forwards replay-buffer states
  RecurrentSACTorchPolicy – policy class wired to the custom loss
  RecurrentSAC            – algorithm class using the custom policy
"""

import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Type, Union

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import apply_grad_clipping, huber_loss
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS

from ray.rllib.algorithms.sac.sac_torch_policy import (
    _get_dist_class,
    stats,
    postprocess_trajectory,
    validate_spaces,
    setup_late_mixins,
    build_sac_model_and_action_dist,
    concat_multi_gpu_td_errors,
    optimizer_fn,
    ComputeTDErrorMixin,
)
from ray.rllib.policy.torch_mixins import TargetNetworkMixin

from ray.rllib.algorithms.sac import SAC, SACConfig


# ────────────────────────────────────────────────────────────────────
#  Custom model
# ────────────────────────────────────────────────────────────────────

class RecurrentSACTorchModel(SACTorchModel):
    """SACTorchModel with a shared LSTM for temporal feature extraction.

    Architecture
    ────────────
    obs ──► LSTM ──► linear projection (→ obs_dim) ──► features
    features ──► policy_model  ──► action distribution
    features + actions ──► q_model ──► Q-value

    During rollout the LSTM state is carried over between steps so the
    agent can build up temporal context. During training, states from the
    replay buffer are fed in when available; otherwise the LSTM starts
    from zeros (single-step fallback).
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: ModelConfigDict,
        name: str,
        policy_model_config: ModelConfigDict = None,
        q_model_config: ModelConfigDict = None,
        twin_q: bool = False,
        initial_alpha: float = 1.0,
        target_entropy: Optional[float] = None,
    ):
        self.cell_size = model_config.get("lstm_cell_size", 256)

        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            policy_model_config=policy_model_config,
            q_model_config=q_model_config,
            twin_q=twin_q,
            initial_alpha=initial_alpha,
            target_entropy=target_entropy,
        )

        obs_dim = int(np.prod(obs_space.shape))
        self.obs_dim = obs_dim
        self.lstm = nn.LSTM(obs_dim, self.cell_size, batch_first=True)
        self.lstm_proj = nn.Linear(self.cell_size, obs_dim)

    # ── forward ──────────────────────────────────────────────────────

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs"].float()

        if not state or len(state) == 0 or seq_lens is None:
            # ── single-step / non-recurrent fallback ──
            B = obs.shape[0]
            device = obs.device
            h0 = torch.zeros(1, B, self.cell_size, device=device)
            c0 = torch.zeros(1, B, self.cell_size, device=device)
            obs_seq = obs.unsqueeze(1)  # [B, 1, D]
            lstm_out, (h, c) = self.lstm(obs_seq, (h0, c0))
            features = self.lstm_proj(lstm_out.squeeze(1))  # [B, D]
            return features, [h.squeeze(0), c.squeeze(0)]

        # ── sequence mode (training with replay sequences) ──
        h = state[0].unsqueeze(0).contiguous()
        c = state[1].unsqueeze(0).contiguous()

        obs_seq = add_time_dimension(
            obs, seq_lens=seq_lens, framework="torch", time_major=False
        )
        lstm_out, (h_new, c_new) = self.lstm(obs_seq, (h, c))
        features = self.lstm_proj(lstm_out.reshape(-1, self.cell_size))
        return features, [h_new.squeeze(0), c_new.squeeze(0)]

    # ── recurrent state ─────────────────────────────────────────────

    @override(ModelV2)
    def get_initial_state(self) -> List[TensorType]:
        # Place hidden states on same device as model parameters
        h = [
            self.lstm_proj.weight.new(1, self.cell_size).zero_().squeeze(0),
            self.lstm_proj.weight.new(1, self.cell_size).zero_().squeeze(0),
        ]
        return h


# ────────────────────────────────────────────────────────────────────
#  Recurrent-aware action distribution function
# ────────────────────────────────────────────────────────────────────

def recurrent_action_distribution_fn(
    policy: Policy,
    model: ModelV2,
    input_dict,
    *,
    state_batches: Optional[List[TensorType]] = None,
    seq_lens: Optional[TensorType] = None,
    prev_action_batch: Optional[TensorType] = None,
    prev_reward_batch=None,
    explore: Optional[bool] = None,
    timestep: Optional[int] = None,
    is_training: Optional[bool] = None,
) -> Tuple[TensorType, Type[TorchDistributionWrapper], List[TensorType]]:
    """Like SAC's action_distribution_fn but propagates LSTM states."""
    model_out, state_out = model(input_dict, state_batches or [], seq_lens)
    action_dist_inputs, _ = model.get_action_model_outputs(model_out)
    action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
    return action_dist_inputs, action_dist_class, state_out


# ────────────────────────────────────────────────────────────────────
#  Recurrent-aware SAC loss
# ────────────────────────────────────────────────────────────────────

def recurrent_actor_critic_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """SAC loss that passes replay-buffer LSTM states to the model."""

    target_model = policy.target_models[model]
    deterministic = policy.config["_deterministic_loss"]

    # ── extract recurrent state from replay batch (if present) ──
    state_in = []
    idx = 0
    while f"state_in_{idx}" in train_batch:
        state_in.append(train_batch[f"state_in_{idx}"])
        idx += 1
    seq_lens = train_batch.get(SampleBatch.SEQ_LENS)

    model_out_t, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], _is_training=True),
        state_in,
        seq_lens,
    )
    model_out_tp1, _ = model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True),
        state_in,
        seq_lens,
    )
    target_model_out_tp1, _ = target_model(
        SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], _is_training=True),
        state_in,
        seq_lens,
    )

    alpha = torch.exp(model.log_alpha)

    # ── continuous action case (this env is continuous) ──
    if model.discrete:
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        log_pis_t = F.log_softmax(action_dist_inputs_t, dim=-1)
        policy_t = torch.exp(log_pis_t)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        log_pis_tp1 = F.log_softmax(action_dist_inputs_tp1, -1)
        policy_tp1 = torch.exp(log_pis_tp1)
        q_t, _ = model.get_q_values(model_out_t)
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1)
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(model_out_t)
            twin_q_tp1, _ = target_model.get_twin_q_values(target_model_out_tp1)
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_tp1 -= alpha * log_pis_tp1
        one_hot = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), num_classes=q_t.size()[-1]
        )
        q_t_selected = torch.sum(q_t * one_hot, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
        q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
        q_tp1_best_masked = (
            1.0 - train_batch[SampleBatch.TERMINATEDS].float()
        ) * q_tp1_best
    else:
        action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
        action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
        action_dist_t = action_dist_class(action_dist_inputs_t, model)
        policy_t = (
            action_dist_t.sample()
            if not deterministic
            else action_dist_t.deterministic_sample()
        )
        log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
        action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
        action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
        policy_tp1 = (
            action_dist_tp1.sample()
            if not deterministic
            else action_dist_tp1.deterministic_sample()
        )
        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)
        q_t, _ = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
        if policy.config["twin_q"]:
            twin_q_t, _ = model.get_twin_q_values(
                model_out_t, train_batch[SampleBatch.ACTIONS]
            )
        q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
        if policy.config["twin_q"]:
            twin_q_t_det_policy, _ = model.get_twin_q_values(model_out_t, policy_t)
            q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)
        q_tp1, _ = target_model.get_q_values(target_model_out_tp1, policy_tp1)
        if policy.config["twin_q"]:
            twin_q_tp1, _ = target_model.get_twin_q_values(
                target_model_out_tp1, policy_tp1
            )
            q_tp1 = torch.min(q_tp1, twin_q_tp1)
        q_t_selected = torch.squeeze(q_t, dim=-1)
        if policy.config["twin_q"]:
            twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
        q_tp1 -= alpha * log_pis_tp1
        q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
        q_tp1_best_masked = (
            1.0 - train_batch[SampleBatch.TERMINATEDS].float()
        ) * q_tp1_best

    # Bellman target
    q_t_selected_target = (
        train_batch[SampleBatch.REWARDS]
        + (policy.config["gamma"] ** policy.config["n_step"]) * q_tp1_best_masked
    ).detach()

    base_td_error = torch.abs(q_t_selected - q_t_selected_target)
    if policy.config["twin_q"]:
        twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
        td_error = 0.5 * (base_td_error + twin_td_error)
    else:
        td_error = base_td_error

    critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))]
    if policy.config["twin_q"]:
        critic_loss.append(
            torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error))
        )

    if model.discrete:
        weighted_log_alpha_loss = policy_t.detach() * (
            -model.log_alpha * (log_pis_t + model.target_entropy).detach()
        )
        alpha_loss = torch.mean(torch.sum(weighted_log_alpha_loss, dim=-1))
        actor_loss = torch.mean(
            torch.sum(
                torch.mul(policy_t, alpha.detach() * log_pis_t - q_t.detach()),
                dim=-1,
            )
        )
    else:
        alpha_loss = -torch.mean(
            model.log_alpha * (log_pis_t + model.target_entropy).detach()
        )
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)

    model.tower_stats["q_t"] = q_t
    model.tower_stats["policy_t"] = policy_t
    model.tower_stats["log_pis_t"] = log_pis_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    model.tower_stats["alpha_loss"] = alpha_loss
    model.tower_stats["td_error"] = td_error

    return tuple([actor_loss] + critic_loss + [alpha_loss])


# ────────────────────────────────────────────────────────────────────
#  Custom policy (uses the recurrent loss)
# ────────────────────────────────────────────────────────────────────

RecurrentSACTorchPolicy = build_policy_class(
    name="RecurrentSACTorchPolicy",
    framework="torch",
    loss_fn=recurrent_actor_critic_loss,
    get_default_config=lambda: SACConfig(),
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=optimizer_fn,
    validate_spaces=validate_spaces,
    before_loss_init=setup_late_mixins,
    make_model_and_action_dist=build_sac_model_and_action_dist,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    action_distribution_fn=recurrent_action_distribution_fn,
)


# ────────────────────────────────────────────────────────────────────
#  Custom algorithm
# ────────────────────────────────────────────────────────────────────

class RecurrentSAC(SAC):
    """SAC variant that uses RecurrentSACTorchPolicy."""

    @classmethod
    @override(SAC)
    def get_default_policy_class(cls, config):
        return RecurrentSACTorchPolicy
