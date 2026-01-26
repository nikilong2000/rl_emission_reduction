"""
V2 Rebuild - Clean RL Controller Implementation

Modules:
- simulation: Wrapper for ICE/PG models with LSTM state management
- environment: Gymnasium-compatible environment (6D obs including error)
- networks: Actor and Critic networks with LSTM
- buffer: Sequence replay buffer with burn-in support
- agent: TD3 agent implementation
- train_serial: Single-threaded training loop
- train_distributed: Ray-based distributed training
"""

from .simulation import Simulation
from .environment import VehicleEnvironment
from .networks import ActorNetwork, CriticNetwork, copy_target, soft_update
from .buffer import SequenceReplayBuffer, SimpleReplayBuffer
from .agent import TD3Agent, OrnsteinUhlenbeckNoise, GaussianNoise

__all__ = [
    "Simulation",
    "VehicleEnvironment",
    "ActorNetwork",
    "CriticNetwork",
    "copy_target",
    "soft_update",
    "SequenceReplayBuffer",
    "SimpleReplayBuffer",
    "TD3Agent",
    "OrnsteinUhlenbeckNoise",
    "GaussianNoise",
]
