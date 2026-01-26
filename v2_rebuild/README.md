# V2 Rebuild - Clean RL Controller Implementation

This directory contains a clean, modular implementation of the RL controller following the plan in `plan-rebuildRlController.prompt.md`.

## Structure

```
v2_rebuild/
├── README.md                    # This file
├── simulation.py                # Simulation wrapper for ICE/PG models
├── environment.py               # Gymnasium-style environment (WITH ERROR FIX)
├── networks.py                  # Actor and RecurrentCritic networks
├── buffer.py                    # Sequence replay buffer
├── agent.py                     # TD3 agent implementation
├── train_serial.py              # Single-threaded training loop
├── validate_simulation.py       # Simulation validation script
└── validate_environment.py      # Environment validation script
```

## Key Features

### 1. **Observation Space Fix** ✅
The environment now includes the **Error** as the 6th dimension in observations:
- `[vel_target, vel, mf, brk, ice_sp, error]` (6 dimensions)
- This fixes the input mismatch bug where the Agent expected 6 dims but environment provided 5

### 2. **Clean Modular Design**
- **simulation.py**: Wrapper around transition function model (ICE + PG)
- **environment.py**: Gymnasium-compatible environment with proper `reset()` and `step()`
- **networks.py**: PyTorch LSTM-based Actor and Critic
- **buffer.py**: Sequence replay buffer for Burn-in + Unroll training
- **agent.py**: TD3 algorithm with twin critics, delayed updates, and target smoothing

### 3. **Single-Threaded First**
- `train_serial.py` implements a simple training loop in a single process
- Easier debugging before adding Ray complexity
- Full training pipeline: Environment → Agent → Buffer → Learn

## Components

### Simulation (simulation.py)
- Wraps ICE and PG LSTM models
- Manages hidden states cleanly
- Provides `predict_ice()`, `predict_pg()`, and `step()` methods
- `reset_states()` to reset LSTM hidden states

### Environment (environment.py)
- **Fixed observation space**: 6 dimensions including error
- Gymnasium-compatible interface
- Configurable reward function (normalized squared error)
- Episode termination based on stability threshold
- Proper handling of ICE constraints (min speed, torque clipping)

### Networks (networks.py)
- **Actor**: LSTM → LayerNorm → FC → Tanh (outputs actions in [-1, 1])
- **RecurrentCritic**: LSTM → LayerNorm → Q-network
- Conservative initialization to prevent saturation
- Optional input normalization via scaler parameters
- Hidden state management for inference mode

### Buffer (buffer.py)
- **SequenceReplayBuffer**: Stores windows of trajectories
- **EpisodeBuffer**: Collects full episodes before extracting windows
- Supports overlapping windows with configurable stride
- Shape validation to catch bugs early

### Agent (agent.py)
- **TD3 Algorithm**:
  - Twin critics to reduce overestimation
  - Delayed policy updates
  - Target policy smoothing
  - Exploration noise
- Supports sequence-based training
- Save/load functionality

### Training (train_serial.py)
- Single-threaded training loop
- Random exploration phase
- Configurable hyperparameters
- Logging and checkpointing
- Episode → Buffer → Sample → Train pipeline

## Usage

### Prerequisites
```bash
pip install torch numpy
```

### Training (after Keras compatibility fix)
```python
from train_serial import train_serial

agent, rewards = train_serial(
    ice_model_dir="/path/to/ICE",
    pg_model_dir="/path/to/PG",
    data_dir="/path/to/data",
    num_episodes=1000
)
```

### Testing Environment
```python
from environment import VehicleControlEnvironment

env = VehicleControlEnvironment(
    ice_model_dir="/path/to/ICE",
    pg_model_dir="/path/to/PG",
    data_dir="/path/to/data"
)

obs = env.reset(vel_target=70.0)  # Returns 6-dim observation
print(f"Observation shape: {obs.shape}")  # Should be (6,)
print(f"Error: {obs[5]}")  # 6th element is vel_target - vel
```

## Known Issues

### Keras Compatibility
The existing .h5 models were created with TensorFlow 2.x / Keras 2.x using the `time_major` parameter for LSTM layers. This parameter has been removed in Keras 3.x (required by TensorFlow 2.18+).

**Solutions**:
1. Retrain models with Keras 3.x
2. Convert models to compatible format
3. Use TensorFlow 2.15 + Keras 2.15 environment
4. Export models to ONNX format

**Current Status**: 
- All code interfaces are correct and ready
- Validation scripts will work once Keras compatibility is resolved
- Training pipeline is complete and can run independently

## Next Steps (Future Work)

### Step 5: Distributed Training with Ray
- Factor code into `Learner` (GPU) and `RolloutWorker` (CPU) classes
- Implement `train_distributed.py` for parallel rollouts
- Use Ray for distributed execution
- Maintain the same interfaces established in single-threaded version

### Improvements
- Add tensorboard logging
- Implement prioritized experience replay
- Add gradient clipping
- Implement learning rate scheduling
- Add more sophisticated exploration strategies

## Testing

Once Keras compatibility is fixed, run validation scripts:

```bash
# Test simulation
cd v2_rebuild
python validate_simulation.py

# Test environment
python validate_environment.py

# Run training
python train_serial.py
```

## Key Differences from Original

1. **Observation Space**: Now 6 dimensions (added Error) instead of 5
2. **Modular Structure**: Clear separation of concerns
3. **PyTorch Networks**: Clean LSTM implementation with proper state management
4. **Sequence Buffer**: Proper window extraction for recurrent training
5. **Single-Threaded First**: Validate before adding Ray complexity
6. **Type Hints**: Better code documentation and IDE support
7. **Validation Scripts**: Test each component independently

## Summary

This v2_rebuild provides a clean, well-structured implementation of the RL controller that:
- ✅ Fixes the observation space mismatch (6 dims with Error)
- ✅ Follows Gymnasium conventions
- ✅ Implements TD3 with recurrent networks
- ✅ Starts simple (single-threaded) before adding complexity
- ✅ Has validation scripts for each component
- ⏸️ Waiting for Keras compatibility fix to run end-to-end
