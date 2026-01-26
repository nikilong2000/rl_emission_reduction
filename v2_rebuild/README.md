# V2 Rebuild - Clean RL Controller

A clean, modular implementation of the TD3 reinforcement learning controller for hybrid vehicle control.

## Key Fixes

- **Observation Space**: Explicitly includes **Error** (6 dimensions) to match Agent expectations
- **Single-threaded first**: Serial training loop for easier debugging before Ray distribution

## Components

### 1. Simulation (`simulation.py`)

Wrapper for ICE/PG models with clean LSTM hidden state management.

```python
from simulation import Simulation

sim = Simulation(
    ice_model_path="path/to/ICE",
    pg_model_path="path/to/PG",
    soc_initial=0.7
)
state = sim.reset()
state = sim.step(mf=0.3, brk=0.0, ice_sp=0.5)
```

### 2. Environment (`environment.py`)

Gymnasium-compatible environment with 6D observation space.

**Observation (6D):**
| Index | Component | Description |
|-------|-----------|-------------|
| 0 | vel_target | Normalized target velocity |
| 1 | velocity | Normalized current velocity |
| 2 | mf | Previous motor front action |
| 3 | brk | Previous brake action |
| 4 | ice_sp | Previous ICE speed action |
| 5 | **error** | Normalized velocity error |

**Action (3D):** `[mf, brk, ice_sp] ∈ [-1, 1]³`

### 3. Networks (`networks.py`)

- `ActorNetwork`: LSTM-based actor with conservative initialization
- `CriticNetwork`: Twin critic for TD3

### 4. Buffer (`buffer.py`)

- `SequenceReplayBuffer`: For recurrent training with burn-in + unroll
- `SimpleReplayBuffer`: Simple flat buffer for non-recurrent comparison

### 5. Agent (`agent.py`)

TD3 agent with:

- Twin critics (reduce overestimation)
- Delayed policy updates
- Target policy smoothing
- LSTM-based networks

### 6. Training Scripts

- `train_serial.py`: Single-threaded loop (debug first!)
- `train_distributed.py`: Ray-based parallel training

## Usage

### Validation Scripts

```bash
# Validate simulation
python validate_simulation.py

# Validate environment
python validate_environment.py
```

### Training

```bash
# Single-threaded (recommended for debugging)
python train_serial.py

# Distributed with Ray
python train_distributed.py
```

## Model Paths

Default paths point to:

- ICE: `../controller_for_ICE_PG/src/models_markus/ICE_Model_Update_01`
- PG: `../controller_for_ICE_PG/src/models_markus/PG_v2`

Each model directory should contain:

- `model.h5` (Keras model)
- `input_scaler.lib` (joblib scaler)
- `output_scaler.lib` (joblib scaler)

## Requirements

- Python 3.10+
- TensorFlow 2.x
- PyTorch
- NumPy
- Gymnasium
- Ray (for distributed training)
- Matplotlib (for plotting)
