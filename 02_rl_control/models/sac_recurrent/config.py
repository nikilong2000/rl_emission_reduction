import os

## DIRECTORIES START
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

ICE_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "internal_lstm_models/NN_Application/Nets/ICE"
)
PG_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "internal_lstm_models/NN_Application/Nets/Drivetrain"
)
ICE_MODEL_DIR_ONNX = os.path.join(PROJECT_ROOT, "CTTC/Models/ICE")
PG_MODEL_DIR_ONNX = os.path.join(PROJECT_ROOT, "CTTC/Models/Drivetrain")
USE_ONNX = False
TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "02_rl_control/data")
## DIRECTORIES END
############################

## TRAINING START
# SAC Hyperparameters
LEARNING_RATE = 3e-4
BUFFER_SIZE = 200_000
BATCH_SIZE = 512
TAU = 0.005
GAMMA = 0.99

# LSTM Configuration
LSTM_CELL_SIZE = 256
MAX_SEQ_LEN = 20
FCNET_HIDDENS = [256, 256]

# Worker Configuration
# NOTE: num_env_runners=0 uses the local worker (most reliable with TF envs).
# Remote workers (>0) require TF eager mode to propagate correctly to each
# subprocess; set via --num_workers CLI flag once confirmed working.
NUM_ENV_RUNNERS = 0
ROLLOUT_FRAGMENT_LENGTH = 200

# Training Duration
TOTAL_TRAINING_ITERATIONS = 500
CHECKPOINT_FREQ = 25

# Reward Weights (matching existing SAC config)
W_SPEED = 1.0
W_EMISSION = 0.85
W_FUEL = 0.0
W_BRAKE = 0.55
W_SOC = 2.5
W_SOC_SQUARED = 0.0
W_FLICKER = 0.75
## TRAINING END
