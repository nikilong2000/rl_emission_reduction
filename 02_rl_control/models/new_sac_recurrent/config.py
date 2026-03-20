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
TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "02_rl_control/data")
## DIRECTORIES END
############################

## TRAINING START
# Recurrent SAC Hyperparameters
LEARNING_RATE = 3e-4
BUFFER_SIZE = 400_000
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
LEARNING_STARTS = 10_000

# Entropy regularization
ENT_COEF = "auto"
TARGET_ENTROPY = "auto"

# Recurrent network architecture
HIDDEN_SIZE = 256
MLP_HIDDEN_SIZE = 256
SEQUENCE_LENGTH = 20

# Observation normalization
CLIP_OBS = 10.0

# Training duration
TOTAL_TIMESTEPS = 1_200_000

# Artifact cadence (mirrors SB3 callback defaults used in other agents)
CHECKPOINT_FREQ = 100_000
PLOT_FREQ = 1_000
ENTROPY_PLOT_FREQ = 10

# Reward Weights
W_SPEED = 1.0
W_EMISSION = 0.85
W_FUEL = 0.0
W_BRAKE = 0.55
W_SOC = 2.5
W_SOC_SQUARED = 0.0
W_FLICKER = 0.75
## TRAINING END
