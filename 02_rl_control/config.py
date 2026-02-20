import os

## DIRECTORIES
# Base Directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Model Paths
ICE_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "internal_lstm_models/NN_Application/Nets/ICE"
)
PG_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "internal_lstm_models/NN_Application/Nets/Drivetrain"
)

# Data Paths
INPUT_DATA_PATH = os.path.join(
    PROJECT_ROOT, "internal_lstm_models/NN_Application/Input_data/WLTC.csv"
)


## TRAINING
# PPO Hyperparameters
LEARNING_RATE = 0.0003
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

# Training Configuration
TOTAL_TIMESTEPS = 400000

# Reward Weights
W_SPEED = 1.0
W_EMISSION = 0.85
W_FUEL = 0.75
W_BRAKE = 0.25


############################

# TODO: Not used for now
# # RL Environment Config
# STATE_DIM = 6  # Car_Speed, Target_Speed, SOC, ICE_Torque, NOx, CO
# ACTION_DIM = 4  # ICE_Speed, EM2_Torque, Fuel_Mass, Brake

# TODO: Not used for now
# # Normalization / Scaling Limits (Approximate - need to be tuned or loaded from scalers)
# # These are used for observation space definition
# MAX_SPEED_KMPH = 160.0
# MAX_TORQUE_NM = 500.0  # Placeholder
# MAX_EMISSIONS = 10.0  # Placeholder (g/s)
