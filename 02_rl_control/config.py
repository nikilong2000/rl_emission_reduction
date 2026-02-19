import os

# Base Directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Model Paths
# Adjust these paths if the internal_lstm_models structure is different
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

# RL Environment Config
STATE_DIM = 6  # Car_Speed, Target_Speed, SOC, ICE_Torque, NOx, CO
ACTION_DIM = 4  # ICE_Speed, EM2_Torque, Fuel_Mass, Brake

# Normalization / Scaling Limits (Approximate - need to be tuned or loaded from scalers)
# These are used for observation space definition
MAX_SPEED_KMPH = 160.0
MAX_TORQUE_NM = 500.0  # Placeholder
MAX_EMISSIONS = 10.0  # Placeholder (g/s)

# Reward Weights
W_SPEED = 1.0
W_EMISSION = 1.0
W_FUEL = 0.05
