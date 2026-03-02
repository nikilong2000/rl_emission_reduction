import os

## DIRECTORIES START
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

## DIRECTORIES END
############################

## TRAINING START
# PPO Hyperparameters
LEARNING_RATE = 0.0003
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

# Training Configuration
# TOTAL_TIMESTEPS = 1600000
TOTAL_TIMESTEPS = 400000

# Reward Weights
W_SPEED = 1.0
W_EMISSION = 0.85
W_FUEL = 0.0
W_BRAKE = 0.5
W_SOC = 10.0
W_SOC_SQUARED = 0.0
W_FLICKER = 0.5

## TRAINING END
############################

## THERMAL OBSERVATION CONFIG START
# ICE LSTM output indices for the 3 thermal variables (from PCA: 95% variance)
# Based on config.txt output order:
#   idx 5  = T_gas_eo_K      (engine-out gas temperature)
#   idx 12 = T_Sub_DPF_K     (DPF substrate temperature)
#   idx 15 = T_gas_tp_K      (tailpipe gas temperature)
THERMAL_OBS_INDICES = [5, 12, 15]
THERMAL_OBS_NAMES = ["T_gas_eo_K", "T_Sub_DPF_K", "T_gas_tp_K"]

# Observation space bounds for thermal variables (K)
# Based on training data distributions (see thermal_analysis_results/)
THERMAL_OBS_LOW = [250.0, 250.0, 250.0]
THERMAL_OBS_HIGH = [900.0, 900.0, 900.0]

# Initial temperature at ambient (cold start)
THERMAL_INIT_K = 298.0

# SCR catalyst light-off temperature (K) — below this, SCR conversion is poor
# ~250°C = 523K is a typical diesel SCR light-off temperature
SCR_LIGHTOFF_K = 523.0

# Multiplier for NOx penalty when SCR is cold (below light-off)
# The base emission penalty is multiplied by this factor when T_gas_eo_K < SCR_LIGHTOFF_K
W_COLD_NOX_MULTIPLIER = 1.5

## THERMAL OBSERVATION CONFIG END
############################

## TODO: Not used for now
# # RL Environment Config
# STATE_DIM = 6  # Car_Speed, Target_Speed, SOC, ICE_Torque, NOx, CO
# ACTION_DIM = 4  # ICE_Speed, EM2_Torque, Fuel_Mass, Brake

# TODO: Not used for now
# # Normalization / Scaling Limits (Approximate - need to be tuned or loaded from scalers)
# # These are used for observation space definition
# MAX_SPEED_KMPH = 160.0
# MAX_TORQUE_NM = 500.0  # Placeholder
# MAX_EMISSIONS = 10.0  # Placeholder (g/s)
