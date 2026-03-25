import os

## DIRECTORIES START
# Base Directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

# Model Paths
ICE_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "internal_lstm_models/NN_Application/Nets/ICE"
)
PG_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "internal_lstm_models/NN_Application/Nets/Drivetrain"
)
ICE_MODEL_DIR_ONNX = os.path.join(PROJECT_ROOT, "CTTC/Models/ICE")
PG_MODEL_DIR_ONNX = os.path.join(PROJECT_ROOT, "CTTC/Models/Drivetrain")

# Data Paths
TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "02_rl_control/data_train")

## DIRECTORIES END
############################

## TRAINING START
# Runtime Selection
USE_ONNX = False

# Training Configuration
TOTAL_TIMESTEPS = 800_000

# PPO Hyperparameters
LEARNING_RATE = 0.0003
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

# Reward Weights
W_SPEED = 1.0
W_EMISSION = 0.0
W_FUEL = 0.0
W_BRAKE = 0.0
W_SOC = 0.0
W_SOC_SQUARED = 0.0
W_FLICKER = 0.0

## TRAINING END
############################
