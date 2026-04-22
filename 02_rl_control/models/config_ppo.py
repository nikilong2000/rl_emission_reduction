import os

## DIRECTORIES START
# Base Directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Model Paths
ICE_MODEL_DIR = os.path.join(PROJECT_ROOT, "plant_lstms/tf/ICE")
PG_MODEL_DIR = os.path.join(PROJECT_ROOT, "plant_lstms/tf/Drivetrain")
ICE_MODEL_DIR_ONNX = os.path.join(PROJECT_ROOT, "plant_lstms/onnx/ICE")
PG_MODEL_DIR_ONNX = os.path.join(PROJECT_ROOT, "plant_lstms/onnx/Drivetrain")

# Data Paths
TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "data_train")

## DIRECTORIES END
############################

## TRAINING START
# Runtime Selection
USE_ONNX = True

# Training Configuration
TOTAL_TIMESTEPS = 4_000_000

# PPO Hyperparameters
LEARNING_RATE = 0.0003
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2

# Policy Configuration
# POLICY_KWARGS = dict(
#     net_arch=dict(pi=[256, 256], vf=[256, 256]),
#     ortho_init=True,
# )

# Reward Weights
from config_rewards import *

## TRAINING END
############################
