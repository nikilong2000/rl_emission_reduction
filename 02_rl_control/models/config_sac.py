import os

## DIRECTORIES START
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

ICE_MODEL_DIR = os.path.join(PROJECT_ROOT, "plant_lstms/tf/ICE")
PG_MODEL_DIR = os.path.join(PROJECT_ROOT, "plant_lstms/tf/Drivetrain")
ICE_MODEL_DIR_ONNX = os.path.join(PROJECT_ROOT, "plant_lstms/onnx/ICE")
PG_MODEL_DIR_ONNX = os.path.join(PROJECT_ROOT, "plant_lstms/onnx/Drivetrain")

TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "data_train")
## DIRECTORIES END
############################

## TRAINING START
# Runtime Selection
USE_ONNX = True

# Training duration
TOTAL_TIMESTEPS = 4_000_000

# SAC Hyperparameters
LEARNING_RATE = 3e-4  # Actor, Critic and entropy coefficient learning rate
BUFFER_SIZE = 1_000_000  # Replay buffer capacity
BATCH_SIZE = 256  # Mini-batch size for each gradient update
TAU = 0.005  # Soft target-network update coefficient
GAMMA = 0.99  # Discount factor
TRAIN_FREQ = 1  # Env steps between gradient updates
GRADIENT_STEPS = 1  # Gradient updates per training call
LEARNING_STARTS = 10_000  # Random-policy warm-up steps before learning

# Entropy regularization
# "auto" lets SAC tune the temperature automatically to match TARGET_ENTROPY.
# "auto" for TARGET_ENTROPY sets it to -dim(action_space) by default.
ENT_COEF = "auto"
TARGET_ENTROPY = "auto"

# State-Dependent Exploration: replaces Gaussian noise with a learnable
# noise process correlated with the state. Often improves sample efficiency
# in systems with slow dynamics (e.g. thermal time constants). Set True to enable.
USE_SDE = False
SDE_SAMPLE_FREQ = -1  # -1 = sample new noise at the start of each rollout

# Policy Configuration
# POLICY_KWARGS = dict(net_arch=[256, 256])

# Reward Weights
from config_rewards import *

## TRAINING END
