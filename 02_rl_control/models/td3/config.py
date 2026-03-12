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
TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, "02_rl_control/data_train")
## DIRECTORIES END
############################

## TRAINING START
# TD3 Hyperparameters
LEARNING_RATE = 1e-3  # Actor and Critic learning rate
BUFFER_SIZE = 200_000  # Replay buffer capacity
BATCH_SIZE = 256  # Mini-batch size for each gradient update
TAU = 0.005  # Soft target-network update coefficient
GAMMA = 0.99  # Discount factor
TRAIN_FREQ = 1  # Env steps between gradient updates
GRADIENT_STEPS = 1  # Gradient updates per training call
LEARNING_STARTS = 10_000  # Random-policy warm-up steps before learning
POLICY_DELAY = 2  # Actor updated every POLICY_DELAY critic updates
TARGET_POLICY_NOISE = 0.2  # Std of noise added to target-policy actions
TARGET_NOISE_CLIP = 0.5  # Clip bound for target-policy noise

# Exploration noise applied to actor output during training
ACTION_NOISE_SIGMA = 0.1

# Training duration
TOTAL_TIMESTEPS = 1_000_000

# Reward Weights
W_SPEED = 1.0
W_EMISSION = 0.85
W_FUEL = 0.0
W_BRAKE = 0.5
W_SOC = 2.0
W_SOC_SQUARED = 0.0
W_FLICKER = 0.5
## TRAINING END
