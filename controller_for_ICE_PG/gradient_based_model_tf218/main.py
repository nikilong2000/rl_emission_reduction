## LIBRARIES

import tensorflow as tf
from joblib import load

# Import the controller model from models.py
from models import ScaledController

# Import the main training function from trainer.py
from trainer import train_and_save_controller


## MODELS
scaler = load("../src/escalados/gbm_v1.lib")
scaler_params = {
    "data_min": scaler.data_min_,  # 6 values
    "data_max": scaler.data_max_,  # 6 values
    "scale": scaler.scale_,  # 6 values  (1/(max-min) or 2/(max-min))
    "min": scaler.min_,  # 6 values  (offset)
}

controller = ScaledController(scaler_params, units=128, alpha=1000)


# Example parameters
epochs = 10000
alpha, beta, gamma = 1.0, 0, 0

NOx_mean = tf.constant(0.0001663531, dtype=tf.float32)
CO_mean = tf.constant(0.0013611736, dtype=tf.float32)
# For the speed term we divide by 1.0 (or change to something more realistic if you have standard deviation)
vel_norm = tf.constant(1.0, dtype=tf.float32)

alpha = alpha / vel_norm
beta = beta / NOx_mean
gamma = gamma / CO_mean


name = "V2.3_perf_control_5_values_clipping_mse_complexlr3_warm_reductlr_best"


# Execute training and saving
out_dir, losses = train_and_save_controller(
    epochs,
    alpha,
    beta,
    gamma,
    name,
    ruta_ICE="../../internal_lstm_models/Final_Neural_Networks/ICE/",
    ruta_PG="../../internal_lstm_models/Final_Neural_Networks/Drivetrain/",
    controller=controller,
    learning_rate=1e-4,
    output_root="models",
    clipping=True,
    evalue=True,
    vel_target=75.0,
    plateau_patience=10,
    early_stop_patience=24,
    warmup_steps=600,
    good_loss=0.05,
)


print("All ready in:", out_dir)
