## LIBRARIES

import tensorflow as tf
from joblib import load
import sys
import os

# Add SHARE to path to find ONNX_Predict if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '../SHARE'))

# Import the controller model from models.py
from models import ScaledController

# Import the main training function from trainer.py
from trainer import train_and_save_controller


## MODELS
# Note: Ensure these paths are correct relative to this new directory
scaler_path = "../src/escalados/gbm_v1.lib"
if not os.path.exists(scaler_path):
    print(f"Warning: Scaler not found at {scaler_path}. Please check path.")
    # Attempt to locate it if path is slightly different
    scaler_path_alt = "../../src/escalados/gbm_v1.lib"
    if os.path.exists(scaler_path_alt):
         scaler_path = scaler_path_alt

if os.path.exists(scaler_path):
    scaler = load(scaler_path)
    scaler_params = {
        "data_min": scaler.data_min_,  # 6 values
        "data_max": scaler.data_max_,  # 6 values
        "scale": scaler.scale_,  # 6 values  (1/(max-min) or 2/(max-min))
        "min": scaler.min_,  # 6 values  (offset)
    }

    controller = ScaledController(scaler_params, units=128, alpha=1000)
else:
    print("CRITICAL: Scaler file not found. Controller cannot be initialized.")
    controller = None


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


name = "V2.3_perf_control_5_values_clipping_mse_complexlr3_warm_reductlr_best_ONNX"


# UPDATED PATHS FOR ONNX MODELS
ruta_ICE_onnx = "../SHARE/CTTC_models/ONNX/ICE"
ruta_PG_onnx = "../SHARE/CTTC_models/ONNX/PG"

print(f"Using ICE ONNX path: {ruta_ICE_onnx}")
print(f"Using PG ONNX path: {ruta_PG_onnx}")

# Execute training and saving
# WARNING: Gradient-based training via ONNX models is NOT supported due to lack of gradients.
# This script will run the structure but gradients will be zero/null.
# Use this for evaluation or inference testing.

if controller is not None:
    out_dir, losses = train_and_save_controller(
        epochs,
        alpha,
        beta,
        gamma,
        name,
        ruta_ICE=ruta_ICE_onnx,
        ruta_PG=ruta_PG_onnx,
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
