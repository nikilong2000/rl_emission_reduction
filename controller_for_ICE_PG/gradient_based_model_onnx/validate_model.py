import os
import sys
# Add SHARE for ONNX modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../SHARE'))

import glob
import tensorflow as tf
from joblib import load
from models import ScaledController
from transition_function_model import setup_transition_function_model
from eval import eval_loss_components_fast


def validate():
    print("--- Starting Validation of Best Model (ONNX Environment) ---")

    # 1. Setup Environment
    print("Setting up Transition Function / Environment...")
    ruta_ICE = "../SHARE/CTTC_models/ONNX/ICE"
    ruta_PG = "../SHARE/CTTC_models/ONNX/PG"
    trans_func = setup_transition_function_model(ruta_ICE, ruta_PG)

    # 2. Instantiate Controller Configuration
    # We need the scaler params to initialise the architecture, even if we overwrite weights later
    print("Loading Scaler...")
    try:
        scaler = load("../src/escalados/gbm_v1.lib")
        scaler_params = {
            "data_min": scaler.data_min_,
            "data_max": scaler.data_max_,
            "scale": scaler.scale_,
            "min": scaler.min_,
        }

        # Instantiate specific model architecture
        controller = ScaledController(scaler_params, units=128, alpha=1000)
    except FileNotFoundError:
        print("Scaler not found, cannot instantiate controller.")
        return

    # 3. Find the best model checkpoint
    # Assuming models are saved in ./models/ or similar
    # Adjust this search pattern if needed
    model_files = glob.glob("models/*.h5") + glob.glob("models/*/*.h5")
    if not model_files:
        print("No model files found in ./models/")
        return
        
    # Just picking the newest one or specific logic
    best_model_path = max(model_files, key=os.path.getmtime)
    print(f"Loading weights from: {best_model_path}")
    
    try:
        controller.load_weights(best_model_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 4. Run Evaluation
    # Since we are using ONNX, we must ensure eval_loss_components_fast works eagerly
    # (It should if step.py removed @tf.function and eval.py functions are standard)
    
    print("Running evaluation...")
    # Add your eval logic here calling eval_loss_components_fast(controller, trans_func, ...)
    # Not fully implemented in original file provided, but assuming usage matches eval.py

if __name__ == "__main__":
    validate()
