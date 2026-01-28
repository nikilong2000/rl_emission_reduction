import os
import glob
import tensorflow as tf
from joblib import load
from models import ScaledController
from transition_function_model import setup_transition_function_model
from eval import eval_loss_components_fast


def validate():
    print("--- Starting Validation of Best Model ---")

    # 1. Setup Environment
    print("Setting up Transition Function / Environment...")
    ruta_ICE = "../src/models_markus/ICE_Model_Update_01"
    ruta_PG = "../src/models_markus/PG_Model_M1.1_without_EM1_Torque"
    trans_func = setup_transition_function_model(ruta_ICE, ruta_PG)

    # 2. Instantiate Controller Configuration
    # We need the scaler params to initialise the architecture, even if we overwrite weights later
    print("Loading Scaler...")
    scaler = load("../src/escalados/gbm_v1.lib")
    scaler_params = {
        "data_min": scaler.data_min_,
        "data_max": scaler.data_max_,
        "scale": scaler.scale_,
        "min": scaler.min_,
    }

    # Instantiate specific model architecture
    # Note: We must match the units/alpha used in training
    controller = ScaledController(scaler_params, units=128, alpha=1000)

    # Build the model by running a dummy input or explicitly calling build
    # This ensures variables are created before loading weights
    dummy_input = tf.zeros((1, 1, 5))
    controller(dummy_input)

    # 3. Find Best/Latest Model
    models_root = "models"
    # Get all subdirectories in 'models'
    dirs = [
        d
        for d in glob.glob(os.path.join(models_root, "*"))
        if os.path.isdir(d) and ".ipynb_checkpoints" not in d
    ]

    if not dirs:
        print("No trained models found in 'models/' directory.")
        return

    # Sort by modification time (latest first)
    latest_run_dir = max(dirs, key=os.path.getmtime)
    print(f"Latest Training Run Found: {latest_run_dir}")

    # The model inside is named after the 'name' variable in main.py
    # We can try to guess it or grab the only subdirectory that looks like a model
    # Or typically it matches the first part of the run directory name

    # Heuristic: The model folder name is usually "V2.3_perf_control..."
    # Let's verify by checking subdirectories
    subdirs = [
        d for d in glob.glob(os.path.join(latest_run_dir, "*")) if os.path.isdir(d)
    ]

    model_path = None
    for d in subdirs:
        # Check if it contains saved_model.pb (TF SavedModel format)
        if os.path.exists(os.path.join(d, "saved_model.pb")):
            model_path = d
            break

    if not model_path:
        # Fallback: maybe it's saved directly in the run dir (unlikely based on trainer.py)
        if os.path.exists(os.path.join(latest_run_dir, "saved_model.pb")):
            model_path = latest_run_dir
        else:
            print(f"Could not find a 'saved_model.pb' in {latest_run_dir}")
            # Try loading weights if h5 exists?
            # For now, let's assume standard structure
            print("Listing contents of run dir:")
            print(os.listdir(latest_run_dir))
            return

    print(f"Loading Model Weights from: {model_path}")

    # Method A: Load the entire model object
    # proper_controller = tf.keras.models.load_model(model_path)

    # Method B: Load weights into our instantiated class (Safer for Custom Subclassed Models)
    # We load the weights from the saved model into our fresh instance
    loaded_model = tf.keras.models.load_model(model_path)
    controller.set_weights(loaded_model.get_weights())
    print("Weights loaded successfully.")

    # 4. Evaluation
    N_EPISODES = 50
    WARMUP = 600  # Matching the 'best' training config
    TOTAL_STEPS = 1200
    LAST_STEPS = TOTAL_STEPS - WARMUP  # Evaluate on the stable part of trajectory
    VEL_TARGET = 75.0

    print(f"\nStarting Evaluation Loop (N={N_EPISODES})...")
    print(f"Config: Target={VEL_TARGET} km/h, Warmup={WARMUP} steps")

    mean_vel, mean_nox, mean_co = eval_loss_components_fast(
        controller,
        trans_func,
        N=N_EPISODES,
        total_steps=TOTAL_STEPS,
        last_steps=LAST_STEPS,
        clipping=True,
        vel_target=VEL_TARGET,
        warmup_steps=WARMUP,
    )

    print("\n" + "=" * 40)
    print("GOLD STANDARD METRICS (Statistics)")
    print("=" * 40)
    print(f"Velocity MAE : {mean_vel.numpy():.4f}  (Target: 0.0)")
    print(f"NOx Mean     : {mean_nox.numpy():.6f}")
    print(f"CO Mean      : {mean_co.numpy():.6f}")
    print("=" * 40)

    # Save results to a simple text file in the run directory
    results_file = os.path.join(latest_run_dir, "gold_standard_metrics.txt")
    with open(results_file, "w") as f:
        f.write("GOLD STANDARD METRICS\n")
        f.write(f"Evaluated on {N_EPISODES} trajectories.\n")
        f.write(f"Velocity MAE: {mean_vel.numpy():.6f}\n")
        f.write(f"NOx Mean:     {mean_nox.numpy():.6f}\n")
        f.write(f"CO Mean:      {mean_co.numpy():.6f}\n")

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    validate()
