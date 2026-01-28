## LIBRARIES

import sys
import os
import tensorflow as tf
from joblib import load
import matplotlib.pyplot as plt

# Add SHARE to path to import trans
share_path = os.path.abspath("../SHARE")
if share_path not in sys.path:
    sys.path.append(share_path)

# Import the controller model from models.py
from models import ScaledController

# Import helpers from local modules (as in trainer.py)
from init_state import sample_init_state
from step import rollout_and_loss
from eval import simulate_trajectory, create_interactive_plot

# Import ONNX setup from SHARE/trans.py
try:
    from trans import setup_transition_function_model as setup_transition_function_model_onnx
except ImportError as e:
    print(f"Error importing trans: {e}")
    print("Ensure 'ONNX_Predict' is installed or available in the environment.")
    raise

# Import Differentiable Wrapper
from onnx_diff_wrapper import DiffONNXWrapper

def train_and_save_controller_onnx(
    epochs,
    alpha,
    beta,
    gamma,
    name,
    ruta_ICE,
    ruta_PG,
    controller,
    learning_rate=1e-4,
    output_root="models",
    clipping=False,
    evalue=False,
    decay=False,
    plateau_patience=20,
    plateau_factor=0.5,
    early_stop_patience=50,
    restore_on_plateau=True,
    vel_target=70.0,
    warmup_steps=200,
    good_loss=10.0,
    num_ini=3,
):
    """
    Trains a controller model using the ONNX transition function via a Numerical Gradient Wrapper.
    """

    # --- 1) Create output folder ---
    folder = os.path.join(
        output_root,
        f"{name}_{vel_target}velTarget_{warmup_steps}warmup_{good_loss}good_loss_{epochs}ep_{alpha:.2f}a_{beta:.2f}b_{gamma:.2f}g",
    )
    os.makedirs(folder, exist_ok=True)
    print(f"📂 Output folder created: {folder}")

    # --- 2) Setup ONNX Transition Function ---
    print("⚙️ Setting up ONNX transition function...")
    raw_trans_func = setup_transition_function_model_onnx(ruta_ICE, ruta_PG)
    
    # WRAP for Gradients
    print("🎁 Wrapping ONNX model with Numerical Gradient Estimator...")
    trans_func = DiffONNXWrapper(raw_trans_func, epsilon=0.1) 
    # epsilon=0.1 chosen for robustness with float32 ONNX models

    # --- 3) Optimiser ---
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # --- 4) Robust Initialization ---
    print(f"🚀 Robust initialization ({num_ini} tries)...")
    best_loss_ini = float("inf")
    best_weights_ini = None

    for i in range(num_ini):
        # Reset controller weights
        cloned_controller = tf.keras.models.clone_model(controller)
        cloned_controller.build((1, 1, 5)) 
        
        # Sample initial state
        init_state_val = sample_init_state()
        init_state = tf.constant(init_state_val, dtype=tf.float32)

        # One rollout to check loss
        try:
            # Note: rollout_and_loss is decorated with @tf.function in step.py.
            # It might fail with ONNX models if they strictly return numpy/run non-TF ops.
            # If it fails, you may need a non-compiled version of rollout_and_loss.
            loss_init = rollout_and_loss(
                cloned_controller,
                trans_func,
                init_state,
                tf.constant(alpha, dtype=tf.float32),
                tf.constant(beta, dtype=tf.float32),
                tf.constant(gamma, dtype=tf.float32),
                opt,
                clipping=clipping,
                total_steps=1200,
                last_steps=200,
                warmup_steps=warmup_steps,
                vel_target=vel_target,
            )
            print(f"   Try {i+1}: Init Loss = {loss_init.numpy():.4f}")

            if loss_init < best_loss_ini:
                best_loss_ini = loss_init
                best_weights_ini = cloned_controller.get_weights()
        except Exception as e:
            print(f"   Try {i+1}: Failed with error: {e}")

    if best_weights_ini is not None:
        controller.set_weights(best_weights_ini)
        print("✅ Best initialization selected.")
    else:
        print("⚠️ Initialization failed or no valid loss returned. Proceeding with current weights.")

    # --- 5) Training Loop ---
    losses = []
    best_loss = float("inf")
    wait_plateau = 0
    wait_early_stop = 0
    
    # Warmup reduction logic
    original_warmup = warmup_steps
    
    print("🔥 Starting training...")
    for epoch in range(epochs):
        # Dynamic warmup reduction
        if best_loss < good_loss:
            # Linearly reduce warmup based on how good the loss is (heuristic)
            # Example: As loss goes from good_loss down to 0, reduce warmup
             ratio = max(0.0, best_loss / good_loss) 
             # If loss is very low, we want low warmup? 
             # Actually the original code reduces warmup (increases difficulty) as agent gets better.
             # If agent is good (low loss), we want warmup to be smaller (eval on longer horizon? or count more steps?)
             # In step.py: loss is calculated from 'warmup_steps' to end.
             # So reducing warmup_steps means optimizing over a longer horizon (harder).
             warmup_steps = int(original_warmup * ratio)
             warmup_steps = max(10, warmup_steps) # Minimum 10 steps

        # Prepare epoch
        init_state_val = sample_init_state()
        init_state = tf.constant(init_state_val, dtype=tf.float32)
        
        # Train step
        # Note: Gradients will only flow if trans_func is differentiable.
        current_loss = rollout_and_loss(
            controller,
            trans_func,
            init_state,
            tf.constant(alpha, dtype=tf.float32),
            tf.constant(beta, dtype=tf.float32),
            tf.constant(gamma, dtype=tf.float32),
            opt,
            clipping=clipping,
            total_steps=1200,
            last_steps=200,
            warmup_steps=warmup_steps,
            vel_target=vel_target
        )
        
        loss_val = current_loss.numpy()
        losses.append(loss_val)

        # Logging
        if epoch % 10 == 0:
            print(f"Ep {epoch}/{epochs} | Loss: {loss_val:.4f} | Warmup: {warmup_steps} | InitVel: {init_state_val[0,3]:.1f}")
            
        # Callbacks
        if loss_val < best_loss:
            best_loss = loss_val
            wait_plateau = 0
            wait_early_stop = 0
            # Save best model
            controller.save(os.path.join(folder, "best_controller.h5"))
        else:
            wait_plateau += 1
            wait_early_stop += 1
            
        # Reduce LR
        if wait_plateau >= plateau_patience:
            old_lr = float(opt.lr.numpy())
            new_lr = old_lr * plateau_factor
            opt.lr.assign(new_lr)
            print(f"📉 Reduce LR: {old_lr:.2e} -> {new_lr:.2e}")
            wait_plateau = 0
            if restore_on_plateau:
                 try:
                     controller.load_weights(os.path.join(folder, "best_controller.h5"))
                     print("   Restored best weights.")
                 except:
                     pass

        # Early Stop
        if wait_early_stop >= early_stop_patience:
            print("🛑 Early stopping triggered.")
            break

    # --- 6) Save Final Artifacts ---
    print("💾 Saving final results...")
    
    # Save Losses
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.yscale('log')
    plt.savefig(os.path.join(folder, "loss_history.png"))
    plt.close()
    
    # Save Model
    controller.save(os.path.join(folder, "final_controller.h5"))
    
    # Evaluation Plot (Interactive)
    # Using the last state
    try:
        ts, vs, mfs, brks, ice_sps, trqs, noxs, cos, socs = simulate_trajectory(
            controller, trans_func, init_state, total_steps=1200, clipping=clipping, vel_target=vel_target
        )
        create_interactive_plot(ts, vs, mfs, brks, ice_sps, trqs, noxs, cos, socs, folder)
    except Exception as e:
        print(f"Warning: Could not create final plot: {e}")

    return folder, losses


## MAIN EXECUTION
if __name__ == "__main__":
    # Load Scaler for Controller
    # (Assuming the scaler path is correct relative to this file)
    scaler_path = "../src/escalados/gbm_v1.lib"
    if not os.path.exists(scaler_path):
        print(f"Scaler not found at {scaler_path}")
        # dummy values to avoid crash if checking code
        scaler_params = {
            "data_min": [0]*6, "data_max": [1]*6, "scale": [1]*6, "min": [0]*6
        }
    else:
        scaler = load(scaler_path)
        scaler_params = {
            "data_min": scaler.data_min_,
            "data_max": scaler.data_max_,
            "scale": scaler.scale_,
            "min": scaler.min_,
        }

    # Initialize Controller
    controller = ScaledController(scaler_params, units=128, alpha=1000)

    # Parameters
    epochs = 10000
    NOx_mean = 0.0001663531
    CO_mean = 0.0013611736
    vel_norm = 1.0

    alpha = 1.0 / vel_norm
    beta = 0.0 / NOx_mean
    gamma = 0.0 / CO_mean

    name = "V2.3_ONNX_TEST"

    # Execute training
    # Changes: passing ONNX paths
    out_dir, losses = train_and_save_controller_onnx(
        epochs,
        alpha,
        beta,
        gamma,
        name,
        ruta_ICE="../SHARE/CTTC_models/ONNX/ICE",
        ruta_PG="../SHARE/CTTC_models/ONNX/PG",
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
