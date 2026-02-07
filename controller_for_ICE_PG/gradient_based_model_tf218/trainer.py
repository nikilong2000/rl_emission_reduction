import os
import tensorflow as tf
import matplotlib.pyplot as plt

# To configure the transition models (ICE and PG)
from transition_function_model import setup_transition_function_model

# To obtain random initial states in each epoch
from init_state import sample_init_state

# To execute a full rollout and calculate the loss
from step import rollout_and_loss

# To simulate, plot and evaluate the final model
from eval import simulate_trajectory, create_interactive_plot, eval_loss_components_fast


def train_and_save_controller(
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
    Trains a controller model using gradient-based optimisation over entire trajectories.

    This function orchestrates the entire training process, which includes:
    1.  **Robust Initialisation**: Performs a short pre-training run with multiple random seeds and selects the one that yields the lowest initial loss to start the main training.
    2.  **Main Training Loop**: In each epoch, it simulates a full trajectory (rollout) from a random initial state, calculates the total loss, and applies gradients to update the controller's weights.
    3.  **Manual Callbacks**: Implements `ReduceLROnPlateau` to decrease the learning rate if the loss stagnates and `EarlyStopping` to halt training if there's no improvement.
    4.  **Dynamic Difficulty Adjustment**: Progressively reduces the `warmup_steps` as the controller improves, forcing it to optimise the trajectory over a longer time horizon.
    5.  **Artifact Saving**: Upon completion, it saves the best model, a plot of the loss evolution, and an interactive simulation of the final trajectory.

    Parameters
    ----------
    epochs : int
        The maximum number of training epochs.
    alpha, beta, gamma : float
        Weights for the three components of the loss function: velocity error (alpha), NOx emissions (beta), and CO emissions (gamma).
    name : str
        A base name for the training run, used to create the results folder.
    ruta_ICE, ruta_PG : str
        Paths to the directories of the engine (ICE) and powertrain (PG) simulation models.
    controller : tf.keras.Model
        The neural network that acts as the controller to be trained.
    learning_rate : float, optional
        The initial learning rate for the Adam optimiser, by default 1e-4.
    output_root : str, optional
        The root directory where the results folder for this run will be created, by default "models".
    clipping : bool, optional
        If True, enables clipping of the controller's actions (likely to a [-1, 1] range), by default False.
    evalue : bool, optional
        If True, performs a final numerical evaluation over 10 trajectories to average the loss components, by default False.
    decay : bool, optional
        Parameter not used in the current implementation, by default False.
    plateau_patience : int, optional
        Number of epochs with no improvement in loss before reducing the learning rate, by default 20.
    plateau_factor : float, optional
        The factor by which the learning rate is multiplied upon reduction (e.g., 0.5 halves it), by default 0.5.
    early_stop_patience : int, optional
        Number of epochs with no improvement in loss before stopping the training prematurely, by default 50.
    restore_on_plateau : bool, optional
        If True, restores the model weights to the best-found checkpoint before reducing the learning rate, by default True.
    vel_target : float, optional
        The target velocity (in km/h) that the controller should attempt to reach, by default 70.0.
    warmup_steps : int, optional
        The initial number of simulation steps to ignore in the loss calculation. This is dynamically reduced as the agent improves, by default 200.
    good_loss : float, optional
        The loss threshold below which the agent is considered to be improving enough to start reducing `warmup_steps`, by default 10.0.

    Returns
    -------
    str
        The path to the folder where all training results were saved.
    list
        A list containing the loss value for each epoch of the training.
    """

    # --- 1) Create output folder ---
    folder = os.path.join(
        output_root,
        f"{name}_{vel_target}velTarget_{warmup_steps}warmup_{good_loss}good_loss_{epochs}ep_{alpha:.2f}a_{beta:.2f}b_{gamma:.2f}g",
    )
    os.makedirs(folder, exist_ok=True)

    # Constants TF
    a = tf.constant(alpha, dtype=tf.float32)
    b = tf.constant(beta, dtype=tf.float32)
    g = tf.constant(gamma, dtype=tf.float32)

    # ── 0) create controller if not provided externally ──
    trans_func = setup_transition_function_model(ruta_ICE, ruta_PG)
    opt = tf.keras.optimizers.Adam(learning_rate, clipnorm=0.8)

    # last_steps always synchronized with warmup_steps
    last_steps = 1200 - warmup_steps

    # === Short multi-start: initialisation pre-selection ===
    seeds = list(range(num_ini))  # 3 starts
    short_epochs = 3  # 5 express epochs
    best_init_loss = float("inf")
    best_init_weights = None

    for seed in seeds:
        tf.random.set_seed(seed)
        losses_tmp = []
        for _ in range(short_epochs):
            st = sample_init_state()
            Ls = rollout_and_loss(
                controller,
                trans_func,
                st,
                a,
                b,
                g,
                opt,
                clipping,
                vel_target=vel_target,
                warmup_steps=warmup_steps,
                last_steps=last_steps,
            )
            losses_tmp.append(Ls.numpy())
        avg = sum(losses_tmp) / short_epochs
        try:
            print(f"[Seed {seed}] short loss: {avg:.4f}")
        except:
            print(f"[Seed {seed}] short loss: {avg}")

        if avg < best_init_loss:
            best_init_loss = avg
            best_init_weights = controller.get_weights()

    # Load best seed before long training
    controller.set_weights(best_init_weights)
    print(f"⚡ Using seed with best short loss: {best_init_loss:.4f}")
    # === end short multi-start ===

    # Save original LR to reset it later
    original_lr = learning_rate

    # --- 2b) Variables for Plateau, EarlyStopping and Manual Checkpoint ---
    best_loss = float("inf")
    best_weights = None
    wait = 0
    es_wait = 0

    # To take snapshots later
    weights_start = None
    weights_mid = None

    # last_steps always synchronized with warmup_steps
    last_steps = 1200 - warmup_steps

    # --- 3) Training loop ---
    losses = []
    for ep in range(epochs):
        init_state = sample_init_state()
        L = rollout_and_loss(
            controller,
            trans_func,
            init_state,
            a,
            b,
            g,
            opt,
            clipping,
            vel_target=vel_target,
            warmup_steps=warmup_steps,
            last_steps=last_steps,
        )
        losses.append(L.numpy())
        print(f"Epoch {ep+1}/{epochs} — Mean loss: {L:.4f}")

        # —— If it improves, save new best_loss and weights
        if L < best_loss:
            best_loss = L
            best_weights = controller.get_weights()
            wait = 0
            es_wait = 0
        else:
            wait += 1
            es_wait += 1

            # — Early Stopping?
            if es_wait >= early_stop_patience:
                print(
                    f"⚠️ Early stopping: no improvement in {early_stop_patience} epochs."
                )
                break

            # — Reduce LR on Plateau?
            if wait >= plateau_patience:
                # (Optional) return to best weights before reducing LR
                if restore_on_plateau and best_weights is not None:
                    controller.set_weights(best_weights)
                    print("  ↳ Best weights restored before reducing LR.")

                old_lr = opt.learning_rate.numpy()
                new_lr = old_lr * plateau_factor
                opt.learning_rate.assign(new_lr)
                print(
                    f"  ↳ No improvement in {plateau_patience} epochs — LR {old_lr:.2e} → {new_lr:.2e}"
                )
                wait = 0

        # —— Dynamic adjustment of warmup_steps
        if L.numpy() <= good_loss and warmup_steps > 5:
            new_ws = max(25, warmup_steps - 25)
            if new_ws != warmup_steps:
                print(f"🔄 warmup_steps {warmup_steps} → {new_ws} (reset LR)")
                warmup_steps = new_ws
                last_steps = 1200 - warmup_steps

                # Reset LR and plateau counter
                opt.learning_rate.assign(original_lr)
                wait = 0
                es_wait = 0
                best_loss = float("inf")

    # Ensure the final model has the best weights before saving
    if best_weights is not None:
        controller.set_weights(best_weights)

    # --- 5) Save final model ---
    model_path = os.path.join(folder, name, ".keras")  # with .keras for keras 3
    controller.save(model_path) # before: with include_optimizer=False
    print(f"✔ Model saved at {model_path}")

    # --- 6) Loss plot ---
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss")
    plt.title(f"Training Loss ({name})")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(folder, "loss.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"✔ Loss plot saved at {plot_path}")

    # --- 7) Plots (final) ---
    df = simulate_trajectory(
        controller, trans_func, init_state, clipping, vel_target=vel_target
    )

    fig = create_interactive_plot(df, "Final Trajectory")
    html_path = os.path.join(folder, "traj_final.html")
    fig.write_html(html_path)

    if evalue:
        #         eval_loss_components(controller, trans_func,
        #                              N=10, total_steps=1200,
        #                              clipping=clipping,
        #                              out_dir=folder, vel_target= vel_target, last_steps=last_steps)

        mean_vel, mean_nox, mean_co = eval_loss_components_fast(
            controller,
            trans_func,
            N=10,
            total_steps=1200,
            last_steps=last_steps,
            clipping=clipping,
            vel_target=vel_target,
            warmup_steps=warmup_steps,
        )

        print("─── Mean loss last 200 iterations ───")
        print(f"|70-vel|: {mean_vel.numpy():.4f}")
        print(f"NOx    : {mean_nox.numpy():.4f}")
        print(f"CO     : {mean_co.numpy():.4f}")

    print(f"✔ Interactive plot saved at {html_path}")

    return folder, losses
