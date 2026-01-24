import tensorflow as tf
import pandas as pd
import plotly.express as px

# Required to obtain random initial states in the evaluation
from init_state import sample_init_state


def simulate_trajectory(
    controller, trans_func, init_state, clipping, K=1200, vel_target=70.0
):
    """
    Simulates a K-step trajectory without training.

    This function runs the controller in a simulated environment for K steps
    and returns a pandas DataFrame containing the history of all state and
    output variables. It also calculates and prints the mean loss over the
    trajectory, excluding the initial `warmup_steps`.

    Parameters
    ----------
    controller : tf.keras.Model
        The trained controller model to be evaluated.
    trans_func : object
        The transition function model that simulates the environment.
    init_state : tf.Tensor
        The initial state for the simulation.
    clipping : bool
        If True, clips the predicted torque to a predefined range.
    K : int, optional
        The total number of steps in the trajectory, by default 1200.
    vel_target : float, optional
        The target velocity for the loss calculation, by default 70.0.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the time series data for all simulated variables.
    """
    tf.py_function(trans_func.reset_models, [], [])
    tf.py_function(controller.reset_states, [], [])

    data = {k: [] for k in ("mf", "brk", "ice_sp", "torque", "vel_out", "nox", "co")}

    # Constant ambient conditions
    p_amb_bar = tf.constant(1, dtype=tf.float32)
    T_amb_K = tf.constant(298, dtype=tf.float32)

    # Convert init_state to individual tensors
    _, mf, brk, ice_sp = tf.unstack(init_state)
    torque, _, _, _, _ = trans_func.predict_ice(ice_sp, mf, T_amb_K, p_amb_bar)

    if clipping:
        torque = tf.clip_by_value(torque, -50, 300.0)

    mask_ice = ice_sp < 900.0  # boolean condition
    torque = tf.where(mask_ice, tf.zeros_like(torque), torque)
    mf = tf.where(mask_ice, tf.zeros_like(mf), mf)

    vel, _ = trans_func.predict_PG(ice_sp, 0.0, torque, brk)

    for _ in range(K):
        # Pack controller input
        x = tf.stack([vel_target, vel, mf, brk, ice_sp])
        inp = tf.reshape(x, (1, 1, 5))  # (1,1,5)

        out = controller(inp, training=False)[0]  # shape (4,)
        mf, brk, ice_sp = tf.unstack(out, num=3)  # now it is a TF op

        torque, nox, _, co, _ = trans_func.predict_ice(ice_sp, mf, T_amb_K, p_amb_bar)

        if clipping:
            torque = tf.clip_by_value(torque, -50, 300.0)

        mask_ice = ice_sp < 900.0  # boolean condition
        torque = tf.where(mask_ice, tf.zeros_like(torque), torque)

        vel_out, _ = trans_func.predict_PG(ice_sp, 0.0, torque, brk)

        vel = vel_out
        data["torque"].append(torque.numpy())  # Save torque at each step
        data["mf"].append(mf.numpy())
        data["brk"].append(brk.numpy())
        data["ice_sp"].append(ice_sp.numpy())
        data["vel_out"].append(vel_out.numpy())
        data["nox"].append(nox.numpy())
        data["co"].append(co.numpy())

    df = pd.DataFrame(data)
    df["step"] = df.index
    return df


def create_interactive_plot(df, title):
    fig = px.line(
        df,
        x="step",
        y=["mf", "brk", "ice_sp", "torque", "vel_out", "nox", "co"],
        title=title,
    )
    # Hide outputs
    for tr in fig.data:
        if tr.name in ("vel_out", "nox", "co"):
            tr.visible = "legendonly"
    buttons = [
        dict(
            label="States",
            method="update",
            args=[
                {
                    "visible": [
                        t.name in ("mf", "brk", "ice_sp", "torque") for t in fig.data
                    ]
                },
                {"title": f"{title} — States"},
            ],
        ),
        dict(
            label="Outputs",
            method="update",
            args=[
                {"visible": [t.name in ("vel_out", "nox", "co") for t in fig.data]},
                {"title": f"{title} — Outputs"},
            ],
        ),
    ]
    fig.update_layout(
        updatemenus=[dict(type="buttons", buttons=buttons, direction="right")]
    )
    return fig


import tensorflow as tf
import numpy as np
import os


###############################################################################
# eval_loss_components  —  optimized version (≈10-15× faster on GPU)
###############################################################################
@tf.function()  # ← XLA optional but speeds up significantly
def eval_loss_components_fast(
    controller,
    trans_func,
    N=20,
    total_steps=1200,
    last_steps=200,
    clipping=True,
    vel_target=75.0,
    warmup_steps=200,
):
    """
    Evaluates N trajectories WITHOUT training and returns the average loss components.

    This function is compiled with `@tf.function` for high performance. It runs
    multiple simulations with random initial states and calculates the average
    absolute velocity error, average NOx, and average CO.

    - Loss accumulation only starts after `warmup_steps`.
    - The final average is calculated over the (`total_steps` - `warmup_steps`) window.

    Parameters
    ----------
    controller : tf.keras.Model
        The controller model to evaluate.
    trans_func : object
        The transition function model that simulates the environment.
    N : int, optional
        The number of trajectories to simulate and average over, by default 20.
    total_steps : int, optional
        The total number of steps for each trajectory, by default 1200.
    last_steps : int, optional
        An unused parameter, by default 200. The loss window is controlled by `warmup_steps`.
    clipping : bool, optional
        If True, clips the predicted torque value, by default True.
    vel_target : float, optional
        The target velocity for the loss calculation, by default 75.0.
    warmup_steps : int, optional
        The number of initial steps to ignore in the loss calculation, by default 200.

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor, tf.Tensor]
        A tuple containing the average loss values for:
        (mean absolute velocity error, mean NOx, mean CO).
    """
    # --------------------------------- Constants ---------------------------------
    p_amb_bar = tf.constant(1.0, tf.float32)
    T_amb_K = tf.constant(298.0, tf.float32)
    threshold = tf.constant(total_steps - last_steps, tf.int32)

    # ---------- Global accumulators (only one scalar per metric) ---------------
    g_vel, g_nox, g_co = 0.0, 0.0, 0.0

    # ----------------------------- Trajectory loop ---------------------------
    for _ in tf.range(N):
        # Reset models and states
        tf.py_function(trans_func.reset_models, [], [])
        tf.py_function(controller.reset_states, [], [])

        # Random initial state
        init_state = sample_init_state()
        _, mf, brk, ice_sp = tf.unstack(init_state)

        torque, _, _, _, _ = trans_func.predict_ice(ice_sp, mf, T_amb_K, p_amb_bar)
        if clipping:
            torque = tf.clip_by_value(torque, -50.0, 300.0)

        mask_ice = ice_sp < 900.0
        torque = tf.where(mask_ice, 0.0, torque)
        mf = tf.where(mask_ice, 0.0, mf)

        vel, _ = trans_func.predict_PG(ice_sp, 0.0, torque, brk)

        # Accumulators by trajectory
        t_vel, t_nox, t_co = 0.0, 0.0, 0.0

        # --------------------------- Simulation loop ---------------------------
        for i in tf.range(total_steps):
            x = tf.stack([vel_target, vel, mf, brk, ice_sp])
            inp = tf.reshape(x, (1, 1, 5))

            out = controller(inp, training=False)[0]
            mf, brk, ice_sp = tf.unstack(out, num=3)

            torque, nox, _, co, _ = trans_func.predict_ice(
                ice_sp, mf, T_amb_K, p_amb_bar
            )
            if clipping:
                torque = tf.clip_by_value(torque, -50.0, 300.0)

            mask_ice = ice_sp < 900.0
            torque = tf.where(mask_ice, 0.0, torque)

            vel_out, _ = trans_func.predict_PG(ice_sp, 0.0, torque, brk)

            w = tf.minimum(
                1.0, tf.cast(i, tf.float32) / tf.cast(warmup_steps, tf.float32)
            )  # warm-up 0→1

            # ---------- acumular SOLO en las últimas `last_steps` ----------
            t_vel += w * tf.abs(vel_target - vel_out)
            t_nox += w * nox
            t_co += w * co

            vel = vel_out  # próximo paso

        # Medias de la trayectoria
        inv_last = 1.0 / tf.cast(last_steps, tf.float32)
        g_vel += t_vel * inv_last
        g_nox += t_nox * inv_last
        g_co += t_co * inv_last

    # Medias globales sobre N trayectorias
    inv_N = 1.0 / tf.cast(N, tf.float32)
    return g_vel * inv_N, g_nox * inv_N, g_co * inv_N
