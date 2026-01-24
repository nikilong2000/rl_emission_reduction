import tensorflow as tf


@tf.function
def rollout_and_loss(
    controller,
    trans_func,
    init_state,
    alpha,
    beta,
    gamma,
    opt,
    clipping,
    total_steps=1200,
    last_steps=200,
    warmup_steps=10,
    vel_target=70.0,
):
    """
    Simulates a full trajectory (rollout), training the controller at each step.

    This function is compiled into a high-performance TensorFlow graph using
    `@tf.function`. It runs a simulation for `total_steps`, and for each step, it:
    1.  Calculates the controller's action.
    2.  Simulates the environment's response.
    3.  Computes a single-step loss.
    4.  Applies a warm-up ramp to the loss for the first `warmup_steps`.
    5.  Calculates gradients and immediately updates the controller's weights.

    It returns the average loss calculated only over the window from
    `warmup_steps` to the end of the trajectory.

    Parameters
    ----------
    controller : tf.keras.Model
        The controller model to be trained.
    trans_func : object
        The differentiable transition function that simulates the environment.
    init_state : tf.Tensor
        The initial state tensor for the trajectory.
    alpha, beta, gamma : tf.Tensor
        Constant tensors representing the weights for the loss components
        (velocity error, NOx, CO).
    opt : tf.keras.optimizers.Optimizer
        The optimizer instance used to apply gradients at each step.
    clipping : bool
        If True, clips the predicted torque value within a safe range.
    total_steps : int, optional
        The total number of steps in the simulation, by default 1200.
    last_steps : int, optional
        An unused parameter, by default 200. The loss window is determined
        by `warmup_steps`.
    warmup_steps : int, optional
        The number of initial steps during which the loss is gradually ramped up
        from 0 to 1. The final loss is also averaged starting from this step,
        by default 10.
    vel_target : float, optional
        The target velocity for the loss calculation, by default 70.0.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing the mean loss over the final part of the
        trajectory (from `warmup_steps` to `total_steps`).
    """
    a = True
    # Reset ICE/PG memories and stateful controller
    tf.py_function(trans_func.reset_models, [], [])
    tf.py_function(controller.reset_states, [], [])

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

    # Accumulators
    loss_total_last = tf.constant(0.0, dtype=tf.float32)

    # Simulation loop
    for i in tf.range(total_steps):
        # Pack controller input
        x = tf.stack([vel_target, vel, mf, brk, ice_sp])
        inp = tf.reshape(x, (1, 1, 5))  # (1,1,5)

        # Gradient calculation and immediate update
        with tf.GradientTape() as tape:
            # forward step - the model already returns absolute and bounded values
            out = controller(inp, training=True)[0]  # shape (4,)
            mf, brk, ice_sp = tf.unstack(out, num=3)  # now it is a TF op
            #             tf.print("mf =", mf, "brk =", brk, "ice_sp =", ice_sp)

            # ICE and PG transitions (differentiable)
            torque, nox, _, co, _ = trans_func.predict_ice(
                ice_sp, mf, T_amb_K, p_amb_bar
            )
            if clipping:
                torque = tf.clip_by_value(torque, -50, 300.0)

            mask_ice = ice_sp < 900.0  # boolean condition
            torque = tf.where(mask_ice, tf.zeros_like(torque), torque)

            vel_out, _ = trans_func.predict_PG(ice_sp, 0.0, torque, brk)

            #             tf.print("mf =", mf, "brk =", brk, "ice_sp =", ice_sp, "torque", torque)

            # Pérdida de este paso
            step_loss = (
                alpha * tf.square(vel_target - vel_out) + beta * nox + gamma * co
            )

            # ---------- ENMASCARAMIENTO ----------
            w = tf.minimum(
                1.0, tf.cast(i, tf.float32) / tf.cast(warmup_steps, tf.float32)
            )  # ➋ rampa 0→1
            step_loss *= w

        # Aplicar gradientes de inmediato
        grads = tape.gradient(step_loss, controller.trainable_weights)
        opt.apply_gradients(zip(grads, controller.trainable_weights))

        # Si estamos en las últimas 200 iteraciones, acumular esa pérdida
        # Convertimos la condición a float (1.0 si i >= threshold, 0.0 en otro caso)
        loss_total_last += step_loss

        # Update speed for the next step
        vel = vel_out

    # Devolver pérdida media de las últimas N iteraciones
    return loss_total_last / tf.cast(total_steps, tf.float32)
