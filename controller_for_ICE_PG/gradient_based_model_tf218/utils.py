import tensorflow as tf
from transition_function_model import setup_transition_function_model


def run_ice_sanity_check(
    path_ice="../src/models_markus/ICE_Model_Update_01",
    path_pg="../src/models_markus/PG_Model_M1.1_without_EM1_Torque",
):
    """
    Runs a manual sanity check for the ICE model by executing 5 iterations
    with constant inputs and printing the results.

    This checks if the recurrent nature of the LSTM models causes value drift
    even with static inputs.
    """
    print(f"Loading modules from 'transition_function_model.py'...")

    # 1. Instantiate the environment
    try:
        tf_env = setup_transition_function_model(path_ice, path_pg)
    except Exception as e:
        print(f"\nError initialising the model. Please verify the paths: {e}")
        return

    # 2. Initial reset
    tf_env.reset_models()

    # 3. Table header for ALL variables
    print("\n--- EXPERIMENT RESULTS (TensorFlow Original) ---")
    # Adjust column width to fit everything
    header = f"{'Iter':<5} | {'Torque':<12} | {'NO':<12} | {'NO2':<12} | {'CO':<12} | {'CO2':<12}"
    print(header)
    print("-" * len(header))

    # 4. Constant input values
    inputs = {
        "Speed_rpm": 2000.0,
        "m_fuel_mg": 35.0,
        "T_amb_K": 298.0,
        "p_amb_bar": 1.013,
    }

    sp_rpm = tf.constant(inputs["Speed_rpm"], dtype=tf.float32)
    fuel = tf.constant(inputs["m_fuel_mg"], dtype=tf.float32)
    t_amb = tf.constant(inputs["T_amb_K"], dtype=tf.float32)
    p_amb = tf.constant(inputs["p_amb_bar"], dtype=tf.float32)

    # 5. Loop of 5 calls
    for i in range(5):
        # Call to the method
        torque_nm, no_out, no2_out, co_out, co2_out = tf_env.predict_ice(
            sp_rpm, fuel, t_amb, p_amb
        )

        # Extract values (numpy)
        v_torque = torque_nm.numpy()
        v_no = no_out.numpy()
        v_no2 = no2_out.numpy()
        v_co = co_out.numpy()
        v_co2 = co2_out.numpy()

        # Print row with all variables
        print(
            f"{i+1:<5} | {v_torque:<12.5f} | {v_no:<12.5f} | {v_no2:<12.5f} | {v_co:<12.5f} | {v_co2:<12.5f}"
        )

    print("-" * len(header))


if __name__ == "__main__":
    run_ice_sanity_check()
