# Make sure there is NOTHING before this line
from transition_function_model import (
    setup_transition_function_model,
)
from TD3_Ray import *
from joblib import load


import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names",
)


if __name__ == "__main__":
    # Define paths to environment models ---
    # Commas at the end have been removed to avoid errors.
    ruta_ICE_model = "../src/models_markus/ICE_Model_Update_01"
    ruta_outlook = "../src/models_markus/PG_Model_M1.1_without_EM1_Torque"

    # Create the environment transition function ---
    # Correct variables defined above are used.
    # This function is the "heart" of the environment the agent will use.
    print("Configuring the environment...")
    t_function = setup_transition_function_model(ruta_ICE_model, ruta_outlook)

    # Load the normalizer and prepare its parameters ---
    # The scaler object containing normalization statistics is loaded.
    print("Loading scaling parameters...")
    scaler = load("../src/escalados/rl.lib")

    # The parameters dictionary that the agent networks need is created.
    scaler_params = {
        "data_min": scaler.data_min_,
        "data_max": scaler.data_max_,
        "scale": scaler.scale_,
        "min": scaler.min_,
    }
    # Start Ray (only once per script)
    # Configured to use system memory if necessary (spilling)
    if ray.is_initialized():
        ray.shutdown()
    ray.init(object_store_memory=5 * 10**9)  # Assigns 5 GB

    bach = 256

    U = 64
    B = 32

    # --- Learner Creation and Start ---
    td3_learner = TD3(
        f_transicio=t_function,
        version="pls5",
        act_dim=3,
        obs_dim=5,
        replay_size=1000000,
        batch_size=bach,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        scaler_params=scaler_params,
        vel_target=70,
        num_workers=3,  # Use 3 CPU cores
        U=U,
        B=B,
        early_stop=500,
        reuse_warmup_buffer=True,
    )

    # Start asynchronous training
    td3_learner.learn(
        total_timesteps=1000000,
        learning_starts=bach * 3,
        train_freq=int(bach * U) * 2,
        gradient_steps=5000,
    )

    # Stop Ray upon completion
    ray.shutdown()
