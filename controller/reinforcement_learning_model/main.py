# Asegúrate de que no haya NADA antes de esta línea
from transition_function_model import (
    setup_transition_function_model,
)
from TD3_Ray import *
from joblib import load


import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names"
)



if __name__ == '__main__':
    #Definir las rutas a los modelos del entorno ---
    # Se han eliminado las comas al final para evitar errores.
    ruta_ICE_model = "../src/models_markus/ICE_Model_Update_01"
    ruta_outlook = "../src/models_markus/PG_Model_M1.1_without_EM1_Torque"

    # Crear la función de transición del entorno ---
    # Se usan las variables correctas definidas arriba.
    # Esta función es el "corazón" del entorno que el agente usará.
    print("Configurando el entorno...")
    t_function = setup_transition_function_model(ruta_ICE_model, ruta_outlook)

    # Cargar el normalizador y preparar sus parámetros ---
    # Se carga el objeto scaler que contiene las estadísticas de normalización.
    print("Cargando los parámetros de escalado...")
    scaler = load("../src/escalados/rl.lib")

    # Se crea el diccionario de parámetros que las redes del agente necesitan.
    scaler_params = {
        "data_min": scaler.data_min_,
        "data_max": scaler.data_max_,
        "scale":    scaler.scale_,
        "min":      scaler.min_,
    }
    # Iniciar Ray (solo una vez por script)
    # Se configura para usar la memoria del sistema si es necesario (spilling)
    if ray.is_initialized():
        ray.shutdown()
    ray.init(object_store_memory=5 * 10**9) # Asigna 5 GB
    
    bach = 256
    
    U = 64
    B = 32
    
    # --- Creación e inicio del Learner ---
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
        num_workers=3, # Usar los 3 núcleos de CPU
        U=U,
        B=B,
        early_stop=500,
        reuse_warmup_buffer= True
    )
    
    # Iniciar el entrenamiento asíncrono
    td3_learner.learn(total_timesteps=1000000, learning_starts=bach*3, train_freq=int(bach*U)*2, gradient_steps=5000)

    # Detener Ray al finalizar
    ray.shutdown()
    