# Reinforcement Learning Model (Recurrent TD3)

This directory contains the implementation of a Reinforcement Learning (RL) agent to control the vehicle's speed. The implemented algorithm is a recurrent adaptation of **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**, designed to work with the sequential, LSTM-based environment.

## 📜 Approach Description

The agent learns an optimal control policy through direct interaction with the environment. As this is a problem with temporal dependencies (the current state depends on previous ones), a **Recurrent Actor-Critic** architecture has been chosen.

* **Algorithm**: TD3, an `Actor-Critic` and `off-policy` method that improves training stability over DDPG.
* **Neural Networks**: Both the Actor and the Critic use **LSTM (Long Short-Term Memory)** layers to process sequences of observations and maintain an internal state or "memory".

## 🧬 Architecture and Components

The project is structured into several key modules:

* `main.ipynb`: **Main Notebook**. This is the entry point for configuring, training, and evaluating the agent.
* `TD3_Ray.py`: **Agent Logic**. Contains the implementation of the TD3 algorithm, managing network updates and interaction with the `ReplayBuffer`.
* `ActorCriticNetworks.py`: **Neural Networks**.
    * `ActorNetwork`: An LSTM network that receives a state and decides the best action to take. The output is a vector of 3 continuous actions (fuel, brake, ICE setpoint).
    * `CriticNetwork`: Two LSTM networks (Twin Critics) that learn to evaluate how good an action taken in a given state is, producing a Q-value.
* `environment.py`: **Simulation Environment**.
    * Defines the interaction loop, state, actions, and reward function.
    * **State Space (Observation)**: `(target_speed, current_speed, mf, brk, ice_sp)` - 5 dimensions.
    * **Action Space**: `(delta_mf, delta_brk, delta_ice_sp)` - 3 continuous dimensions in `[-1, 1]`.
    * **Reward Function**: The squared speed error is penalized to encourage the agent to follow the reference: $$ R_t = - \alpha (v_{\text{target}} - v_{\text{actual}})^2 $$
* `ReplayBuffer.py`: **Experience Memory**. Stores sequences of transitions (`state, action, reward, next_state`) so the agent can learn from them in a decorrelated manner (off-policy).
* `Noise.py`: **Exploration**. Implements Gaussian noise that is added to actions during training to encourage exploration of the environment.
* `transition_function_model.py`: **Vehicle Dynamics Model**. Abstracts the neural networks for the engine (ICE) and powertrain (PG) that simulate the vehicle's physics.
* `results/`: **Output Directory**. Trained models, performance plots, and logs are saved here.

## ▶️ Execution

The entire workflow (training and evaluation) is controlled from the `main.ipynb` notebook.

1.  **Start Jupyter Notebook or Jupyter Lab** in the project directory.
2.  **Open the `main.ipynb` file**.
3.  **Configure the hyperparameters** in the initial cells (e.g., `learning_rate`, `buffer_size`, `batch_size`).
4.  **Run the cells** in order to train the agent. The progress and results will be displayed and saved in the `results/` folder.

## 📈 Expected Results

Upon completion of the notebook's execution, you will obtain:

* **Saved Models**: The weights of the Actor and Critic networks are saved in the `results/` folder.
* **Training Plots**: A plot is generated showing the cumulative reward per episode, allowing visualization of the agent's learning curve.
* **Evaluation Plots**: A final comparison between the reference speed and the speed achieved by the trained agent.