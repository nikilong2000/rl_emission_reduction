# Gradient Based Model (Step-by-Step BPTT)

This directory contains the implementation of a neural controller trained using gradient backpropagation through the differentiable simulation environment (BPTT - Backpropagation Through Time).

## 📜 Approach Description

The core of this approach is leveraging that the entire simulation environment (the ICE and PG LSTM models) is completely differentiable. This allows treating the **controller and the environment as a single large recurrent neural network**.

Training is performed _online_ at each simulation step:

1.  At a time instant $t$, the controller predicts an action.
2.  The action is applied to the environment (`transition_function_model`) to obtain the next state and emissions.
3.  A loss function (or cost) is calculated at that instant, penalizing the deviation from the target velocity.
4.  Using `tf.GradientTape`, the gradient of this instantaneous loss with respect to the controller weights is calculated.
5.  The gradient is propagated backwards through the environment networks (PG and then ICE) until it reaches the controller.
6.  The controller weights are updated immediately before moving to the next step $t+1$.

This method allows the controller to learn very efficiently how its actions affect the future state of the vehicle.

- **Algorithm**: Stochastic Gradient Descent applied at each step of a simulated trajectory (step-by-step unrolled BPTT).
- **Loss Function**: The goal is to minimize the squared velocity error, although the formula is prepared to include emissions:
  $$ L*t = \alpha (v*{\text{target}} - v\_{\text{actual}})^2 + \beta \cdot \text{NOx}\_t + \gamma \cdot \text{CO}\_t $$

## 🧬 Architecture and Components

The project is structured into several key modules to separate controller logic, environment, training, and evaluation.

- `main.ipynb`: **Main Notebook**. Entry point for configuring hyperparameters, instantiating models, executing the training loop, and visualizing evaluation results.
- `models.py`: **Controller Architecture**. Defines the `ScaledController` neural network (based on LSTM) that learns the control policy. It includes internal logic to scale inputs/outputs and a special sigmoid gate to nullify fuel (`mf`) when engine revolutions are low.
- `transition_function_model.py`: **Differentiable Environment**. Encapsulates the pre-trained ICE and PG neural models. All its operations are implemented with TensorFlow to allow gradient flow through it.
- `trainer.py` and `step.py`: **Training Logic**. Contains the main function `rollout_and_loss`, decorated with `@tf.function` for high performance. This function executes a complete trajectory (rollout) and applies gradient updates at each step.
- `eval.py`: **Evaluation Module**. Provides functions to simulate trajectories with an already trained controller (without calculating gradients) and generate interactive plots to analyze performance.
- `init_state.py`: **State Initialization**. Offers functions to sample random initial states from real data (CSVs) or uniform distributions, ensuring robust training.
- `models/`: **Output Directory**. The trained controller weights are saved here.

## ▶️ Execution

The entire workflow, from configuration to final evaluation, is managed from the `main.ipynb` notebook.

1.  **Start Jupyter Notebook or Jupyter Lab** in the project directory.
2.  **Open the file `main.ipynb`**.
3.  **Configure hyperparameters** in the initial cells (e.g., `learning_rate`, `epochs`, `alpha`, `beta`, `gamma`).
4.  **Execute the cells in order**. The notebook will take care of loading data, training the controller, and displaying the results. Trained models will be saved in the `models/` folder.

## 📈 Expected Results

Upon completion of the notebook execution, you will get:

- **Saved Model**: The `ScaledController` weights will be saved in the `models/` folder.
- **Training Plot**: The evolution of the loss throughout the training epochs will be shown.
- **Evaluation Plot**: An interactive plot comparing the target velocity with the actual velocity achieved by the trained controller, as well as showing the evolution of control actions and emissions.
