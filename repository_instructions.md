# Repository Instructions and Overview

## 1. Project Overview

This project focuses on the implementation and evaluation of advanced control strategies for vehicular environments, specifically targeting Internal Combustion Engine (ICE) and Hybrid Powertrain (PG) systems. The core objective is to optimize vehicle performance (speed control) and potentially emissions using various control methodologies, including Predictive Control, Gradient-Based Models, and Reinforcement Learning.

The project utilizes LSTM (Long Short-Term Memory) neural networks to simulate the vehicle's dynamic behavior, replacing traditional physical simulations for faster and more efficient evaluation.

## 2. Directory Structure & Purpose

The repository is organized into several key directories, each serving a specific purpose in the development and evolution of the control models.

### `v2_class_layout`

**Status:** **Newer / Modularized Version**
This directory contains a refactored, object-oriented implementation of the control logic found in `pydesk_model_20240414graphs`. It is designed for better maintainability and scalability.

- **`LDCI_Layout.py`**: The main class that orchestrates the simulation. It integrates the engine and controller, handles data logging, and plotting.
- **`PredictiveController.py`**: Implements the predictive control logic.
- **`ExtendedEngine.py`**: Encapsulates the engine model and its interaction with the neural network surrogates.
- **`LDCI_study.py`**: Likely the execution script to run studies using the class layout.

### `pydesk_model_20240414graphs`

**Status:** **Older / Script-Based Version**
This directory contains a monolithic script-based implementation. It appears to be a snapshot of the work as of April 14, 2024.

- **`LDproject_to_script.py`**: A large script containing the entire control loop, model loading, and plotting logic in a procedural format.
- **`cornet_model_20240414.ipynb`**: A Jupyter notebook version of the model, likely used for initial development and visualization.
- **`model.h5` / `model_weights.hdf5`**: Keras model files used for the simulation environment.

### `controller_for_ICE_PG`

**Status:** **Advanced Control Strategies**
This directory explores advanced control methodologies using the LSTM environment.

- **`model_discussion`**: Preliminary phase for validating the LSTM models (ICE and PG).
- **`gradient_based_model`**: Implements control strategies that leverage the differentiable nature of the neural networks to optimize control inputs via gradient propagation.
- **`reinforcement_learning_model`**: Implements a Reinforcement Learning (RL) agent trained to control the vehicle through trial and error.
- **`src`**: Shared modules and functions used across these advanced models.

### `SHARE`

**Status:** **Shared Resources**
Contains shared models and data used across different parts of the project.

- **`CTTC_models`**: Likely contains ONNX or other model formats shared between different implementations.

## 3. Detailed Component Analysis

### Comparison: `v2_class_layout` vs. `pydesk_model_20240414graphs`

- **`pydesk_model_20240414graphs`** is a procedural implementation where all logic (data loading, control loops, plotting) is mixed in a single file. It is useful for understanding the raw flow of data but harder to extend.
- **`v2_class_layout`** structures this logic into classes:
  - `LDCI_Layout`: Manages the experiment.
  - `ExtendedEngine`: Manages the "plant" (vehicle model).
  - `PredictiveController`: Manages the "controller" (logic).
  - **Recommendation**: Use `v2_class_layout` for future development and `pydesk_model_20240414graphs` as a reference for the original logic.

## 4. Usage Instructions

### Running the Class-Based Model (`v2_class_layout`)

1.  Navigate to `v2_class_layout`.
2.  Ensure you have the required libraries (numpy, matplotlib, tensorflow/keras).
3.  Run the main script (likely `LDCI_study.py` or by instantiating `LDCI_Layout` in a new script).
    ```bash
    cd v2_class_layout
    python3 LDCI_study.py
    ```

### Running the Advanced Models (`controller_for_ICE_PG`)

Each subdirectory has its own specific instructions (refer to their respective `README.md` files).

- **Gradient-Based Model**:

  ```bash
  cd controller_for_ICE_PG/gradient_based_model
  # Follow instructions in README_GBM.md
  ```

- **Reinforcement Learning Model**:
  ```bash
  cd controller_for_ICE_PG/reinforcement_learning_model
  # Follow instructions in README_RLM.md
  ```

## 5. Dependencies

The project relies on standard Python scientific and machine learning libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow` / `keras`
- `scikit-learn` (for scaling)
- `joblib`

Ensure these are installed in your environment.
