---
# Controller for a Neuronal Vehicular Environment (ICE+PG) 🚀

This repository contains the implementation and evaluation of advanced control strategies to optimize speed in a vehicular simulation environment. Although the architecture is designed to also manage emissions, the current focus is exclusively on speed control.

Including emissions control in the future would require minimal modifications.
---

## The Simulation Environment: LSTM (ICE+PG)

The project uses a dynamic simulation environment based on two pre-trained LSTM (Long Short-Term Memory) neural networks:

- 🧠 **ICE**: An Internal Combustion Engine model.
- ⚡ **PG**: A hybrid Powertrain model.

These neural networks replace traditional physical simulations, offering a **fast, accurate, and efficient** representation of the vehicle's dynamic behavior. This allows for much more agile evaluation and optimization of control strategies.

Different versions of these LSTM networks were used during development.

---

## Methodologies Explored

To achieve the goal of controlling speed, different approaches have been implemented and compared:

### 0. Model Discussion and Validation (`model_discussion`)

This is a preliminary phase (phase 0) whose objective is to ensure that the implementation and configuration of the LSTM models (ICE and PG) are correct. In this section, the results and plots of a reference execution are replicated to validate that the simulation environment behaves as expected before being used in the optimization phases.

### 1. Gradient-Based Model (`gradient_based_model`)

This approach leverages the differentiable nature of the neural environment. The gradient is propagated through the LSTM networks to efficiently calculate how to adjust the controls, thereby optimizing the strategy to follow a reference speed.

### 2. Reinforcement Learning (`reinforcement_learning_model`)

In this approach, a **Reinforcement Learning (RL) agent** has been trained. The agent learns an optimal control policy by interacting directly with the LSTM environment (trial and error), receiving rewards or penalties based on its performance in maintaining the desired speed.

---

## Project Structure

The repository is organized as follows to clearly separate each approach and the shared code.

```
controller\_for\_ICE\_PG/
├── 📂 gradient\_based\_model/
│   └── 📄 README_GBM.md  (Instructions for this model)
│
├── 📂 model\_discussion/
│   └── 📄 README_MD.md  (Instructions for this model)
│
├── 📂 reinforcement\_learning\_model/
│   └── 📄 README_RLM.md  (Instructions for this model)
│
├── 📂 src/
│   └── 📄 ... (Módulos y funciones compartidas)
│
└── 📄 requirements.txt (Dependencias del proyecto)
```

## ▶️ How to Run the Models

Each approach has its own scripts and execution parameters. For detailed instructions, please consult the corresponding `README.md` file within each directory:

- **For the Gradient-Based Model:**
  - 👉 **[See instructions here](./gradient_based_model/README.md)**

- **For the Reinforcement Learning Model:**
  - 👉 **[See instructions here](./reinforcement_learning_model/README.md)**

```

```
