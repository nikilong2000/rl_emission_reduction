# RL Emission Reduction

## Project Overview

This project focuses on the implementation and evaluation of advanced control strategies for a vehicular environment, specifically targeting Internal Combustion Engine (ICE) and Hybrid Powertrain (PG) systems. The internal models for ICE and PG utilise LSTMs (Long Short-Term Memory) neural networks. The objective is speed control and emission reduction using various control methodologies, including Predictive Control, Gradient-Based Models, and Reinforcement Learning.

## Repository Structure

### 📂 [controller_for_ICE_PG](./controller_for_ICE_PG)

**Status:** Research & Development

Contains the implementation and evaluation of advanced control strategies to control speed in a vehicular simulation environment. Includes gradient-based models and initial reinforcement learning experiments. This codebase has been developed by Artur Aubach Altes and has been cleaned and translated (spanish &rarr; english) by Niklas Long Schiefelbein.

**[See Documentation](./controller_for_ICE_PG/README.md)**

### 📂 [v2_rebuild](./v2_rebuild)

**Status:** Clean Implementation (PyTorch)

A clean, modular implementation of the TD3 reinforcement learning controller for hybrid vehicle control. Focuses on single-threaded debugging and clear observation spaces.

**[See Documentation](./v2_rebuild/README.md)**

### 📂 [v2_rebuild_onnx](./v2_rebuild_onnx) (not working yet)

**Status:** Clean Implementation (ONNX)

Similar to `v2_rebuild`, but utilises ONNX models for the improved inference performance of the LSTM simulation environment.

**[See Documentation](./v2_rebuild_onnx/README.md)**

### 📂 [internal_lstm_models](./internal_lstm_models)

**Status:** Model Weights & Assets

Contains the latest version of the pre-trained neural network models (LSTM) used for the ICE and PG simulations (as of 03.02.2026).

**[See Documentation (PDF)](./internal_lstm_models/LDCI2027_Model_Final_ReadMe.pdf)**

### 📂 [pdfs](./pdfs)

**Status:** Documentation

Contains PDF documentation and reference materials for the project.

---
