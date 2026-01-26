# V2 Rebuild Notes

## Step 1: Simulation Core - Status

The simulation.py wrapper has been created but cannot be fully validated due to Keras version incomp atibility.

### Issue
The existing .h5 models were created with TensorFlow 2.x and Keras 2.x, which used the `time_major` parameter for LSTM layers. This parameter has been removed in Keras 3.x (which is required by TensorFlow 2.18+).

### Options
1. **Retrain the models** with Keras 3.x
2. **Convert the models** to a compatible format
3. **Use TensorFlow 2.15** and Keras 2.15 (requires downgrade)
4. **Skip model validation** for now and proceed with environment/agent implementation

### Decision
For now, we've created a wrapper interface that reuses the existing `transition_function_model` from the old codebase. When the Keras/TensorFlow compatibility issue is resolved (by retraining models or using a compatible environment), the simulation will work.

The interface is defined and ready to use once models are compatible:
- `SimulationModel` class with `reset_states()`, `predict_ice()`, `predict_pg()`, and `step()` methods
- Clean wrapper around the transition function model

##Next Steps
Proceed to Step 2: Environment Component with the observation space fix.
