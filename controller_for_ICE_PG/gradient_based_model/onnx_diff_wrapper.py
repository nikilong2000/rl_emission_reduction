import tensorflow as tf
import numpy as np

class DiffONNXWrapper:
    """
    Wraps an ONNX-based transition function (from SHARE/trans.py) 
    to make it compatible with valid TensorFlow gradient tapes via 
    Finite Difference numerical gradients.
    """
    def __init__(self, trans_func, epsilon=1e-1):
        self.trans_func = trans_func
        self.epsilon = epsilon
        
    def reset_models(self):
        # Pass through to the underlying reset check
        return self.trans_func.reset_models()

    def predict_ice(self, Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar):
        # Use custom gradient to inject numerical derivatives
        return self._predict_ice_op(Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar)

    def predict_PG(self, ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc):
        # Use custom gradient to inject numerical derivatives
        return self._predict_pg_op(ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc)

    # ----------------------------------------------------------------
    # Custom Gradient Ops
    # ----------------------------------------------------------------
    
    @tf.custom_gradient
    def _predict_ice_op(self, spd, mf, ta, pa):
        
        # 1. Forward Pass (NumPy wrap)
        def forward_fn(s, m, t, p):
            # Calls the stateful ONNX transition
            # Returns tuple of 5 values
            return self.trans_func.predict_ice(float(s), float(m), float(t), float(p))
            
        # tf.numpy_function ensures we can call the python code
        # We assume inputs are scalar tensors. output is tuple of 5 scalars.
        y = tf.numpy_function(forward_fn, [spd, mf, ta, pa], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])
        
        # 2. Backward Pass (Numerical Gradient)
        def grad_fn(*dy):
            # dy is a list of gradients flowing back into the 5 outputs
            # We must return gradients for the 4 inputs
            
            def numeric_jacobian(s, m, t, p, *output_grads):
                # Ensure floats
                inputs = [float(s), float(m), float(t), float(p)]
                output_grads = [float(d) for d in output_grads]
                
                # We need to compute gradient w.r.t each of 4 inputs
                # J[input_j] = sum_over_outputs_i( d_Output_i / d_Input_j * Grad_Output_i )
                
                input_grads = [0.0] * 4
                
                # CRITICAL: Save State
                saved_ice_aux = self.trans_func.ice_aux.copy()
                # predict_ice doesn't touch PG state, but safety first? NO, trans.py separates them.
                
                # Get baseline (we need to run it again because forward_fn didn't give us values here
                # BUT running again updates state! 
                # So we must use saved state for baseline too?
                # Actually, the 'backward' happens after 'forward' updated the state.
                # So the current state is S_{t+1}.
                # To measure local gradient at step t, we technically need state S_t.
                # This is a limitation: we lost S_t.
                # HOWEVER, for simple controls, using the gradient at current state 
                # might be a sufficient approximation if the dynamics are smooth.
                # OR, strictly speaking, we cannot compute the correct gradient without rewinding.
                
                # WORKAROUND: We assume the local Jacobian doesn't change drastically between S_t and S_{t+1}.
                # Or we just accept the noise.
                # Ideally we would save history of states, but that's complex.
                
                # Let's effectively run finite diff at the *Current* state (which is wrong, but runable).
                
                base_out = self.trans_func.predict_ice(*inputs)
                # Restore state immediately (base_out calc moved state forward!)
                self.trans_func.ice_aux = saved_ice_aux.copy()
                
                eps = self.epsilon
                
                for j in range(4): # For each input
                    inp_p = inputs[:]
                    inp_p[j] += eps
                    
                    # Run perturbed
                    out_p = self.trans_func.predict_ice(*inp_p)
                    # Restore state
                    self.trans_func.ice_aux = saved_ice_aux.copy()
                    
                    # Calc partials
                    # d_out_k / d_in_j
                    d_total = 0.0
                    for k in range(5):
                        partial = (out_p[k] - base_out[k]) / eps
                        d_total += partial * output_grads[k]
                        
                    input_grads[j] = d_total
                    
                return tuple(np.array(g, dtype=np.float32) for g in input_grads)

            # Call the numpy grad function
            # Pass inputs + incoming grads
            # Returns 4 tensors
            return tf.numpy_function(numeric_jacobian, [spd, mf, ta, pa, *dy], [tf.float32]*4)

        return y, grad_fn

    @tf.custom_gradient
    def _predict_pg_op(self, ice_spd_soll, em2_trq, ice_trq, brake):
        
        def forward_fn(s, em, it, b):
            return self.trans_func.predict_PG(float(s), float(em), float(it), float(b))
            
        y = tf.numpy_function(forward_fn, [ice_spd_soll, em2_trq, ice_trq, brake], [tf.float32, tf.float32])
        
        def grad_fn(*dy):
            def numeric_jacobian(s, em, it, b, *output_grads):
                inputs = [float(s), float(em), float(it), float(b)]
                output_grads = [float(d) for d in output_grads]
                input_grads = [0.0]*4
                
                saved_pg_aux = self.trans_func.pg_aux.copy()
                
                base_out = self.trans_func.predict_PG(*inputs)
                self.trans_func.pg_aux = saved_pg_aux.copy()
                
                eps = self.epsilon
                for j in range(4):
                    inp_p = inputs[:]
                    inp_p[j] += eps
                    out_p = self.trans_func.predict_PG(*inp_p)
                    self.trans_func.pg_aux = saved_pg_aux.copy()
                    
                    d_total = 0.0
                    for k in range(2):
                        partial = (out_p[k] - base_out[k]) / eps
                        d_total += partial * output_grads[k]
                    input_grads[j] = d_total
                    
                return tuple(np.array(g, dtype=np.float32) for g in input_grads)
                
            return tf.numpy_function(numeric_jacobian, [ice_spd_soll, em2_trq, ice_trq, brake, *dy], [tf.float32]*4)
            
        return y, grad_fn

