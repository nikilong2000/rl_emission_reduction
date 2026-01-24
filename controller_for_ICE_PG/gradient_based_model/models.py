import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import tensorflow_addons as tfa  # Used for LayerNormLSTMCell
import tensorflow as tf
from tensorflow.keras import layers

# class ScaledController(tf.keras.Model):

#     """
#     A recurrent neural network controller that handles input/output scaling internally.

#     This model takes a 5-dimensional state vector and outputs a 3-dimensional
#     action vector corresponding to 'mf' (fuel mass), 'brk' (brake), and
#     'ice_sp' (engine speed).
#     """
#     def __init__(self,
#                  scaler_params: dict,
#                  units: int = 128,
#                  alpha: float = 250):

#         """
#         Initializes the ScaledController model.

#         Parameters
#         ----------
#         scaler_params : dict
#             A dictionary containing the parameters of a scikit-learn MinMaxScaler,
#             including 'data_min', 'data_max', 'scale', and 'min' as NumPy arrays.
#         units : int, optional
#             The number of hidden units in the LSTM cell, by default 128.
#         alpha : float, optional
#             A sharpness parameter for the sigmoid gate function, controlling how
#             quickly the fuel mass is forced to its minimum value, by default 250.
#         """


#         super().__init__(name="scaled_controller")

#         # 1) we get data_min, data_max, scale and min
#         dmin   = scaler_params['data_min'].astype('float32')
#         dmax   = scaler_params['data_max'].astype('float32')
#         scale  = scaler_params['scale'].astype('float32')
#         min_tf = scaler_params['min'].astype('float32')

#         # TF constants
#         self._min_tf  = tf.constant(min_tf)
#         self._scale   = tf.constant(scale)
#         self._max     = tf.constant(dmax)
#         self._range   = self._max - tf.constant(dmin)


#         # 2) threshold for ice_sp = 900
#         self.tau_norm = 900.0 * self._scale[4] + self._min_tf[4]
#         tf.print("self.tau_norm", self.tau_norm)
#         self.alpha = alpha

#         # 3) layers
#         ln_cell = tfa.rnn.LayerNormLSTMCell(units)  # <-- ADDED
#         self.lstm = tf.keras.layers.RNN(            # <-- SUBSTITUTED
#             ln_cell, stateful=True, return_sequences=False
#         )

#         self.dense = layers.Dense(units // 2, activation="linear", name="deltas")
#         self.delta = layers.Dense(3, activation="tanh")


#     def _to_norm(self, x):
#         return x * self._scale + self._min_tf

#     def _from_norm(self, n):
#         return (n - self._min_tf) / self._scale

#     def call(self, inputs):
#         """
#         Defines the forward pass of the controller.

#         Parameters
#         ----------
#         inputs : tf.Tensor
#             A batch of state vectors with shape (batch_size, 1, 5).

#         Returns
#         -------
#         tf.Tensor
#             A batch of predicted *normalized* action vectors [mf, brk, ice_sp]
#             with shape (batch_size, 3). The output must be descaled manually
#             using the `_from_norm` method if physical values are needed.
#         """

#         x      = tf.squeeze(inputs, axis=1)     # (batch,5)
#         x_norm = self._to_norm(x)               # (batch,5)

#         h      = self.lstm(tf.expand_dims(x_norm, 1))
#         h      = self.dense(h)
#         deltas = self.delta(h)                  # (batch,3)

#         orig    = x_norm[:, 2:]                 # (batch,3)
#         abs_raw = orig + deltas                 # tanto orig como deltas en rango [-1,1]


#         # 4) bounded range [-1,1]
#         bounded = tf.tanh(abs_raw)

#         # 5) We take ice
#         ice_norm = bounded[:, 2:3]

#         # 6) Gate for ice_sp < 900 = -0.6 in the relevant range
#         gate     = tf.sigmoid((ice_norm - self.tau_norm) * self.alpha)

#         # 7) applies the gate using interpolation to replace the value
#         valor_minimo_mf = -1.0
#         mf_norm = gate * bounded[:, 0:1] + (1.0 - gate) * valor_minimo_mf


#         # 8) Rest of the prediction
#         rest_norm = bounded[:, 1:]                             # brk_norm, ice_sp_norm

#         # 9) returns the 3 physical outputs
#         return tf.concat([mf_norm, rest_norm], axis=1)           # (batch,3)

#     def reset_states(self):
#         """Resets the internal states of the stateful LSTM layer."""
#         self.lstm.reset_states()


import tensorflow as tf
from tensorflow.keras import layers


class ScaledController(tf.keras.Model):
    def __init__(self, scaler_params: dict, units: int = 128, alpha: float = 100):

        super().__init__(name="scaled_controller")

        # 1) we obtain data_min, data_max, scale and min
        dmin = scaler_params["data_min"].astype("float32")
        dmax = scaler_params["data_max"].astype("float32")
        scale = scaler_params["scale"].astype("float32")
        min_tf = scaler_params["min"].astype("float32")

        # TF constants
        self._min_tf = tf.constant(min_tf)
        self._scale = tf.constant(scale)
        self._max = tf.constant(dmax)
        self._range = self._max - tf.constant(dmin)

        # 2) physical limits in normalized space
        phys_lower = tf.constant([3.0, 0.0, 0.0], dtype=tf.float32)
        phys_upper = tf.constant([70.0, 100.0, 4500.0], dtype=tf.float32)

        # now we use x * scale + min
        self.lower_norm = phys_lower * self._scale[2:] + self._min_tf[2:]
        self.upper_norm = phys_upper * self._scale[2:] + self._min_tf[2:]

        #         print(self.lower_norm)
        #         print(self.upper_norm)

        # 3) threshold for ice_sp = 900
        self.tau_norm = 900.0 * self._scale[4] + self._min_tf[4]
        #         print(f"τ_norm = {self.tau_norm.numpy():.6f}")
        self.alpha = alpha

        # 4) layers
        ln_cell = tfa.rnn.LayerNormLSTMCell(units)  # <-- ADDED
        self.lstm = tf.keras.layers.RNN(  # <-- REPLACED
            ln_cell, stateful=True, return_sequences=False
        )

        self.dense = layers.Dense(units // 2, activation="tanh")
        self.delta = layers.Dense(3, activation="linear", name="deltas")

    def _to_norm(self, x):
        return x * self._scale + self._min_tf

    def _from_norm(self, n):
        return (n - self._min_tf) / self._scale

    def call(self, inputs):
        x = tf.squeeze(inputs, axis=1)  # (batch,5)
        x_norm = self._to_norm(x)  # (batch,5)

        h = self.lstm(tf.expand_dims(x_norm, 1))
        h = self.dense(h)
        deltas = self.delta(h)  # (batch,3)

        orig = x_norm[:, 2:]  # (batch,2)
        abs_raw = orig + deltas

        # 4) soft clipping with sigmoid
        bounded = self.lower_norm + (self.upper_norm - self.lower_norm) * tf.sigmoid(
            abs_raw
        )
        #         tf.print("bounded =", bounded, summarize=-1)

        # 5) gate for ice_sp < 900
        ice_norm = bounded[:, 2:3]

        gate = tf.sigmoid((ice_norm - self.tau_norm) * self.alpha)
        #         tf.print("gate =", gate, "self.tau_norm = ",self.tau_norm,"ice_norm=",ice_norm, summarize=-1)

        # 6) de-normalize EVERYTHING
        full_norm = tf.concat([x_norm[:, :2], bounded], axis=1)  # (batch,5)
        full_phys = self._from_norm(full_norm)  # (batch,5) in physical space

        # 7) apply gate in physical space
        mf_phys = full_phys[:, 2:3] * gate  # if gate=0 → mf_phys=0
        rest_phys = full_phys[:, 3:]  # brk_phys, ice_sp_phys

        # 8) return 3 physical outputs
        #         tf.print("gate =", gate, "self.tau_norm = ",self.tau_norm,"ice_norm=",ice_norm, summarize=-1)
        return tf.concat([mf_phys, rest_phys], axis=1)  # (batch,3)

    def reset_states(self):
        self.lstm.reset_states()
