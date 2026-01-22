import os  # Operative system
import numpy as np  # Mathematics: arrays
import pandas as pd
from keras.models import load_model
import joblib


class ExtendedEngine:  # M0 ANN equivalent: engine + auxiliary controls that are now embedded inside M0
    def __init__(self):

        return

    def load_model(self, str_dir_models, fileroot_model):
        file_model = fileroot_model + ".h5"
        self.model_filename = os.path.join(str_dir_models, file_model)
        self.model = load_model(self.model_filename)

        file_weights = fileroot_model + "_weights.hdf5"
        self.weights_filename = os.path.join(str_dir_models, file_weights)
        self.model.load_weights(self.weights_filename)

        self.model.compile()
        print(self.model.summary())

        return

    def load_scalers(self, str_dir_scalers, fileroot_input, fileroot_output):
        file_input = fileroot_input + ".lib"
        self.scalerIn_filename = os.path.join(str_dir_scalers, file_input)
        self.scalerIn = joblib.load(self.scalerIn_filename)

        file_output = fileroot_output + ".lib"
        self.scalerOut_filename = os.path.join(str_dir_scalers, file_output)
        self.scalerOut = joblib.load(self.scalerOut_filename)
        return

    def create_main_in(self, Nobserv, Linput, Nvar):
        self.main_in = np.ones((Nobserv, Linput, Nvar))
        return

    def create_aux_ini(
        self, par1, par2, par3
    ):  # d_aux_ini to aux_ini, default/initial can be stored directly in the same variable
        self.aux_ini = np.ones((par1, par2, par3))
        self.aux_ini[0, 0, 0] = self.scalerOut.data_range_[0] / 2
        self.aux_ini[0, 0, 1] = self.scalerOut.data_range_[1] / 2
        self.aux_ini[0, 0, 2] = self.scalerOut.data_range_[0] / 2
        self.aux_ini[0, 0, 3] = self.scalerOut.data_range_[0] / 2
        return

    def get_aux_ini(self):
        return self.scalerOut.transform(self.aux_ini[0])

    def set_aux_ini(self, aux_ini):
        self.aux_ini = aux_ini
        return

    def set_main_in(self, v_k, n_ice_sp, P_EM1, P_EM2, GP, m_inj):
        self.main_in[0, 0, 0] = v_k  # Velocity_kph
        self.main_in[0, 0, 1] = n_ice_sp  # Engine_Speed_RPM similar to set point
        self.main_in[0, 0, 2] = P_EM1  # EM1 Power similar to set point
        self.main_in[0, 0, 3] = P_EM2  # EM2 Power similar to set point
        self.main_in[0, 0, 4] = GP  # Grade in percentage
        self.main_in[0, 0, 5] = m_inj  # Injection mass similar to set point

        # Scaling input main
        self.main_in[0] = self.scalerIn.transform(self.main_in[0])
        return

    def run(self):

        v = self.model.predict([self.main_in, self.aux_ini], batch_size=1, steps=1)
        vo = self.scalerOut.inverse_transform(v[0])
        vi = self.scalerIn.inverse_transform(self.main_in[0])

        return v, vi, vo
