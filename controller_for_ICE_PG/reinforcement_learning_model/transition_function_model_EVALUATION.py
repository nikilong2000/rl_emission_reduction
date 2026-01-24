# transition_function_model_EVALUATION.py (CORRECTED AND VERIFIED VERSION)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import clone_model, load_model
import joblib
import os
import time


def get_model(ruta_modelo):
    ruta_completa = os.path.join(ruta_modelo, "model.h5")
    ruta_completa = os.path.normpath(ruta_completa)
    print(f"🔍 Attempting to load model from: {ruta_completa}")
    try:
        return load_model(ruta_completa)
    except Exception as e:
        print(f"❌ Error loading model from '{ruta_completa}': {e}")
        raise FileNotFoundError(f"Model not found at '{ruta_completa}': {e}") from e


def setup_transition_function_model(ruta_ICE, ruta_PG, SOC_ini=0.7):
    ICE_model = get_model(ruta_ICE)
    PG_model = get_model(ruta_PG)
    ICE_scale_input = get_scaler(ruta_ICE, type="input")
    ICE_scale_output = get_scaler(ruta_ICE, type="output")
    PG_scale_input = get_scaler(ruta_PG, type="input")
    PG_scale_output = get_scaler(ruta_PG, type="output")
    return transition_function_model(
        ICE_model,
        PG_model,
        ICE_scale_input,
        ICE_scale_output,
        PG_scale_input,
        PG_scale_output,
        SOC_ini=SOC_ini,
    )


def get_scaler(ruta_model, type="input"):
    fitxer = os.path.join(ruta_model, f"{type}_scaler.lib")
    return joblib.load(fitxer)


class transition_function_model:
    def __init__(
        self,
        ICE_model,
        PG_modl,
        ICE_input_scaler,
        ICE_ouput_scaler,
        PG_input_scaler,
        PG_ouput_scaler,
        Torque=0,
        NO_ini=0,
        NO2_ini=0,
        CO_ini=0,
        CO2_ini=0,
        velocity_ini=0,
        SOC_ini=0.7,
    ):
        self.ICE_model = clone_model(ICE_model)
        self.ICE_model.set_weights(ICE_model.get_weights())
        self.PG_modl = clone_model(PG_modl)
        self.PG_modl.set_weights(PG_modl.get_weights())
        self.ICE_model.trainable = False
        self.PG_modl.trainable = False
        self._ICE_in_scale = tf.constant(ICE_input_scaler.scale_, dtype=tf.float32)
        self._ICE_in_min = tf.constant(ICE_input_scaler.min_, dtype=tf.float32)
        self._ICE_out_scale = tf.constant(ICE_ouput_scaler.scale_, dtype=tf.float32)
        self._ICE_out_min = tf.constant(ICE_ouput_scaler.min_, dtype=tf.float32)
        self._PG_in_scale = tf.constant(PG_input_scaler.scale_, dtype=tf.float32)
        self._PG_in_min = tf.constant(PG_input_scaler.min_, dtype=tf.float32)
        self._PG_out_scale = tf.constant(PG_ouput_scaler.scale_, dtype=tf.float32)
        self._PG_out_min = tf.constant(PG_ouput_scaler.min_, dtype=tf.float32)
        self.initial_ice_aux = ICE_ouput_scaler.transform(
            [[Torque, NO_ini, NO2_ini, CO_ini, CO2_ini]]
        ).reshape((1, 1, 5))
        self.initial_PG_aux = PG_ouput_scaler.transform(
            [[velocity_ini, SOC_ini]]
        ).reshape((1, 1, 2))
        self.ice_aux = tf.constant(self.initial_ice_aux, dtype=tf.float32)
        self.PG_aux = tf.constant(self.initial_PG_aux, dtype=tf.float32)

    @staticmethod
    def _scale_minmax(x, scale, min_):
        return x * scale + min_

    @staticmethod
    def _descale_minmax(x_scaled, scale, min_):
        return (x_scaled - min_) / scale

    def reset_models(self):
        if hasattr(self.ICE_model, "reset_states"):
            self.ICE_model.reset_states()
        if hasattr(self.PG_modl, "reset_states"):
            self.PG_modl.reset_states()
        self.ice_aux = tf.constant(self.initial_ice_aux, dtype=tf.float32)
        self.PG_aux = tf.constant(self.initial_PG_aux, dtype=tf.float32)

    def predict_ice(self, Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar):
        timings = {}
        t0 = time.perf_counter()
        x = tf.stack([Speed_rpm, m_fuel_mg, T_amb_K, p_amb_bar], axis=0)
        x = tf.cast(x, tf.float32)
        t1 = time.perf_counter()
        timings["1_package_input"] = t1 - t0
        x_scaled = self._scale_minmax(x, self._ICE_in_scale, self._ICE_in_min)
        x_scaled = tf.reshape(x_scaled, (1, 1, 4))
        t2 = time.perf_counter()
        timings["2_scale"] = t2 - t1
        y_scaled = self.ICE_model([x_scaled, self.ice_aux], training=False)
        y_scaled = tf.reshape(y_scaled, (5,))
        t3 = time.perf_counter()
        timings["3_model_inference"] = t3 - t2
        y = self._descale_minmax(y_scaled, self._ICE_out_scale, self._ICE_out_min)
        outputs = tf.unstack(y, num=5)
        t4 = time.perf_counter()
        timings["4_descale"] = t4 - t3
        return outputs, timings

    def predict_PG(self, ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc):
        timings = {}
        t0 = time.perf_counter()
        x = tf.stack(
            [ICE_Speed_soll_rpm, EM2_Torque_Nm, ICE_Torque_Nm, Brake_perc], axis=0
        )
        x = tf.cast(x, tf.float32)
        t1 = time.perf_counter()
        timings["1_package_input"] = t1 - t0
        x_scaled = self._scale_minmax(x, self._PG_in_scale, self._PG_in_min)
        x_scaled = tf.reshape(x_scaled, (1, 1, 4))
        t2 = time.perf_counter()
        timings["2_scale"] = t2 - t1
        y_scaled = self.PG_modl([x_scaled, self.PG_aux], training=False)
        y_scaled = tf.reshape(y_scaled, (2,))
        t3 = time.perf_counter()
        timings["3_model_inference"] = t3 - t2
        y = self._descale_minmax(y_scaled, self._PG_out_scale, self._PG_out_min)
        outputs = tf.unstack(y, num=2)
        t4 = time.perf_counter()
        timings["4_descale"] = t4 - t3
        return outputs, timings
