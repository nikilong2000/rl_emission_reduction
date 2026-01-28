
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

path_ice = "../SHARE/CTTC_models/ONNX/ICE/model.h5"
path_pg = "../SHARE/CTTC_models/ONNX/PG/model.h5"

print(f"Checking {path_ice}...")
try:
    m = load_model(path_ice, compile=False)
    print("Success loading ICE h5")
    m.summary()
except Exception as e:
    print(f"Failed ICE: {e}")

print(f"Checking {path_pg}...")
try:
    m = load_model(path_pg, compile=False)
    print("Success loading PG h5")
    m.summary()
except Exception as e:
    print(f"Failed PG: {e}")
