import pandas as pd
import numpy as np

# 16 dims for ICE
_ICE_COLS = [
    "ice_torque_nm", "fuel_tot_gps", "nox_eo_gps", "co_eo_gps", "thc_eo_gps", "t_gas_eo_k", "nox_tp_gps", "co_tp_gps", "co2_tp_gps", "thc_tp_gps", "t_wall_scr1_k", "t_wall_doc_k", "t_sub_dpf_k", "t_wall_scr2_k", "t_wall_scr3_k", "t_gas_tp_k",
]

_ICE_DEFAULTS = [0.0, 0.0, 0.0, 0.0, 0.0, 298.0, 0.0, 0.0, 0.0, 0.0, 298.0, 298.0, 298.0, 298.0, 298.0, 298.0]

df = pd.read_csv("02_rl_control/data_train/WLTC.csv")
df.columns = [col.strip() for col in df.columns]

ice_init_val_row = []
for c, d in zip(_ICE_COLS, _ICE_DEFAULTS):
    if c in df.columns and not pd.isna(df.loc[0, c]):
        ice_init_val_row.append(float(df.loc[0, c]))
    else:
        ice_init_val_row.append(d)

print("ICE INIT OLD: ", _ICE_DEFAULTS)
print("ICE INIT NEW: ", ice_init_val_row)

diff = [a - b for a, b in zip(ice_init_val_row, _ICE_DEFAULTS)]
print("DIFF: ", diff)

