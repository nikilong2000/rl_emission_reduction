import pandas as pd
import numpy as np

df_old = pd.read_csv("internal_lstm_models/NN_Application/Input_data/WLTC.csv", sep=";")
df_new = pd.read_csv("02_rl_control/data_train/WLTC.csv", sep=",")

# old columns are mixed case, new columns are lowercase and use _ instead of spaces/semi-colons
def clean_col(col):
    return col.replace(";", "_").replace(" ", "_").lower()

old_cols_cleaned = [clean_col(c) for c in df_old.columns]
df_old.columns = old_cols_cleaned

# Check shape
print(f"Old shape: {df_old.shape}")
print(f"New shape: {df_new.shape}")

# Check columns
missing_in_new = set(df_old.columns) - set(df_new.columns)
missing_in_old = set(df_new.columns) - set(df_old.columns)
print(f"Missing in new: {missing_in_new}")
print(f"Missing in old: {missing_in_old}")

# Compare values
common_cols = list(set(df_old.columns) & set(df_new.columns))
differences = {}
for col in common_cols:
    if not np.allclose(df_old[col].fillna(0).values, df_new[col].fillna(0).values, atol=1e-5):
        differences[col] = np.abs(df_old[col].fillna(0).values - df_new[col].fillna(0).values).max()

print(f"Columns with differences: {differences}")

