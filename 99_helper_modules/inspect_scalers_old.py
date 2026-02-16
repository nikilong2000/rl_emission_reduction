import os
import joblib
import numpy as np


def inspect_scalers():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    ice_path = os.path.join(
        base_dir, "controller_for_ICE_PG/src/models_markus/ICE_Model_Update_01"
    )
    pg_path = os.path.join(
        base_dir,
        "controller_for_ICE_PG/src/models_markus/PG_Model_M1.1_without_EM1_Torque",
    )

    print("\n--- ICE Scalers ---")
    try:
        ice_in = joblib.load(os.path.join(ice_path, "input_scaler.lib"))
        print(f"ICE Input Scaler Features: {len(ice_in.scale_)}")
    except Exception as e:
        print(f"Error loading ICE input scaler: {e}")

    try:
        ice_out = joblib.load(os.path.join(ice_path, "output_scaler.lib"))
        print(f"ICE Output Scaler Features: {len(ice_out.scale_)}")
    except Exception as e:
        print(f"Error loading ICE output scaler: {e}")

    print("\n--- PG Scalers ---")
    try:
        pg_in = joblib.load(os.path.join(pg_path, "input_scaler.lib"))
        print(f"PG Input Scaler Features: {len(pg_in.scale_)}")
    except Exception as e:
        print(f"Error loading PG input scaler: {e}")

    try:
        pg_out = joblib.load(os.path.join(pg_path, "output_scaler.lib"))
        print(f"PG Output Scaler Features: {len(pg_out.scale_)}")
    except Exception as e:
        print(f"Error loading PG output scaler: {e}")


if __name__ == "__main__":
    inspect_scalers()
