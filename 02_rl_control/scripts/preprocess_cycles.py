import os
import glob
import pandas as pd
import json

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
WLTC_DIR = os.path.join(PROJECT_ROOT, "internal_lstm_models/NN_Application/Input_data")
ICE_DIR = os.path.join(PROJECT_ROOT, "internal_lstm_models/data_ice_processed")
DRIVETRAIN_DIR = os.path.join(
    PROJECT_ROOT, "internal_lstm_models/data_drivetrain_processed"
)
OUTPUT_DIR = os.path.join(CURRENT_DIR, "../data_train")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def standardize_columns(columns):
    new_cols = []
    mapping = {}
    for col in columns:
        cleaned = col.replace(";", "_").replace(" ", "_").lower()

        # exact replacements for Drivetrain terminology or ICE terminology mapped to WLTC standard
        if cleaned == "tot_burned_fuel_gps":
            cleaned = "fuel_tot_gps"
        elif cleaned == "t_avg_scr1_k":
            cleaned = "t_gas_eo_k"
        elif cleaned == "t_tailpipe_k":
            cleaned = "t_gas_tp_k"
        elif cleaned == "time01":
            cleaned = "time_s"
        elif cleaned == "soc":
            cleaned = "soc_1"
        else:
            # substring replacements to change ICE naming format out_m/in_m back to WLTC tp/eo
            cleaned = cleaned.replace("_in_m_", "_eo_")
            cleaned = cleaned.replace("_out_m_", "_tp_")

        new_cols.append(cleaned)
        if cleaned != col.lower():
            # Only record if the meaning or structure fundamentally changed beyond simple lowercasing
            mapping[col] = cleaned
    return new_cols, mapping


def process_files():
    all_mappings = {}
    total_processed = 0

    # 1. Process WLTC files
    print("Processing WLTC Cycles...")
    wltc_files = glob.glob(os.path.join(WLTC_DIR, "WLTC*.csv"))
    for filepath in wltc_files:
        filename = os.path.basename(filepath)
        try:
            df = pd.read_csv(filepath, sep=";")
            df.columns, mapping = standardize_columns(df.columns)
            all_mappings.update(mapping)

            out_path = os.path.join(OUTPUT_DIR, f"{filename}")
            df.to_csv(out_path, index=False)
            total_processed += 1
        except Exception as e:
            print(f"Failed to process WLTC {filename}: {e}")

    # 2. Process ICE files
    print("Processing ICE Cycles...")
    ice_files = glob.glob(os.path.join(ICE_DIR, "*.csv"))
    for filepath in ice_files:
        filename = os.path.basename(filepath)
        try:
            df = pd.read_csv(filepath, sep=",")
            df.columns, mapping = standardize_columns(df.columns)
            all_mappings.update(mapping)

            out_path = os.path.join(OUTPUT_DIR, f"ice_{filename}")
            df.to_csv(out_path, index=False)
            total_processed += 1
        except Exception as e:
            print(f"Failed to process ICE {filename}: {e}")

    # 3. Process Drivetrain files
    print("Processing Drivetrain Cycles...")
    dt_files = glob.glob(os.path.join(DRIVETRAIN_DIR, "*.csv"))
    for filepath in dt_files:
        filename = os.path.basename(filepath)
        try:
            df = pd.read_csv(filepath, sep=",")
            df.columns, mapping = standardize_columns(df.columns)
            all_mappings.update(mapping)

            out_path = os.path.join(OUTPUT_DIR, f"drivetrain_{filename}")
            df.to_csv(out_path, index=False)
            total_processed += 1
        except Exception as e:
            print(f"Failed to process Drivetrain {filename}: {e}")

    print(f"Processed {total_processed} files into {OUTPUT_DIR}")

    # Save mapping for expert review
    mapping_out_path = os.path.join(CURRENT_DIR, "column_mapping_assumptions.json")
    with open(mapping_out_path, "w") as f:
        json.dump(all_mappings, f, indent=4)
    print(f"Saved column mapping rules to {mapping_out_path}")


if __name__ == "__main__":
    process_files()
