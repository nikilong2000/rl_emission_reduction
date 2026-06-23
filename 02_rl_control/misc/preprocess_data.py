import os
import pandas as pd
import glob

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
TEST_CYCLES_DIR = os.path.join(PROJECT_ROOT, "internal_lstm_models/Test_Cycles")
DRIVETRAIN_DIR = os.path.join(PROJECT_ROOT, "internal_lstm_models/Data_Drivetrain")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_test_cycles():
    print("Processing Test Cycles...")
    for filename in ["WLTC_low.csv", "WLTC_high.csv"]:
        filepath = os.path.join(TEST_CYCLES_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, sep=";")
            if "Time_s" in df.columns and "Car_Speed_kmph" in df.columns:
                out_df = df[["Time_s", "Car_Speed_kmph"]].copy()
                out_path = os.path.join(OUTPUT_DIR, f"test_{filename}")
                out_df.to_csv(out_path, index=False)
                print(f"Saved {out_path}")
            else:
                print(f"Missing columns in {filename}")
        else:
            print(f"File not found: {filepath}")


def process_drivetrain_cycles():
    print("Processing Drivetrain Cycles...")
    files = glob.glob(os.path.join(DRIVETRAIN_DIR, "*.csv"))
    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            # Read Data_Drivetrain format
            # Line 1: Comment
            # Line 2: Empty
            # Line 3: Headers
            # Line 4: Units
            df = pd.read_csv(filepath, sep="\t", skiprows=2)

            # The first row of dataframe corresponds to units
            df = df.drop(0).reset_index(drop=True)

            # Clean columns
            df.columns = [col.strip() for col in df.columns]

            if "Time01" in df.columns and "Car_Speed;kmph" in df.columns:
                out_df = pd.DataFrame()
                out_df["Time_s"] = pd.to_numeric(df["Time01"], errors="coerce")
                out_df["Car_Speed_kmph"] = pd.to_numeric(
                    df["Car_Speed;kmph"], errors="coerce"
                )

                # Drop potentially induced NaNs from casting errors (e.g. if a string got through)
                out_df = out_df.dropna()

                out_path = os.path.join(OUTPUT_DIR, f"drive_{filename}")
                out_df.to_csv(out_path, index=False)
            else:
                print(f"Missing required columns in {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
    print(f"Processed {len(files)} drivetrain files.")


if __name__ == "__main__":
    process_test_cycles()
    process_drivetrain_cycles()
    print("Preprocessing completed!")
