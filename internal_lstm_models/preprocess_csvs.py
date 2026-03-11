import os
import glob
import csv


def process_file(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    # Find the header line
    header_idx = -1
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith("#"):
            header_idx = i
            break

    if header_idx == -1:
        print(f"No header found in {in_path}")
        return

    header_line = lines[header_idx]
    # Check for tab delimiter
    delimiter = "\t" if "\t" in header_line else ";"
    headers = [h.strip() for h in header_line.strip().split(delimiter)]

    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter=",")
        writer.writerow(headers)

        for line in lines[header_idx + 2 :]:  # skips unit line
            stripped_line = line.strip()
            if not stripped_line:
                continue
            row = [v.strip() for v in stripped_line.split(delimiter)]
            writer.writerow(row)


def main():
    base_dir = "/Users/niklaslongschiefelbein/local_documents/THESIS/code/rl_emission_reduction/internal_lstm_models"
    dirs_to_process = [
        ("Data_Drivetrain", "data_drivetrain_processsed"),
        ("Data_ICE", "data_ice_processed"),
    ]

    for in_dir_name, out_dir_name in dirs_to_process:
        in_dir = os.path.join(base_dir, in_dir_name)
        out_dir = os.path.join(base_dir, out_dir_name)

        os.makedirs(out_dir, exist_ok=True)

        csv_files = glob.glob(os.path.join(in_dir, "*.csv"))

        print(f"Processing {len(csv_files)} files from {in_dir_name} to {out_dir_name}")
        for i, c_file in enumerate(csv_files):
            basename = os.path.basename(c_file)
            out_file = os.path.join(out_dir, basename)
            try:
                process_file(c_file, out_file)
            except Exception as e:
                print(f"Error processing {c_file}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(csv_files)}")

        out_files = glob.glob(os.path.join(out_dir, "*.csv"))
        print(f"Finished {in_dir_name}. In: {len(csv_files)}, Out: {len(out_files)}")


if __name__ == "__main__":
    main()
