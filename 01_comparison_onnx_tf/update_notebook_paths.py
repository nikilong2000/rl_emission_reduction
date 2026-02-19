import json
import os

notebooks = ["comparing_tf_and_onnx.ipynb", "tf_vs_onnx_dynamic.ipynb"]
cwd = os.getcwd()

for nb_file in notebooks:
    path = os.path.join(cwd, nb_file)
    if not os.path.exists(path):
        print(f"Skipping {nb_file}, not found.")
        continue

    print(f"Processing {nb_file}...")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    modified = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                # Update ICE_folder
                if "ICE_folder = os.path.join('CTTC_models', 'ONNX', 'ICE')" in line:
                    new_line = "    ICE_folder = os.path.join('..', 'controller_for_ICE_PG', 'SHARE', 'CTTC_models', 'ONNX', 'ICE')\n"
                    new_source.append(new_line)
                    modified = True
                # Update PG_folder
                elif "PG_folder = os.path.join('CTTC_models', 'ONNX', 'PG')" in line:
                    new_line = "    PG_folder = os.path.join('..', 'controller_for_ICE_PG', 'SHARE', 'CTTC_models', 'ONNX', 'PG')\n"
                    new_source.append(new_line)
                    modified = True
                # Update csvPath
                elif (
                    'csvPath = "../../internal_lstm_models/Test_Cycles/WLTC.csv"'
                    in line
                ):
                    new_line = '    csvPath = "../internal_lstm_models/NN_Application/Input_data/WLTC.csv"\n'
                    new_source.append(new_line)
                    modified = True
                else:
                    new_source.append(line)
            cell["source"] = new_source

    if modified:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {nb_file}")
    else:
        print(f"No changes made to {nb_file}")
