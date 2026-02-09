import os
import onnxruntime as ort
import numpy as np


def inspect_model(model_path):
    print(f"\n--- Inspecting {os.path.basename(model_path)} ---")
    try:
        sess = ort.InferenceSession(model_path)

        print("Inputs:")
        for inp in sess.get_inputs():
            print(f"  Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

        print("Outputs:")
        for out in sess.get_outputs():
            print(f"  Name: {out.name}, Shape: {out.shape}, Type: {out.type}")

    except Exception as e:
        print(f"Error loading model: {e}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming standard relative path as per previous findings
    # controller_for_ICE_PG/SHARE/CTTC_models/ONNX/ICE/ICE_onnx.onnx

    ice_path = os.path.join(
        base_dir, "controller_for_ICE_PG/SHARE/CTTC_models/ONNX/ICE/ICE_onnx.onnx"
    )
    pg_path = os.path.join(
        base_dir, "controller_for_ICE_PG/SHARE/CTTC_models/ONNX/PG/PG_onnx.onnx"
    )

    inspect_model(ice_path)
    inspect_model(pg_path)


if __name__ == "__main__":
    main()
