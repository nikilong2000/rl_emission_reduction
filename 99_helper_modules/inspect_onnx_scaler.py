import os
import onnxruntime as ort


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
    scaler_path = os.path.join(
        base_dir, "controller_for_ICE_PG/SHARE/CTTC_models/ONNX/ICE/scaler_input.onnx"
    )
    inspect_model(scaler_path)


if __name__ == "__main__":
    main()
