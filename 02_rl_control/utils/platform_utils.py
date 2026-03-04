import os
import sys


def configure_environment():
    """
    Pre-TF-import setup. Call before any module that imports TensorFlow.
    Sets helpful environment variables (logging suppression etc.).
    Does NOT force CPU — GPU is probed later via configure_tf_devices().
    """
    if sys.platform.startswith("linux"):
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        print("[platform_utils] Linux detected.")
    elif sys.platform == "darwin":
        print("[platform_utils] macOS detected.")
    else:
        print(f"[platform_utils] Platform '{sys.platform}' detected.")


def configure_tf_devices():
    """
    Post-TF-import device configuration. Call once after 'import tensorflow'
    and before any model loading.

    Probes available GPUs with a trivial op. Falls back to CPU-only if the
    probe fails (e.g., CUDA driver too old for the TF binary's CUDA version —
    TF 2.18 requires CUDA 12.3 / driver >= 545, server has driver 535).

    On macOS: no-op (no NVIDIA GPU present).
    """
    import tensorflow as tf

    if sys.platform == "darwin":
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[platform_utils] No GPU detected – using CPU.")
        return

    print(f"[platform_utils] GPU(s) found: {[g.name for g in gpus]}. Probing...")

    if _probe_gpu(tf):
        print("[platform_utils] GPU operational – using GPU for inference.")
    else:
        print(
            "[platform_utils] GPU probe failed (CUDA driver / PTX version mismatch). "
            "Falling back to CPU-only execution."
        )
        tf.config.set_visible_devices([], "GPU")


def _probe_gpu(tf):
    """Execute a trivial GPU op. Returns True if CUDA is functional."""
    try:
        with tf.device("/GPU:0"):
            _ = tf.constant(1.0) + tf.constant(1.0)
        return True
    except Exception:
        return False
