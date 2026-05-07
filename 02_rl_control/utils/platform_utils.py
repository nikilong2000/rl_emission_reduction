import os
import sys

_TF_FORCE_CPU_MODELS_ENV = "RL_TF_FORCE_CPU_MODELS"


def configure_environment():
    """
    Pre-TF-import setup. Call before any module that imports TensorFlow.
    Sets helpful environment variables (logging suppression etc.).
    Does NOT force CPU — GPU is probed later via configure_tf_devices().
    """
    if sys.platform.startswith("linux"):
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        os.environ.setdefault(_TF_FORCE_CPU_MODELS_ENV, "0")
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
        os.environ[_TF_FORCE_CPU_MODELS_ENV] = "0"
        print("[platform_utils] GPU operational – using GPU for inference.")
    else:
        print(
            "[platform_utils] GPU probe failed (CUDA driver / PTX version mismatch). "
            "Falling back to CPU-only execution."
        )
        _enable_cpu_fallback(tf)


def should_force_cpu_for_tf_models():
    """Return True when TF model loading/inference should be pinned to CPU."""
    return os.environ.get(_TF_FORCE_CPU_MODELS_ENV, "0") == "1"


def _enable_cpu_fallback(tf):
    os.environ[_TF_FORCE_CPU_MODELS_ENV] = "1"
    try:
        tf.config.set_visible_devices([], "GPU")
        print("[platform_utils] Disabled visible GPUs for TensorFlow.")
    except RuntimeError as exc:
        print(
            "[platform_utils] Could not disable visible GPUs after TF init. "
            f"Will force CPU placement for model loading/inference instead. ({exc})"
        )


def _probe_gpu(tf):
    """Execute representative GPU ops. Returns True if CUDA runtime is functional."""
    try:
        with tf.device("/GPU:0"):
            x = tf.random.uniform((64, 64), dtype=tf.float32)
            y = tf.matmul(x, x)
            _ = float(tf.reduce_sum(y).numpy())
        return True
    except Exception as exc:
        print(f"[platform_utils] GPU probe exception: {exc}")
        return False
