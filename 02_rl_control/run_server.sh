#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_server.sh  –  Linux-server entry point for train_ppo.py
#
# Problem: TF 2.18 bundles CUDA runtime 12.8 and generates PTX 8.7 at runtime.
#          Server driver 535.113.01 only supports up to CUDA 12.2 (PTX 8.2),
#          so driver-side JIT compilation fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION.
#
# Fix:     The pip package `nvidia-cuda-nvcc-cu12` ships ptxas 12.8.
#          When ptxas is on PATH, TF pre-compiles PTX → SASS (machine code)
#          before passing it to the driver, so no JIT is needed.
#
# Install once (already done if you see ptxas below):
#   pip install "nvidia-cuda-nvcc-cu12==12.8.*"
#
# Usage:
#   bash 02_rl_control/run_server.sh [--continue_from <path/to/model.zip>]
# ─────────────────────────────────────────────────────────────────────────────

VENV="/home/usuaris.new/niklas.long.schiefelbein/environments/.venv_py312_tf218"
PTXAS_DIR="$VENV/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin"

if [ ! -f "$PTXAS_DIR/ptxas" ]; then
    echo "ERROR: ptxas not found at $PTXAS_DIR"
    echo "Run: $VENV/bin/pip install 'nvidia-cuda-nvcc-cu12==12.8.*'"
    exit 1
fi

export PATH="$PTXAS_DIR:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# LSTMs (TF/Keras) run on GPU; PPO agent (SB3/PyTorch) runs on CPU
exec "$VENV/bin/python" "$SCRIPT_DIR/train_ppo.py" --agent_device cpu "$@"
