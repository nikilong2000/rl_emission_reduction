#!/bin/bash
#SBATCH --job-name=phase2_sweep
#SBATCH --time=72:00:00
#SBATCH --chdir=.
#SBATCH --array=0-8
#SBATCH --nodelist=jff111,jff116
#SBATCH --ntasks-per-core=1
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err
# ============================================================================
# PHASE 2 SWEEP LAUNCHER — 3x3 grid over (W_EMISSION, W_SOC_SQUARED)
# ============================================================================
# Each array task fine-tunes one grid cell from the phase-1 best-seed
# checkpoint and runs full evaluation. After all tasks complete, run the
# selection script to pick the winning cell per algorithm.
#
# Edit ALGORITHM below before submitting; submit one job per algo.
# ============================================================================
CONDA_ENV="${CONDA_DEFAULT_ENV:-none}"
if [ "$CONDA_ENV" = "none" ]; then
    CONDA_ENV="(base)"
fi
PYTHON_VERSION=$(python -V 2>&1 | awk '{print $2}')
if [ -n "$SLURM_NODELIST" ]; then
    NODE_NAME="$SLURM_NODELIST"
else
    NODE_NAME=$(hostname)
fi
NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"
echo "============================================================================"
echo "PHASE 2 SWEEP — TASK $SLURM_ARRAY_TASK_ID"
echo "============================================================================"
echo "Conda Environment:     $CONDA_ENV"
echo "Python Version:        $PYTHON_VERSION"
echo "Assigned Node:         $NODE_NAME"
echo "Number of Threads:     $NUM_THREADS"
echo "Execution Start Time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"
echo ""
# ============================================================================
# Configuration — edit before submitting
# ============================================================================
ALGORITHM="ppo"
CELL_ID=$SLURM_ARRAY_TASK_ID
TOTAL_TIMESTEPS=4000000
DEVICE="auto"
echo "Algorithm:       $ALGORITHM"
echo "Cell ID:         $CELL_ID (of 0-8)"
echo "Total steps:     $TOTAL_TIMESTEPS"
echo "Device:          $DEVICE"
echo ""
# ============================================================================
# Run single-cell fine-tune + evaluation
# ============================================================================
python 02_rl_control/hyperparameter_search/run_phase2_cell.py \
    --algorithm "$ALGORITHM" \
    --cell_id "$CELL_ID" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --agent_device "$DEVICE"
echo ""
echo "Cell $CELL_ID finished at $(date '+%Y-%m-%d %H:%M:%S')"
