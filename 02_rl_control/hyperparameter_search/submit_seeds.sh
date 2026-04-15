#!/bin/bash
#SBATCH --job-name=hpo_seeds
#SBATCH --time=06:00:00
#SBATCH --chdir=.
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --array=0-9
#SBATCH -o seed_%a_%x_%j.out
#SBATCH -e seed_%a_%x_%j.err
# ============================================================================
# SEED VALIDATION LAUNCHER — 10 seeds with best HPO config
# ============================================================================
# Each array task trains one seed (0–9) with best_params.json, then runs
# full evaluation.  After all tasks complete, run plot_seeds.py manually:
#
#   python 02_rl_control/hyperparameter_search/plot_seeds.py \
#       --results_dir 02_rl_control/logs/<ALGO>/optuna/seeds/ \
#       --algorithm <ALGO>
# ============================================================================
# --- Environment activation (uncomment the one you use) ---
# conda activate your_env_name
# source /path/to/.venv/bin/activate
# --- System info header ---
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
echo "SEED VALIDATION — TASK $SLURM_ARRAY_TASK_ID"
echo "============================================================================"
echo "Conda Environment:     $CONDA_ENV"
echo "Python Version:        $PYTHON_VERSION"
echo "Assigned Node:         $NODE_NAME"
echo "Number of Threads:     $NUM_THREADS"
echo "Execution Start Time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"
echo ""
# ============================================================================
# Configuration — edit these before submitting
# ============================================================================
ALGORITHM="ppo"
BEST_CONFIG="02_rl_control/logs/${ALGORITHM}/optuna/best_params.json"
TOTAL_TIMESTEPS=4000000
DEVICE="auto"
SEED=$SLURM_ARRAY_TASK_ID
echo "Algorithm:       $ALGORITHM"
echo "Config:          $BEST_CONFIG"
echo "Seed:            $SEED"
echo "Total steps:     $TOTAL_TIMESTEPS"
echo "Device:          $DEVICE"
echo ""
# ============================================================================
# Run single-seed training + evaluation
# ============================================================================
python 02_rl_control/hyperparameter_search/run_seeds.py \
    --algorithm "$ALGORITHM" \
    --config "$BEST_CONFIG" \
    --seed "$SEED" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --agent_device "$DEVICE"
echo ""
echo "Seed $SEED finished at $(date '+%Y-%m-%d %H:%M:%S')"