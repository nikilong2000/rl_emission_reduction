#!/bin/bash
#SBATCH --job-name=phase2_seeds
#SBATCH --time=72:00:00
#SBATCH --chdir=.
#SBATCH --array=0-9
#SBATCH --nodelist=jff111,jff116
#SBATCH --ntasks-per-core=1
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err

# ============================================================================
# PHASE-2 SEED VALIDATION LAUNCHER — 10 seeds with Optuna-best weights
# ============================================================================
# Each array task fine-tunes one seed (0-9) from the phase-1 best-seed
# checkpoint using the Optuna-chosen (w_emission, w_soc_squared), then runs
# full evaluation. Edit ALGORITHM per submission.
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
echo "PHASE-2 SEED VALIDATION — TASK $SLURM_ARRAY_TASK_ID"
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
PHASE2_CONFIG="02_rl_control/logs/${ALGORITHM}/phase2_optuna/best_params_phase2.json"
TOTAL_TIMESTEPS=4000000
DEVICE="auto"
SEED=$SLURM_ARRAY_TASK_ID
echo "Algorithm:       $ALGORITHM"
echo "Phase-2 config:  $PHASE2_CONFIG"
echo "Seed:            $SEED"
echo "Total steps:     $TOTAL_TIMESTEPS"
echo "Device:          $DEVICE"
echo ""
# ============================================================================
python 02_rl_control/hyperparameter_search/run_phase2_seed.py \
    --algorithm "$ALGORITHM" \
    --phase2_config "$PHASE2_CONFIG" \
    --seed "$SEED" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --agent_device "$DEVICE"
echo ""
echo "Seed $SEED finished at $(date '+%Y-%m-%d %H:%M:%S')"
