#!/bin/bash
#SBATCH --job-name=phase2_hpo
#SBATCH --time=72:00:00
#SBATCH --chdir=.
#SBATCH --array=1-10
#SBATCH --nodelist=jff111,jff116
#SBATCH --ntasks-per-core=1
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err

# ============================================================================
# PHASE-2 OPTUNA HPO LAUNCHER — refine (W_EMISSION, W_SOC_SQUARED)
# ============================================================================
# JournalFileStorage is append-only and crash-safe. Resubmit on timeout —
# Optuna resumes via load_if_exists=True. Edit ALGORITHM per submission.
# Default: 10 array tasks * 2 trials/task = 20 trials per algo.
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
echo "PHASE-2 HPO — TASK $SLURM_ARRAY_TASK_ID"
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
ALGORITHM="ppo"            # ppo | sac | td3
N_TRIALS=2                 # trials per array task; total = N_TRIALS * array size
N_ENVS=8                   # parallel envs per trial
TRIAL_TIMESTEPS=4000000    # fine-tune steps per trial
LAMBDA_RMSE=20.0           # RMSE-overshoot penalty constant
LAMBDA_SOC=1000.0          # SOC-overshoot penalty constant
DEVICE="auto"
echo "Algorithm:        $ALGORITHM"
echo "Trials/task:      $N_TRIALS"
echo "Subenvs:          $N_ENVS"
echo "Steps/trial:      $TRIAL_TIMESTEPS"
echo "lambda_rmse:      $LAMBDA_RMSE"
echo "lambda_soc:       $LAMBDA_SOC"
echo "Device:           $DEVICE"
echo ""
# ============================================================================
python 02_rl_control/hyperparameter_search/tune_phase2_hpo.py \
    --algorithm "$ALGORITHM" \
    --n_trials "$N_TRIALS" \
    --n_envs "$N_ENVS" \
    --trial_timesteps "$TRIAL_TIMESTEPS" \
    --lambda_rmse "$LAMBDA_RMSE" \
    --lambda_soc "$LAMBDA_SOC" \
    --agent_device "$DEVICE" \
    --n_jobs 1
echo ""
echo "Phase-2 HPO task $SLURM_ARRAY_TASK_ID finished at $(date '+%Y-%m-%d %H:%M:%S')"
