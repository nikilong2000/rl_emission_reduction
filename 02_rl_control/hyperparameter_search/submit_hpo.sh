#!/bin/bash
#SBATCH --job-name=hpo_tune
#SBATCH --time=48:00:00
#SBATCH --chdir=.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10
#SBATCH --exclusive
#SBATCH -o hpo_%x_%A_%a.out
#SBATCH -e hpo_%x_%A_%a.err
# ============================================================================
# HPO LAUNCHER — Optuna hyperparameter search
# ============================================================================
# NOTE: Study uses JournalFileStorage (append-only log, crash-safe).
#       If this job times out, simply resubmit — Optuna resumes automatically
#       via load_if_exists=True.
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
echo "HPO JOB EXECUTION INFORMATION"
echo "============================================================================"
echo "Conda Environment:     $CONDA_ENV"
echo "Python Version:        $PYTHON_VERSION"
echo "Assigned Node:         $NODE_NAME"
echo "Number of Threads:     $NUM_THREADS"
echo "Execution Start Time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"
echo ""
# Configuration — edit these before submitting
# ============================================================================
ALGORITHM="ppo"           # ppo | sac | td3
N_TRIALS=5                # trials PER NODE. Total = N_TRIALS * SLURM_ARRAY_TASK_COUNT
N_ENVS=8                  # parallel envs per trial (match cpus-per-task)
TRIAL_TIMESTEPS=4000000   # training steps per trial
DEVICE="cpu"             # cpu | cuda | auto
echo "Algorithm:       $ALGORITHM"
echo "Trials/Node:     $N_TRIALS"
echo "Subenvs:         $N_ENVS"
echo "Steps/trial:     $TRIAL_TIMESTEPS"
echo "Device:          $DEVICE"
echo ""
# ============================================================================
# Run HPO
# ============================================================================
python 02_rl_control/hyperparameter_search/tune_hpo.py \
    --algorithm "$ALGORITHM" \
    --n_trials "$N_TRIALS" \
    --n_envs "$N_ENVS" \
    --trial_timesteps "$TRIAL_TIMESTEPS" \
    --agent_device "$DEVICE" \
    --n_jobs 1
echo ""
echo "HPO finished at $(date '+%Y-%m-%d %H:%M:%S')"