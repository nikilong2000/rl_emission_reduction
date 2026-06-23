#!/bin/bash
#SBATCH --job-name=phase2_rerun_sac
#SBATCH --time=72:00:00
#SBATCH --chdir=.
#SBATCH --ntasks-per-core=1
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%A.out
#SBATCH -e slurm_logs/%x_%A.err

# ============================================================================
# SAC PHASE-2 HPO RERUN — rerun every non-COMPLETE trial in the SAC study
# ============================================================================
# Reads journal at logs/sac/phase2_optuna/study_journal.log.
# For each RUNNING/FAIL trial: marks it FAIL, enqueues new trial with same
# (w_emission, w_soc_squared), runs fine-tune + eval. Sequential loop within
# this single job.
#
# Submit:
#   sbatch 02_rl_control/bash_scripts/submit_phase2_hpo_rerun_sac.sh
#
# Note: --requeue means SLURM auto-resubmits if the job dies with NODE_FAIL.
# To disable nodelist constraint, leave it unset. If you want specific nodes,
# add #SBATCH --nodelist=... above.
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
echo "============================================================================"
echo "PHASE-2 HPO RERUN — SAC"
echo "============================================================================"
echo "Conda Environment:  $CONDA_ENV"
echo "Python Version:     $PYTHON_VERSION"
echo "Assigned Node:      $NODE_NAME"
echo "Execution Start:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"
echo ""
# ============================================================================
# Hardcoded recovery config — edit only if defaults need changing
# ============================================================================
ALGORITHM="sac"
N_ENVS=20
TRIAL_TIMESTEPS=4000000
LAMBDA_RMSE=20.0
LAMBDA_SOC=1000.0
DEVICE="auto"

echo "Algorithm:        $ALGORITHM"
echo "Mode:             --auto --mark_failed"
echo "Subenvs:          $N_ENVS"
echo "Steps/trial:      $TRIAL_TIMESTEPS"
echo "lambda_rmse:      $LAMBDA_RMSE"
echo "lambda_soc:       $LAMBDA_SOC"
echo "Device:           $DEVICE"
echo ""
# ============================================================================
python 02_rl_control/hyperparameter_search/rerun_phase2_trial.py \
    --algorithm "$ALGORITHM" \
    --auto \
    --mark_failed \
    --n_envs "$N_ENVS" \
    --trial_timesteps "$TRIAL_TIMESTEPS" \
    --lambda_rmse "$LAMBDA_RMSE" \
    --lambda_soc "$LAMBDA_SOC" \
    --agent_device "$DEVICE"
echo ""
echo "SAC rerun finished at $(date '+%Y-%m-%d %H:%M:%S')"
