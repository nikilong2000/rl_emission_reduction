#!/bin/bash
#SBATCH --job-name=phase2_hpo_rerun
#SBATCH --time=24:00:00
#SBATCH --chdir=.
#SBATCH --nodelist=jff111,jff116
#SBATCH --ntasks-per-core=1
#SBATCH -o slurm_logs/%x_%A.out
#SBATCH -e slurm_logs/%x_%A.err

# ============================================================================
# PHASE-2 OPTUNA TRIAL RERUN — deterministic re-execution of a specific trial
# ============================================================================
# Use when a SLURM array task (commonly the 2nd) failed or was killed and the
# original trial's (w_emission, w_soc_squared) need to be retried exactly.
#
# The helper reads trial params from the JournalFileStorage and enqueues a
# new trial with identical params via study.enqueue_trial — bypasses TPE
# resampling, so the exact same configuration is run.
#
# Submit examples:
#   sbatch --export=ALL,ALGORITHM=ppo,TRIAL=1 02_rl_control/bash_scripts/submit_phase2_hpo_rerun.sh
#   sbatch --export=ALL,ALGORITHM=sac,TRIAL=3,MARK_FAILED=1 02_rl_control/bash_scripts/submit_phase2_hpo_rerun.sh
#   sbatch --export=ALL,ALGORITHM=td3,AUTO=1,MARK_FAILED=1 02_rl_control/bash_scripts/submit_phase2_hpo_rerun.sh
#
# Or edit the variables below and `sbatch 02_rl_control/bash_scripts/submit_phase2_hpo_rerun.sh`.
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
echo "PHASE-2 HPO RERUN"
echo "============================================================================"
echo "Conda Environment:  $CONDA_ENV"
echo "Python Version:     $PYTHON_VERSION"
echo "Assigned Node:      $NODE_NAME"
echo "Execution Start:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"
echo ""
# ============================================================================
# Configuration — set via --export=ALL,KEY=VAL or edit defaults
# ============================================================================
ALGORITHM="${ALGORITHM:-ppo}"      # ppo | sac | td3
TRIAL="${TRIAL:-}"                 # trial number to rerun (omit if AUTO=1)
AUTO="${AUTO:-0}"                  # 1 = rerun all non-COMPLETE trials
MARK_FAILED="${MARK_FAILED:-0}"    # 1 = mark original trial FAIL before rerun
N_ENVS="${N_ENVS:-8}"
TRIAL_TIMESTEPS="${TRIAL_TIMESTEPS:-4000000}"
LAMBDA_RMSE="${LAMBDA_RMSE:-20.0}"
LAMBDA_SOC="${LAMBDA_SOC:-1000.0}"
DEVICE="${DEVICE:-auto}"

echo "Algorithm:        $ALGORITHM"
echo "Trial:            ${TRIAL:-<auto>}"
echo "Auto-mode:        $AUTO"
echo "Mark-failed:      $MARK_FAILED"
echo "Subenvs:          $N_ENVS"
echo "Steps/trial:      $TRIAL_TIMESTEPS"
echo "lambda_rmse:      $LAMBDA_RMSE"
echo "lambda_soc:       $LAMBDA_SOC"
echo "Device:           $DEVICE"
echo ""

# ============================================================================
EXTRA_FLAGS=()
if [ "$AUTO" = "1" ]; then
    EXTRA_FLAGS+=(--auto)
else
    if [ -z "$TRIAL" ]; then
        echo "ERROR: TRIAL is empty and AUTO=0. Set TRIAL=<N> or AUTO=1."
        exit 1
    fi
    EXTRA_FLAGS+=(--trial "$TRIAL")
fi
if [ "$MARK_FAILED" = "1" ]; then
    EXTRA_FLAGS+=(--mark_failed)
fi

python 02_rl_control/hyperparameter_search/rerun_phase2_trial.py \
    --algorithm "$ALGORITHM" \
    --n_envs "$N_ENVS" \
    --trial_timesteps "$TRIAL_TIMESTEPS" \
    --lambda_rmse "$LAMBDA_RMSE" \
    --lambda_soc "$LAMBDA_SOC" \
    --agent_device "$DEVICE" \
    "${EXTRA_FLAGS[@]}"
echo ""
echo "Rerun finished at $(date '+%Y-%m-%d %H:%M:%S')"
