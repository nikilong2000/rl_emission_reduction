"""
Re-run a specific phase-2 Optuna trial deterministically.

When a SLURM array task fails (job killed, RUNNING-stuck, exception inside
objective), the trial keeps its sampled (w_emission, w_soc_squared) in the
JournalFileStorage. This script:
  1. Loads the existing study.
  2. Reads trial N's params from the journal.
  3. (Optional) Marks the original trial as FAIL so it stops showing as
     RUNNING zombie.
  4. Enqueues a fresh trial with identical params and calls
     study.optimize(n_trials=1) — produces a NEW trial number with the
     exact same reward weights. TPE sampler is bypassed for that one trial.

Usage:
    # Rerun the second trial of the PPO phase-2 study
    python rerun_phase2_trial.py --algorithm ppo --trial 1

    # Rerun all RUNNING zombie trials (killed/timed-out jobs only)
    python rerun_phase2_trial.py --algorithm ppo --auto

    # Also include FAIL trials (exceptions), not just RUNNING zombies
    python rerun_phase2_trial.py --algorithm ppo --auto --include_failed

    # Also mark the original as FAIL to clean the journal
    python rerun_phase2_trial.py --algorithm ppo --trial 1 --mark_failed

NOTE: --auto NEVER targets PRUNED trials. PRUNED is a deliberate MedianPruner
decision (underperforming trial stopped early) and is a valid study outcome —
re-running it would waste compute and pollute the study.
"""

import os
import sys
import json
import argparse

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.trial import TrialState

current_dir = os.path.dirname(os.path.abspath(__file__))
rl_control_dir = os.path.dirname(current_dir)
models_dir = os.path.join(rl_control_dir, "models")
sys.path.insert(0, rl_control_dir)
sys.path.insert(0, models_dir)
sys.path.insert(0, current_dir)

from select_best_seed import find_best_seed
from tune_phase2_hpo import objective


def _load_study(algo_key, study_name=None, logs_dir=None):
    base_log_dir = logs_dir or os.path.join(
        rl_control_dir, "logs", algo_key, "phase2_optuna"
    )
    journal_path = os.path.join(base_log_dir, "study_journal.log")
    if not os.path.exists(journal_path):
        sys.exit(f"No journal at {journal_path}. Has phase-2 HPO ever run?")
    storage = JournalStorage(JournalFileStorage(journal_path))
    name = study_name or f"{algo_key}_phase2_hpo"
    study = optuna.load_study(study_name=name, storage=storage)
    return study, base_log_dir


def _resolve_setup(args):
    """Mirrors tune_phase2_hpo.main prelude — locate phase-1 ckpt + HPO config."""
    seeds_dir = args.seeds_dir or os.path.join(
        rl_control_dir,
        "logs_cluster_phase1",
        "logs",
        args.algorithm,
        "optuna",
        "seeds",
    )
    phase1_best, _ = find_best_seed(seeds_dir, args.algorithm)
    print(
        f"Phase-1 best seed: {phase1_best['seed']} "
        f"(rmse={phase1_best['rmse_speed_kmph']:.3f}, "
        f"nox={phase1_best['total_nox_g']:.3f}g)"
    )
    hpo_config_path = args.hpo_config or os.path.join(
        rl_control_dir,
        "logs_cluster_phase1",
        "logs",
        args.algorithm,
        "optuna",
        "best_params.json",
    )
    with open(hpo_config_path) as f:
        hpo_overrides = json.load(f)
    print(f"Applied HPO overrides from {hpo_config_path}")
    return phase1_best, hpo_overrides


def _print_trial_table(study):
    print(f"\nExisting trials in study '{study.study_name}':")
    print(f"  {'num':>4} {'state':<12} {'value':>10} {'w_emission':>12} {'w_soc_squared':>15}")
    for t in study.trials:
        v = f"{t.value:.4f}" if t.value is not None else "—"
        we = t.params.get("w_emission")
        wsq = t.params.get("w_soc_squared")
        print(
            f"  {t.number:>4} {str(t.state).replace('TrialState.', ''):<12} "
            f"{v:>10} {we if we is None else f'{we:.4f}':>12} "
            f"{wsq if wsq is None else f'{wsq:.2f}':>15}"
        )


def _mark_failed(study, trial_number):
    target = next((t for t in study.trials if t.number == trial_number), None)
    if target is None:
        return
    if target.state == TrialState.COMPLETE:
        print(f"Trial {trial_number} already COMPLETE — not marking failed.")
        return
    study._storage.set_trial_state_values(target._trial_id, TrialState.FAIL)
    print(f"Marked trial {trial_number} as FAIL (was {target.state}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trial", type=int, help="Rerun this specific trial number.")
    group.add_argument(
        "--auto",
        action="store_true",
        help="Rerun all RUNNING zombie trials (killed jobs). Never touches PRUNED.",
    )
    group.add_argument(
        "--list", action="store_true", help="Just list all trials and exit."
    )
    parser.add_argument(
        "--include_failed",
        action="store_true",
        help="In --auto mode, also rerun FAIL trials (not just RUNNING). PRUNED still excluded.",
    )
    parser.add_argument(
        "--mark_failed",
        action="store_true",
        help="Mark original trial state=FAIL before re-enqueue (cleans RUNNING zombies).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --trial: rerun even if the trial is PRUNED or COMPLETE.",
    )
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--trial_timesteps", type=int, default=4_000_000)
    parser.add_argument("--lambda_rmse", type=float, default=20.0)
    parser.add_argument("--lambda_soc", type=float, default=1000.0)
    parser.add_argument("--agent_device", default="auto")
    parser.add_argument("--seeds_dir", default=None)
    parser.add_argument("--hpo_config", default=None)
    parser.add_argument("--use_thermal", action="store_true", default=False)
    parser.add_argument("--study_name", default=None)
    parser.add_argument(
        "--logs_dir",
        default=None,
        help="Override phase-2 Optuna log dir. Default: logs/<algo>/phase2_optuna. "
        "Set to logs_cluster/logs/<algo>/phase2_optuna for local inspection of pulled results.",
    )
    args = parser.parse_args()

    study, base_log_dir = _load_study(args.algorithm, args.study_name, args.logs_dir)
    _print_trial_table(study)
    if args.list:
        return

    if args.auto:
        # Only RUNNING zombies (killed/timed-out jobs). PRUNED trials are
        # deliberate MedianPruner outcomes and must never be re-run.
        targets = [
            t.number
            for t in study.trials
            if t.state == TrialState.RUNNING and t.params
        ]
        if args.include_failed:
            targets += [
                t.number
                for t in study.trials
                if t.state == TrialState.FAIL and t.params
            ]
            targets = sorted(set(targets))
        if not targets:
            print("\nNo RUNNING zombie trials to rerun.")
            return
        kinds = "RUNNING+FAIL" if args.include_failed else "RUNNING"
        print(f"\nAuto-rerun targets ({kinds} only, PRUNED excluded): {targets}")
    else:
        targets = [args.trial]

    phase1_best, hpo_overrides = _resolve_setup(args)

    for trial_num in targets:
        match = next((t for t in study.trials if t.number == trial_num), None)
        if match is None:
            print(f"\nTrial {trial_num} not in study — skipping.")
            continue
        if not match.params:
            print(
                f"\nTrial {trial_num} has no recorded params "
                "(crashed before suggest_*) — cannot rerun deterministically."
            )
            continue
        if match.state == TrialState.PRUNED and not args.force:
            print(
                f"\nTrial {trial_num} is PRUNED (deliberate MedianPruner outcome) — "
                "skipping. Pass --force to rerun anyway."
            )
            continue
        if match.state == TrialState.COMPLETE and not args.force:
            print(
                f"\nTrial {trial_num} is already COMPLETE — skipping. "
                "Pass --force to rerun anyway."
            )
            continue
        params = dict(match.params)
        print(
            f"\n=== Re-enqueueing trial {trial_num} with params {params} "
            f"(was state={match.state}) ==="
        )
        if args.mark_failed:
            _mark_failed(study, trial_num)
        study.enqueue_trial(params)

        # Run exactly one trial — Optuna pulls from queue first.
        study.optimize(
            lambda t: objective(t, args.algorithm, args, base_log_dir, phase1_best, hpo_overrides),
            n_trials=1,
        )

    # Refresh all_trials.csv + best_params_phase2.json from the journal so the
    # exported files reflect the reruns (study.optimize alone does not write them).
    try:
        from export_phase2_study import export

        export(study, base_log_dir)
    except Exception as e:
        print(
            f"WARNING: could not refresh CSV/best-params ({e}). "
            "Run export_phase2_study.py manually."
        )


if __name__ == "__main__":
    main()
