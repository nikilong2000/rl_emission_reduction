"""
Regenerate all_trials.csv + best_params_phase2.json from a phase-2 Optuna
journal — WITHOUT running any training.

Needed because rerun_phase2_trial.py appends trials to the journal but does
not rewrite the CSV / best-params files (only tune_phase2_hpo.main() does, at
the end of a full study.optimize). After reruns, run this to refresh outputs.

Usage:
    # On cluster (default logs/<algo>/phase2_optuna)
    python export_phase2_study.py --algorithm sac

    # On local, against pulled results
    python export_phase2_study.py --algorithm sac \
        --logs_dir 02_rl_control/logs_cluster/logs/sac/phase2_optuna
"""

import os
import sys
import json
import argparse

import optuna
import pandas as pd
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.trial import TrialState

current_dir = os.path.dirname(os.path.abspath(__file__))
rl_control_dir = os.path.dirname(current_dir)


def _load_study(algo_key, logs_dir=None, study_name=None):
    base_log_dir = logs_dir or os.path.join(
        rl_control_dir, "logs", algo_key, "phase2_optuna"
    )
    journal_path = os.path.join(base_log_dir, "study_journal.log")
    if not os.path.exists(journal_path):
        sys.exit(f"No journal at {journal_path}.")
    storage = JournalStorage(JournalFileStorage(journal_path))
    name = study_name or f"{algo_key}_phase2_hpo"
    study = optuna.load_study(study_name=name, storage=storage)
    return study, base_log_dir


def export(study, base_log_dir):
    rows = []
    for t in study.trials:
        if t.datetime_start and t.datetime_complete:
            duration_s = (t.datetime_complete - t.datetime_start).total_seconds()
        else:
            duration_s = None
        row = {
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "datetime_start": t.datetime_start,
            "datetime_complete": t.datetime_complete,
            "duration_s": duration_s,
        }
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)
    csv_path = os.path.join(base_log_dir, "all_trials.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {len(rows)} rows -> {csv_path}")

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed:
        best = min(completed, key=lambda t: t.value)
        best_path = os.path.join(base_log_dir, "best_params_phase2.json")
        with open(best_path, "w") as f:
            json.dump(best.params, f, indent=4)
        print(f"Best trial {best.number} (score={best.value:.4f}) -> {best_path}")
    else:
        print("No COMPLETE trials — best_params_phase2.json not written.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument("--logs_dir", default=None)
    parser.add_argument("--study_name", default=None)
    args = parser.parse_args()

    study, base_log_dir = _load_study(args.algorithm, args.logs_dir, args.study_name)

    from collections import Counter

    states = Counter(str(t.state).replace("TrialState.", "") for t in study.trials)
    valid = states.get("COMPLETE", 0) + states.get("PRUNED", 0)
    print(f"\n{args.algorithm.upper()} study: {len(study.trials)} trials  {dict(states)}")
    print(f"  valid (COMPLETE+PRUNED) = {valid}")
    running = [t.number for t in study.trials if t.state == TrialState.RUNNING]
    if running:
        print(f"  WARNING: still RUNNING (zombies) = {running} — rerun before trusting results")
    print()
    export(study, base_log_dir)


if __name__ == "__main__":
    main()
