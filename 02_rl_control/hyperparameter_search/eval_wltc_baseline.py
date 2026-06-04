#!/usr/bin/env python3
"""
Batch-evaluate phase-2 RL agents (PPO, SAC) on the real-world WLTC / WLTC_high
drive cycles for the rule-based-controller comparison.

Each run is written to ``<seed_dir>/eval_<cycle>/`` so the existing staircase
evaluation outputs (``evaluation_metrics.json`` / ``evaluation_data.csv`` in the
seed dir) are never overwritten.

Notes
-----
- TD3 is intentionally excluded (too unstable for this comparison).
- ``random_target=False`` is passed explicitly so the phase-2 ``train_config``
  (which has ``random_target=True``) cannot override and force the staircase.
  With ``cycle`` set, the env loads the CSV speed trace deterministically.
- RL force-starts at SOC 0.7 in eval_mode (charge-sustaining); this is matched
  only against the rule-based isoSOC variant downstream.

Usage
-----
    # full sweep (2 algos x 10 seeds x 2 cycles = 40 runs)
    python eval_wltc_baseline.py

    # smoke test (single run)
    python eval_wltc_baseline.py --algos ppo --seeds 0 --cycles wltc
"""
import os
import sys
import json
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.dirname(THIS_DIR)  # 02_rl_control
sys.path.insert(0, os.path.join(RL_DIR, "models"))
sys.path.insert(0, RL_DIR)

from eval import evaluate_model  # noqa: E402  (models/eval.py)

LOGS = os.path.join(RL_DIR, "logs_cluster_phase2", "logs")
ALGOS = ["ppo", "sac"]
SEEDS = list(range(10))
CYCLES = ["wltc", "wltc_high"]


def seed_dir(algo, seed):
    return os.path.join(LOGS, algo, "phase2_seeds", f"seed_{seed}")


def run_one(algo, seed, cycle):
    sd = seed_dir(algo, seed)
    model_path = os.path.join(sd, f"{algo}_phase2_seed_{seed}_final.zip")
    if not os.path.exists(model_path):
        print(f"SKIP missing model: {model_path}")
        return False

    train_config = None
    cfg = os.path.join(sd, "train_config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            train_config = json.load(f)

    out = os.path.join(sd, f"eval_{cycle}")
    os.makedirs(out, exist_ok=True)
    print(f"\n>>> {algo.upper()} seed_{seed} cycle={cycle} -> {out}")

    try:
        evaluate_model(
            model_path,
            eval_log_dir=out,
            train_config=train_config,
            algorithm=algo,
            use_thermal=False,
            random_target=False,  # force deterministic CSV trace, ignore train_config
            target_speed=None,
            cycle=cycle,
        )
        return True
    except SystemExit as e:
        print(f"FAILED (SystemExit {e.code}): {algo} seed_{seed} {cycle}")
        return False
    except Exception as e:  # noqa: BLE001 - keep the batch alive
        import traceback

        traceback.print_exc()
        print(f"FAILED ({e}): {algo} seed_{seed} {cycle}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=["ppo", "sac"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--cycles", nargs="+", default=CYCLES, choices=["wltc", "wltc_high"])
    args = ap.parse_args()

    results = []
    for algo in args.algos:
        for seed in args.seeds:
            for cycle in args.cycles:
                ok = run_one(algo, seed, cycle)
                results.append((algo, seed, cycle, ok))

    n_ok = sum(1 for *_, ok in results if ok)
    print(f"\n==== DONE: {n_ok}/{len(results)} runs succeeded ====")
    for algo, seed, cycle, ok in results:
        if not ok:
            print(f"  FAIL  {algo} seed_{seed} {cycle}")


if __name__ == "__main__":
    main()
