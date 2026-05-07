"""
Identify the phase-1 seed with the lowest speed RMSE per algorithm.
Used as the starting checkpoint for phase-2 fine-tuning.

Usage:
    python select_best_seed.py --algorithm ppo
    python select_best_seed.py --algorithm ppo \
        --seeds_dir 02_rl_control/logs_cluster_01/logs/ppo/optuna/seeds
"""

import os
import sys
import json
import argparse
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
rl_control_dir = os.path.dirname(current_dir)


def find_best_seed(seeds_dir, algo_key):
    pattern = os.path.join(seeds_dir, "seed_*", "evaluation_metrics.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No evaluation_metrics.json found under {seeds_dir}")

    rows = []
    for f in files:
        with open(f) as fh:
            m = json.load(fh)
        seed_dir = os.path.dirname(f)
        seed_id = int(os.path.basename(seed_dir).replace("seed_", ""))
        model_zip = os.path.join(seed_dir, f"{algo_key}_seed_{seed_id}_final.zip")
        if not os.path.exists(model_zip):
            print(f"Warning: model zip missing for seed {seed_id}: {model_zip}")
            continue
        rows.append(
            {
                "seed": seed_id,
                "rmse_speed_kmph": m.get("rmse_speed_kmph", float("inf")),
                "total_nox_g": m.get("total_nox_g", float("inf")),
                "delta_soc": m.get("delta_soc", 0.0),
                "model_zip": model_zip,
                "seed_dir": seed_dir,
            }
        )
    if not rows:
        raise RuntimeError("No valid seeds with both metrics and model zip.")

    rows.sort(key=lambda r: (r["rmse_speed_kmph"], r["total_nox_g"]))
    return rows[0], rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=["ppo", "sac", "td3"])
    parser.add_argument(
        "--seeds_dir",
        default=None,
        help="Override seeds dir. Default: logs_cluster_01/logs/<algo>/optuna/seeds.",
    )
    parser.add_argument(
        "--write_baseline",
        action="store_true",
        help="Write baseline JSON next to seeds_dir for reuse.",
    )
    args = parser.parse_args()

    seeds_dir = args.seeds_dir or os.path.join(
        rl_control_dir,
        "logs_cluster_01",
        "logs",
        args.algorithm,
        "optuna",
        "seeds",
    )
    best, rows = find_best_seed(seeds_dir, args.algorithm)

    print(f"=== Phase-1 seed comparison for {args.algorithm.upper()} ===")
    print(f"{'seed':<6}{'rmse':>10}{'nox_g':>10}{'dSOC':>10}")
    for r in rows:
        print(
            f"{r['seed']:<6}"
            f"{r['rmse_speed_kmph']:>10.3f}"
            f"{r['total_nox_g']:>10.3f}"
            f"{r['delta_soc']:>10.4f}"
        )
    print()
    print(f"Best seed: {best['seed']}")
    print(f"  rmse_speed_kmph = {best['rmse_speed_kmph']:.4f}")
    print(f"  total_nox_g     = {best['total_nox_g']:.4f}")
    print(f"  delta_soc       = {best['delta_soc']:.4f}")
    print(f"  model_zip       = {best['model_zip']}")

    if args.write_baseline:
        out = os.path.join(seeds_dir, "phase1_best_seed.json")
        with open(out, "w") as f:
            json.dump(best, f, indent=4)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
