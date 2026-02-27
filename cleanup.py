import os
import shutil
import json

LOGS_DIR = "02_rl_control/logs"
THRESHOLD = 100000


def cleanup_short_runs():
    if not os.path.exists(LOGS_DIR):
        print(f"Directory {LOGS_DIR} does not exist.")
        return

    deleted_count = 0
    kept_count = 0

    for run_dir in os.listdir(LOGS_DIR):
        full_path = os.path.join(LOGS_DIR, run_dir)

        # Ensure it's a directory
        if not os.path.isdir(full_path):
            continue

        metrics_file = os.path.join(full_path, "evaluation_metrics.json")

        # If no metrics file exists, the run likely crashed or is very new. Default to deleting.
        if not os.path.exists(metrics_file):
            print(f"Deleting incomplete run (no metrics file): {run_dir}")
            shutil.rmtree(full_path)
            deleted_count += 1
            continue

        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            # The exact path to total_timesteps depends on the version of the config tracking
            timesteps = None
            if (
                "configuration" in metrics
                and "total_timesteps" in metrics["configuration"]
            ):
                timesteps = metrics["configuration"]["total_timesteps"]
            elif "total_timesteps" in metrics:
                timesteps = metrics["total_timesteps"]

            # Delete if timesteps were found and are less than the threshold
            if timesteps is not None:
                if timesteps < THRESHOLD:
                    print(f"Deleting run {run_dir} (Timesteps: {timesteps})")
                    shutil.rmtree(full_path)
                    deleted_count += 1
                else:
                    print(f"Keeping run {run_dir} (Timesteps: {timesteps})")
                    kept_count += 1
            else:
                print(
                    f"Deleting run {run_dir} (Could not find total_timesteps in metrics)"
                )
                shutil.rmtree(full_path)
                deleted_count += 1

        except Exception as e:
            print(
                f"Error reading {metrics_file}: {e}. Deleting corrupted run: {run_dir}"
            )
            shutil.rmtree(full_path)
            deleted_count += 1

    print("\n--- Cleanup Summary ---")
    print(f"Deleted: {deleted_count} runs")
    print(f"Kept: {kept_count} runs")


if __name__ == "__main__":
    cleanup_short_runs()
