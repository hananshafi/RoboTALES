"""
Calculate per-task success rate from the planner experiment JSON files.

Reads both gpu0 and gpu1 experiment records, computes success rate
over completed demos (up to 50 per task), and prints a summary table.

Usage:
    python scripts/sampling/calculate_success_rate.py [--dir EXPERIMENTS_DIR]
"""

import argparse
import json
import os
from collections import OrderedDict


def load_experiment_record(json_path):
    """Load a single experiment record JSON and return {task_name: {demo: info}}."""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found, skipping.")
        return {}
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("environments", {})


def compute_success_rates(experiments_dir):
    """Compute success rates across both gpu0 and gpu1 JSON files."""

    gpu0_path = os.path.join(
        experiments_dir,
        "example_inference_our_method_25k_actions__gpu0_with_planner",
        "multi_environment_experiment_record.json",
    )
    gpu1_path = os.path.join(
        experiments_dir,
        "example_inference_our_method_25k_actions__gpu1_with_planner",
        "multi_environment_experiment_record.json",
    )

    # Merge all tasks from both files
    all_tasks = OrderedDict()

    for path in [gpu0_path, gpu1_path]:
        envs = load_experiment_record(path)
        for task_name, task_data in envs.items():
            if task_name not in all_tasks:
                all_tasks[task_name] = task_data["experiments"]
            else:
                all_tasks[task_name].update(task_data["experiments"])

    if not all_tasks:
        print("No experiment data found!")
        return

    # Compute per-task stats
    print("=" * 100)
    print(f"{'Task Name':<85} {'Done':>5} {'Succ':>5} {'Rate':>7}")
    print("=" * 100)

    total_done = 0
    total_success = 0

    for task_name, demos in all_tasks.items():
        done_demos = {
            k: v for k, v in demos.items()
            if v.get("status") == "done"
        }
        successes = sum(1 for v in done_demos.values() if v.get("success") == 1)
        num_done = len(done_demos)

        total_done += num_done
        total_success += successes

        rate = (successes / num_done * 100) if num_done > 0 else 0.0

        # Clean up display name (remove _demo suffix)
        display_name = task_name.replace("_demo", "")
        print(f"{display_name:<85} {num_done:>5} {successes:>5} {rate:>6.1f}%")

    print("=" * 100)
    overall_rate = (total_success / total_done * 100) if total_done > 0 else 0.0
    print(f"{'OVERALL':<85} {total_done:>5} {total_success:>5} {overall_rate:>6.1f}%")
    print("=" * 100)

    # Also show pending/in-progress counts
    print(f"\n{'Task Name':<85} {'Pend':>5} {'InPr':>5}")
    print("-" * 100)
    for task_name, demos in all_tasks.items():
        pending = sum(1 for v in demos.values() if v.get("status") == "pending")
        in_progress = sum(1 for v in demos.values() if v.get("status") == "in_progress")
        if pending > 0 or in_progress > 0:
            display_name = task_name.replace("_demo", "")
            print(f"{display_name:<85} {pending:>5} {in_progress:>5}")

    all_pending = sum(
        1 for demos in all_tasks.values()
        for v in demos.values() if v.get("status") == "pending"
    )
    all_in_progress = sum(
        1 for demos in all_tasks.values()
        for v in demos.values() if v.get("status") == "in_progress"
    )
    if all_pending > 0 or all_in_progress > 0:
        print(f"{'TOTAL remaining':<85} {all_pending:>5} {all_in_progress:>5}")
    else:
        print("All demos completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate task success rates from experiment JSONs.")
    parser.add_argument(
        "--dir",
        default="experiments",
        help="Path to experiments directory (default: experiments/)",
    )
    args = parser.parse_args()
    compute_success_rates(args.dir)
