import json
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python eval_script.py <results_file.json>")
        sys.exit(1)

    filename = sys.argv[1]

    # --- Step 1: Load JSON ---
    with open(filename, "r") as f:
        data = json.load(f)

    task_success_rates = {}
    overall_sum = 0
    num_tasks = 0

    # --- Step 2: Iterate through each environment/task ---
    for env_name, env_data in data["environments"].items():
        experiments = env_data["experiments"]

        total_success = 0
        total_demos = 0

        for _, demo_data in experiments.items():
            if demo_data["status"] == "done":
                total_success += demo_data["success"]

            total_demos += 1  # count all demos

        success_rate = total_success / total_demos if total_demos > 0 else 0.0

        task_success_rates[env_name] = success_rate

        overall_sum += success_rate
        num_tasks += 1

    # --- Step 3: Compute overall average ---
    overall_avg = overall_sum / num_tasks if num_tasks > 0 else 0.0

    # --- Step 4: Print results ---
    print("\n=== Success Rates ===")
    for task, rate in task_success_rates.items():
        print(f"{task}: {rate:.3f}")

    print(f"\nOverall Average Success Rate over all tasks: {overall_avg:.3f}")


if __name__ == "__main__":
    main()