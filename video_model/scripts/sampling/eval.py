import json
import traceback
import os

def calculate_and_write_rates(input_file_path, output_file_path):
    try:
        with open(input_file_path, "r") as f:
            data = json.load(f)
            
        environments = data.get("environments", {})
        
        results = []         
        for task_name, task_data in environments.items():
            experiments = task_data.get("experiments", {})
            
            successful_demos = 0
            completed_demos = 0 
            total_demos_in_task = len(experiments)
            
            for demo_name, demo_results in experiments.items():
                success_status = demo_results.get("success", -1)
                
                if success_status == 1:
                    successful_demos += 1
                    completed_demos += 1
                elif success_status == 0:
                    completed_demos += 1
            
            if completed_demos > 0:
                success_rate = successful_demos / completed_demos
            else:
                success_rate = None
        
            result_data = {
                "task_name": task_name,
                "success_rate_percent": success_rate,
                "total_demos": total_demos_in_task,
                "successful_demos": successful_demos,
                "completed_demos": completed_demos,
            }
            
            results.append(result_data)
            
        with open(output_file_path, "w") as f:
            for r in results:
                    f.write(json.dumps(r) + "\n")        
    except Exception as e:
        traceback.print_exc()
        
if __name__ == "__main__":
    models = ["example_inference_planner_40k_step", "example_inference_planner_plus_ddpo_40k_step", "example_inference"]
    # input_filename = "/home/hanan/dev/videopolicy/video_model/experiments/example_inference_planner_40k_step/multi_environment_experiment_record.json"
    # output_filename = "/home/hanan/dev/videopolicy/video_model/experiments/example_inference_planner_40k_step/eval.jsonl"
    for model in models:
        if model == "example_inference_planner_plus_ddpo_40k_step":
            input_filename = os.path.join("/home/hanan/dev/videopolicy/video_model/experiments", model, "machine2_ours_40k_experiment_record.json.json")
        else:
            input_filename = os.path.join("/home/hanan/dev/videopolicy/video_model/experiments", model, "multi_environment_experiment_record.json")
        output_filename = os.path.join("/home/hanan/dev/videopolicy/video_model/experiments", model, "eval.jsonl")
        calculate_and_write_rates(input_filename, output_filename)