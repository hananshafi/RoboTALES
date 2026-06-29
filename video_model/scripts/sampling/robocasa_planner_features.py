import argparse
from omegaconf import OmegaConf
import os
import math
import cv2
import numpy as np
import torch
from einops import rearrange, repeat

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config

import open_clip

from sgm.data import VideoDataset, SINGLE_STAGE_TASK_DATASETS
import json
import pickle
from filelock import FileLock
import imageio
import robocasa.utils.dataset_registry
import robosuite
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True

import pdb
from termcolor import colored
np.set_printoptions(precision=5, suppress=True)

# Import the planner - adjust path as needed
import sys
sys.path.append('/bigdata/hanan/dev/videopolicy/video_model')
from videopolicy_planner import plan_steps, plan_task
from sgm.data.llm_planner import OnlinePlanner
from scripts.sampling.feature_extractor import BottleneckFeatureExtractor, FeatureLog

def create_environment_data_from_yaml(config, output_json_file):
    # Initialize data structure
    data = {"environments": {}}

    # Populate the data structure with experiments
    for env_name, env_details in config.data.params.tasks.items():
        num_experiments = env_details.get("num_experiments", 0)
        experiments = {
            f"demo_{i}": {"status": "pending", "success": -1}
            for i in range(num_experiments)
        }
        data["environments"][env_name] = {
            "experiments": experiments
        }

    # Write the JSON file
    with open(output_json_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"JSON file '{output_json_file}' created successfully.")


def get_earliest_pending_experiments(json_file, max_experiments):
    # Load the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    # Iterate through environments
    for env_name, env_details in data["environments"].items():
        # Filter pending experiments
        pending_experiments = {
            key: experiment
            for key, experiment in env_details["experiments"].items()
            if experiment["status"] == "pending"
        }

        # If there are pending experiments, build and return the structure
        if pending_experiments:
            # Limit to max_experiments
            limited_pending_experiments = dict(
                list(pending_experiments.items())[:max_experiments]
            )
            return {
                "environments": {
                    env_name: {
                        "experiments": limited_pending_experiments
                    }
                }
            }

    return None
def set_all_status_to_in_progress(data):
    # Iterate through environments
    for env in data["environments"].values():
        # Iterate through experiments within the environment
        for experiment in env["experiments"].values():
            # Update the status
            experiment["status"] = "in_progress"
    
    return data

def update_json_file(json_file, updated_data):
    with open(json_file, "r") as f:
        existing_data = json.load(f)

    def merge_dicts(source, target):

        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                merge_dicts(value, target[key])
            else:
                target[key] = value

    merge_dicts(updated_data, existing_data)

    with open(json_file, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"JSON file '{json_file}' updated successfully.")

def create_eval_env_modified(env_name, robots="PandaMobile", controllers="OSC_POSE", 
                           camera_names=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
                           camera_widths=256, camera_heights=256, seed=None,
                           obj_instance_split="B", generative_textures=None, randomize_cameras=False,
                           layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
                           controller_configs=None, id_selection=None):
    # controller_configs = load_controller_config(default_controller=controllers)   # This function is not found somehow

    layout_and_style_ids = (layout_and_style_ids[id_selection],)

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )

    env = robosuite.make(**env_kwargs)
    return env

def run_pred(model, value_dict, filter, shape, num_frames, num_pose_frames, action_dim, decoding_t, device):
    with torch.no_grad():
        with torch.autocast(device):

            extra_keys = ['cond_frames_2', 'cond_frames_3', 'cond_frames_4']

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner, extra_keys),
                value_dict,
                [1, num_frames],
                T=num_frames,
                T_p=num_pose_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            video_randn = torch.randn(shape, device=device)
            pose_randn = torch.randn((num_pose_frames, action_dim), device=device)

            noised_input = {
                'noised_video_input': video_randn,
                'noised_pose_input': pose_randn
            }

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]
            additional_model_inputs["num_pose_frames"] = batch["num_pose_frames"]

            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model, input, sigma, c, **additional_model_inputs
                )

            samples_output = model.sampler(denoiser, noised_input, cond=c, uc=uc)

            action_pred = samples_output['noised_pose_input']

            samples_z = samples_output['noised_video_input']
            model.en_and_decode_n_samples_a_time = decoding_t
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            samples = filter(samples)
            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            
            vid_1 = vid[1:9]
            vid_2 = vid[9:17]
            vid_3 = vid[17:25]

    return action_pred, vid_1, vid_2, vid_3

def run_experiment_with_planner(
    model,
    dataset,
    config,
    filter,
    device: str = "cuda",
    use_planner: bool = True,
    planner_api_key: str = None,
    extract_features: bool = False,
    feature_log: "FeatureLog" = None,
    feature_extractor: "BottleneckFeatureExtractor" = None,
    model_id: str = "ours",
):
    """
    Enhanced experiment that uses planner for task decomposition.

    If `extract_features` is True, captures the U-Net bottleneck activation at
    the first action-horizon iteration of each demo (initial-observation
    feature) and appends it to `feature_log` with task / category metadata.
    """
    log_folder = f"experiments/{config['log_folder']}_with_planner"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    OmegaConf.save(config, f'{log_folder}/inference_config.yaml')

    experiment_record = f"{log_folder}/multi_environment_experiment_record.json"
    experiment_record_lock = f"{experiment_record}.lock"
    lock = FileLock(experiment_record_lock)

    with lock:
        if os.path.exists(experiment_record):
            print("Experiment record already exists.")
        else:
            print("Experiment record does not exist.")
            create_environment_data_from_yaml(config, experiment_record)

        # Recover stale 'in_progress' demos from a prior crashed/killed run.
        # Single-process design: anything still in_progress at startup must be a
        # leftover, so flip it back to pending so it gets picked up below.
        with open(experiment_record, "r") as _f:
            _record = json.load(_f)
        _reset = 0
        for _env in _record["environments"].values():
            for _exp in _env["experiments"].values():
                if _exp.get("status") == "in_progress":
                    _exp["status"] = "pending"
                    _reset += 1
        if _reset:
            with open(experiment_record, "w") as _f:
                json.dump(_record, _f, indent=4)
            print(f"Recovered {_reset} stale in_progress demos -> pending.")

    # Outer loop: keep grabbing the next pending task until the record is exhausted.
    while True:
        with lock:
            experiments_data = get_earliest_pending_experiments(experiment_record, config['number_of_experiments'])
            if experiments_data is None:
                print('All experiments are in progress or completed.')
                return
            experiments_data = set_all_status_to_in_progress(experiments_data)
            update_json_file(experiment_record, experiments_data)

        task_name = list(experiments_data['environments'].keys())[0]

        # Fix the unpacking error - SINGLE_STAGE_TASK_DATASETS might have different structure
        for env_data in SINGLE_STAGE_TASK_DATASETS:
            if isinstance(env_data, tuple) and len(env_data) >= 2:
                env, _ = env_data
            else:
                env = env_data  # Handle case where it's not a tuple
            
            # Only update if 'human_path' exists
            if 'human_path' in SINGLE_STAGE_TASK_DATASETS.get(env, {}):
                SINGLE_STAGE_TASK_DATASETS[env]['human_path'] = "/bigdata/hanan/dev/videopolicy/video_model/" + SINGLE_STAGE_TASK_DATASETS[env]['human_path']

        with open(SINGLE_STAGE_TASK_DATASETS["ExampleEnvironmentData"], "rb") as pickle_file:
            environment_data = pickle.load(pickle_file)

        demos = list(experiments_data['environments'][task_name]['experiments'].keys())

        max_traj_len = config.max_traj_len
        camera_names = environment_data['env_kwargs']['camera_names']
        camera_height = config.data.params.frame_height
        camera_width = config.data.params.frame_width
        action_horizon = config.action_horizon
        cond_aug = config.data.params.cond_aug
        motion_bucket_id = config.data.params.motion_bucket_id
        fps_id = config.data.params.fps_id

        num_frames = config.model.params.sampler_config.params.guider_config.params.num_frames
        num_pose_frames = config.model.params.sampler_config.params.guider_config.params.num_pose_frames
        action_dim = config.data.params.action_dim
        decoding_t = config.decoding_t
        shape = (num_frames, 4, camera_width // 8, camera_height // 8)

        for demo in demos:
            demo_number = int(demo.replace("demo_", ""))
            env = create_eval_env_modified(
                env_name=task_name, 
                controller_configs=environment_data['env_kwargs']['controller_configs'], 
                id_selection=demo_number//10
            )

            env.reset()
            task_description = env.get_ep_meta()["lang"]
        
            # PLANNER INTEGRATION
            if use_planner:
                try:
                    print(f"🔧 Planning task: {task_description}")
                    # Use cache-aware OnlinePlanner (reads planner_cache.jsonl).
                    if not hasattr(run_experiment_with_planner, "_online_planner"):
                        run_experiment_with_planner._online_planner = OnlinePlanner()
                    planner_result = run_experiment_with_planner._online_planner.plan_steps(
                        task_description, mode='test'
                    )
                    subtasks = planner_result["steps"] if isinstance(planner_result, dict) else planner_result
                    print(f"📋 Planner generated {len(subtasks)} subtasks:")
                    for i, subtask in enumerate(subtasks):
                        print(f"   {i+1}. {subtask}")
                    current_subtask_index = 0
                    current_subtask = subtasks[current_subtask_index]
                except Exception as e:
                    print(f"❌ Planner failed: {e}. Using single-task mode.")
                    subtasks = [task_description]
                    current_subtask_index = 0
                    current_subtask = task_description
            else:
                subtasks = [task_description]
                current_subtask_index = 0
                current_subtask = task_description

            task_tokens = open_clip.tokenize([task_description]).to(device)

            video_path = f'{log_folder}/{task_name}_{demo}.mp4'
            video_writer = imageio.get_writer(video_path, fps=30)

            subtask_success_count = 0
        
            for i in range(int(max_traj_len/action_horizon)):
                # Update current subtask
                if use_planner and current_subtask_index < len(subtasks):
                    current_subtask = subtasks[current_subtask_index]
                    print(f"🎯 Executing subtask {current_subtask_index + 1}/{len(subtasks)}: {current_subtask}")

                video_img = []
                for cam_name in camera_names:
                    im = env.sim.render(height=camera_height, width=camera_width, camera_name=cam_name)[::-1]
                    video_img.append(im)

                image_0 = video_img[2]  # robot0_eye_in_hand
                image_1 = video_img[2]  # robot0_eye_in_hand  
                image_2 = video_img[0]  # robot0_agentview_left
                image_3 = video_img[1]  # robot0_agentview_right

                value_dict = convert_observations(
                    dataset=dataset,
                    image_0=image_0,
                    image_1=image_1,
                    image_2=image_2,
                    image_3=image_3,
                    task_description=task_tokens,
                    cond_aug=cond_aug,
                    motion_bucket_id=motion_bucket_id,
                    fps_id=fps_id,
                    device=device
                )
            
                # Reset hook buffer right before sampling so the captured activation
                # corresponds to this run_pred call (the final/lowest-sigma denoiser
                # invocation overwrites any earlier captures within the loop).
                if extract_features and feature_extractor is not None and i == 0:
                    feature_extractor.reset()

                # Use the existing run_pred function (planner integration can be enhanced later)
                action_pred, vid_1, vid_2, vid_3 = run_pred(
                    model=model,
                    value_dict=value_dict,
                    filter=filter,
                    shape=shape,
                    num_frames=num_frames,
                    num_pose_frames=num_pose_frames,
                    action_dim=action_dim,
                    decoding_t=decoding_t,
                    device=device
                )

                # Capture initial-observation feature once per demo.
                if extract_features and feature_extractor is not None and i == 0 and feature_log is not None:
                    feat = feature_extractor.pop()
                    feature_log.add(
                        feature=feat,
                        task_name=task_name,
                        demo_id=demo,
                        model_id=model_id,
                    )
            
                action_pred = ((action_pred.detach().cpu().numpy() + 1) / 2) * (dataset.max - dataset.min) + dataset.min
                action_pred = np.hstack((action_pred, [[0, 0, 0, 0, -1]] * action_pred.shape[0]))
                action_pred = action_pred[0:action_horizon]

                print(f"Step {i}, Subtask: {current_subtask}")

                subtask_success = False
            
                for step in range(action_pred.shape[0]):
                    obs, reward, done, info = env.step(action_pred[step])

                    # Simple subtask completion heuristic (you can enhance this)
                    if use_planner and _check_subtask_completion(env, current_subtask, step, i):
                        subtask_success = True
                        subtask_success_count += 1
                        print(f"✅ Subtask {current_subtask_index + 1} completed!")

                    # Video rendering
                    video_img = []
                    for cam_name in camera_names:
                        im = env.sim.render(height=camera_height, width=camera_width, camera_name=cam_name)[::-1]
                        video_img.append(im)
                    video_img = np.concatenate(video_img, axis=1)

                    if step < vid_1.shape[0]:
                        new_row = np.concatenate((vid_2[step], vid_3[step], vid_1[step]), axis=1)
                        video_img = np.concatenate((video_img, new_row), axis=0)
                    else:
                        new_row = np.concatenate((vid_2[-1], vid_3[-1], vid_1[-1]), axis=1)
                        video_img = np.concatenate((video_img, new_row), axis=0)

                    video_writer.append_data(video_img)

                    if env._check_success():
                        break
            
                # Move to next subtask if current one is completed
                if use_planner and subtask_success and current_subtask_index < len(subtasks) - 1:
                    current_subtask_index += 1
            
                if env._check_success():
                    break
        
            # Record results
            experiments_data['environments'][task_name]['experiments'][demo]['status'] = 'done'
            if env._check_success():
                experiments_data['environments'][task_name]['experiments'][demo]['success'] = 1
            else:
                experiments_data['environments'][task_name]['experiments'][demo]['success'] = 0
        
            # Add planner metrics
            if use_planner:
                experiments_data['environments'][task_name]['experiments'][demo]['planner_metrics'] = {
                    'subtasks_generated': len(subtasks),
                    'subtasks_completed': subtask_success_count,
                    'all_subtasks': subtasks
                }

            print(colored(f"Saved video to {video_path}", "green"))
            video_writer.close()

            with lock:
                update_json_file(experiment_record, experiments_data)

            # Force cleanup
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("GPU memory cleaned up")

def _check_subtask_completion(env, subtask_description, step, iteration):
    """
    Simple heuristic to check if a subtask is completed.
    Enhance this with environment-specific checks.
    """
    # Placeholder implementation - you should enhance this based on your environment
    # For now, we'll use a simple step-based completion
    completion_steps = 8  # Assume each subtask takes ~8 steps
    
    if "open" in subtask_description.lower() and "door" in subtask_description.lower():
        return step >= completion_steps
    elif "close" in subtask_description.lower() and "door" in subtask_description.lower():
        return step >= completion_steps  
    elif "turn on" in subtask_description.lower():
        return step >= completion_steps
    elif "turn off" in subtask_description.lower():
        return step >= completion_steps
    else:
        return step >= completion_steps

def convert_observations(dataset, image_0, image_1, image_2, image_3, task_description, cond_aug, motion_bucket_id, fps_id, device):
    image_0 = dataset.convert_frame(frame=image_0, swap_rgb=dataset.swap_rgb)
    image_1 = dataset.convert_frame(frame=image_1, swap_rgb=dataset.swap_rgb)
    image_2 = dataset.convert_frame(frame=image_2, swap_rgb=dataset.swap_rgb)
    image_3 = dataset.convert_frame(frame=image_3, swap_rgb=dataset.swap_rgb)

    cond_frames = np.expand_dims(image_0, axis=0)
    cond_frames_2 = np.expand_dims(image_1, axis=0)
    cond_frames_3 = np.expand_dims(image_2, axis=0)
    cond_frames_4 = np.expand_dims(image_3, axis=0)

    cond_frames = (cond_frames + cond_aug * np.random.randn(*cond_frames.shape))
    cond_frames_2 = (cond_frames_2 + cond_aug * np.random.randn(*cond_frames_2.shape))
    cond_frames_3 = (cond_frames_3 + cond_aug * np.random.randn(*cond_frames_3.shape))
    cond_frames_4 = (cond_frames_4 + cond_aug * np.random.randn(*cond_frames_4.shape))

    cond_frames = torch.from_numpy(cond_frames.astype(np.float32)).to(device)
    cond_frames_2 = torch.from_numpy(cond_frames_2.astype(np.float32)).to(device)
    cond_frames_3 = torch.from_numpy(cond_frames_3.astype(np.float32)).to(device)
    cond_frames_4 = torch.from_numpy(cond_frames_4.astype(np.float32)).to(device)

    value_dict = {}
    value_dict["cond_frames"] = cond_frames
    value_dict["cond_frames_2"] = cond_frames_2
    value_dict["cond_frames_3"] = cond_frames_3
    value_dict["cond_frames_4"] = cond_frames_4
    value_dict["cond_frames_without_noise"] = task_description
    value_dict["cond_aug"] = cond_aug
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id

    return value_dict
    
def get_unique_embedder_keys_from_conditioner(conditioner, extra_keys=None):
    unique_keys = list(set([x.input_key for x in conditioner.embedders]))
    if extra_keys:
        unique_keys.extend(extra_keys)
    return unique_keys

def get_batch(keys, value_dict, N, T, T_p, device):
    batch = {}
    batch_uc = {}

    for key in keys:    # keys are ['motion_bucket_id', 'cond_frames', 'cond_frames_without_noise', 'fps_id', 'cond_aug']
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        # elif key == "cond_frames":
        #     batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[1]   # I don't do repeating in the original script so text embedding needs to be repeated here
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    if T_p is not None:
        batch["num_pose_frames"] = T_p

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def prepare_model(model_config: str, device: str = "cuda"):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    model, filter, config = load_model(
        model_config,
        device,
    )
    
    return model, filter, config

def load_model(config: str, device: str):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter, config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RoboCasa sampling experiment with planner.")
    parser.add_argument(
        "-c", "--config",
        default="scripts/sampling/configs/svd_xt.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--use_planner", 
        action="store_true", 
        default=False,
        help="Use the VideoPolicy planner to break down tasks"
    )
    parser.add_argument(
        "--planner_api_key",
        type=str,
        default=None,
        help="Gemini API key for the planner (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--extract_features",
        action="store_true",
        default=False,
        help="Capture U-Net bottleneck features per demo and save .npz for UMAP."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="ours",
        help="Label for this run in the saved feature file (e.g. 'ours' or 'videopolicy')."
    )
    parser.add_argument(
        "--features_out",
        type=str,
        default=None,
        help="Path to .npz feature file. Defaults to <log_folder>/features_<model_id>.npz."
    )
    args = parser.parse_args()

    # Load model & config
    model, filter, config = prepare_model(model_config=args.config)

    dataset = VideoDataset(
        n_frames=config.data.params.n_frames,
        cond_aug=config.data.params.cond_aug,
        motion_bucket_id=config.data.params.motion_bucket_id,
        fps_id=config.data.params.fps_id,
        frame_width=config.data.params.frame_width,
        frame_height=config.data.params.frame_height,
        tasks=config.data.params.tasks,
        skip_demos=config.data.params.skip_demos,
        video_stride=config.data.params.video_stride,
        video_pred_horizon=config.data.params.video_pred_horizon,
        aug=config.data.params.aug,
        action_dim=config.data.params.action_dim,
        swap_rgb=config.data.params.swap_rgb,
        mode=config.data.params.mode,
    )

    feature_extractor = None
    feature_log = None
    if args.extract_features:
        feature_extractor = BottleneckFeatureExtractor(model)
        feature_log = FeatureLog()

    try:
        run_experiment_with_planner(
            model=model,
            dataset=dataset,
            config=config,
            filter=filter,
            use_planner=args.use_planner,
            planner_api_key=args.planner_api_key,
            extract_features=args.extract_features,
            feature_log=feature_log,
            feature_extractor=feature_extractor,
            model_id=args.model_id,
        )
    finally:
        if args.extract_features and feature_log is not None:
            log_folder = f"experiments/{config['log_folder']}_with_planner"
            features_out = args.features_out or os.path.join(
                log_folder, f"features_{args.model_id}.npz"
            )
            os.makedirs(os.path.dirname(features_out), exist_ok=True)
            feature_log.save(features_out)
        if feature_extractor is not None:
            feature_extractor.close()