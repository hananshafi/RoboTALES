import argparse
from omegaconf import OmegaConf
import os
import math
import cv2
import numpy as np
import torch
from einops import rearrange, repeat

from scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config

import open_clip

from sgm.data import VideoDataset, SINGLE_STAGE_TASK_DATASETS
import json
import pickle
from filelock import FileLock
import imageio
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
import time
from PIL import Image
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True

import pdb
from termcolor import colored
np.set_printoptions(precision=5, suppress=True)

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

def create_eval_env_modified(
    env_name,
    task_suite_name="libero_10",
):
    env_name = env_name.replace("_demo", "")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    # Build mapping from task name to task id
    name_to_id = {}
    for i in range(task_suite.n_tasks):
        task = task_suite.get_task(i)
        name_to_id[task.name] = i

    # Get the corresponding task id
    if env_name in name_to_id:
        task_id = name_to_id[env_name]
        task = task_suite.get_task(task_id)
        print(f"Retrieved task: {task.name}")
    else:
        raise ValueError(f"Task name '{env_name}' not found in task suite.")

    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(int(time.time()) & 0x00FF_FFFF )
    
    return env, task_description

def run_pred(model, value_dict, filter, shape, num_frames, num_pose_frames, action_dim, decoding_t, device):

    with torch.no_grad():
        with torch.autocast(device):

            extra_keys = ['cond_frames_2', 'cond_frames_3']

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
            
            vid_1 = vid[1:13]
            vid_2 = vid[13:25]

    return action_pred, vid_1, vid_2

def run_experiment(
    model,
    dataset,
    config,
    filter,
    device: str = "cuda",
):
    log_folder = f"experiments/{config['log_folder']}"
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

    max_traj_len = config.max_traj_len
    camera_names = ['eye_in_hand_rgb', 'agentview_rgb']
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

    while True:
        with lock:
            experiments_data = get_earliest_pending_experiments(
                experiment_record, config['number_of_experiments']
            )
            if experiments_data is None:
                print('All experiments are in progress or completed.')
                break
            experiments_data = set_all_status_to_in_progress(experiments_data)
            update_json_file(experiment_record, experiments_data)

        task_name = list(experiments_data['environments'].keys())[0]
        demos = list(experiments_data['environments'][task_name]['experiments'].keys())
        print(colored(f"=== Running task: {task_name} ({len(demos)} demos) ===", "cyan"))

        for demo in demos:

            demo_number = int(demo.replace("demo_", ""))
            env, task_description = create_eval_env_modified(
                env_name=task_name,
                task_suite_name=config.data.params.get("task_suite_name", "libero_10"),
            )

            env.reset()

            task_description = open_clip.tokenize([task_description]) # returns torch.Size([1, 77])

            video_path = f'{log_folder}/{task_name}_{demo}.mp4'
            video_writer = imageio.get_writer(video_path, fps=30)

            for _ in range(10):  # simulate the physics without any actions
                env.step(np.zeros(7))

            for i in range(int(max_traj_len/action_horizon)):

                obs = env.env._get_observations()

                image_0 = obs['robot0_eye_in_hand_image']
                image_1 = obs['robot0_eye_in_hand_image']
                image_2 = obs['agentview_image']

                value_dict = convert_observations(dataset=dataset,
                                                image_0=image_0,
                                                image_1=image_1,
                                                image_2=image_2,
                                                task_description=task_description,
                                                cond_aug=cond_aug,
                                                motion_bucket_id=motion_bucket_id,
                                                fps_id=fps_id,
                                                device=device)

                action_pred, vid_1, vid_2 = run_pred(model=model,
                                                    value_dict=value_dict,
                                                    filter=filter,
                                                    shape=shape,
                                                    num_frames=num_frames,
                                                    num_pose_frames=num_pose_frames,
                                                    action_dim=action_dim,
                                                    decoding_t=decoding_t,
                                                    device=device)

                action_pred = action_pred.detach().cpu().numpy()
                action_pred = action_pred[0:action_horizon]

                print(i)

                for step in range(action_pred.shape[0]):

                    obs, reward, done, info = env.step(action_pred[step])

                    video_img = np.concatenate(
                        [cv2.resize(obs['robot0_eye_in_hand_image'], (256, 256)), cv2.resize(obs['agentview_image'], (256, 256))], axis=1
                    )  # concatenate horizontally

                    if step < vid_1.shape[0]:
                        new_row = np.concatenate((vid_1[step], vid_2[step]), axis=1)
                        video_img = np.concatenate((video_img, new_row), axis=0)
                    else:
                        new_row = np.concatenate((vid_1[-1], vid_2[-1]), axis=1)
                        video_img = np.concatenate((video_img, new_row), axis=0)

                    video_writer.append_data(video_img)

                    if reward == 1.0:
                        break

                if reward == 1.0:
                    break

            experiments_data['environments'][task_name]['experiments'][demo]['status'] = 'done'
            if reward == 1.0:
                experiments_data['environments'][task_name]['experiments'][demo]['success'] = 1
            else:
                experiments_data['environments'][task_name]['experiments'][demo]['success'] = 0

            print(colored(f"Saved video to {video_path}", "green"))
            video_writer.close()

            with lock:
                update_json_file(experiment_record, experiments_data)


def convert_observations(dataset, image_0, image_1, image_2, task_description, cond_aug, motion_bucket_id, fps_id, device):
    
    image_0 = dataset.convert_frame(frame=image_0, size=(dataset.frame_width,dataset.frame_height), swap_rgb=dataset.swap_rgb)
    image_1 = dataset.convert_frame(frame=image_1, size=(dataset.frame_width,dataset.frame_height), swap_rgb=dataset.swap_rgb)
    image_2 = dataset.convert_frame(frame=image_2, size=(dataset.frame_width,dataset.frame_height), swap_rgb=dataset.swap_rgb)

    cond_frames = np.expand_dims(image_0, axis=0)
    cond_frames_2 = np.expand_dims(image_1, axis=0)
    cond_frames_3 = np.expand_dims(image_2, axis=0)

    cond_frames = (cond_frames + cond_aug * np.random.randn(*cond_frames.shape))
    cond_frames_2 = (cond_frames_2 + cond_aug * np.random.randn(*cond_frames_2.shape))
    cond_frames_3 = (cond_frames_3 + cond_aug * np.random.randn(*cond_frames_3.shape))

    cond_frames = torch.from_numpy(cond_frames.astype(np.float32)).to(device)
    cond_frames_2 = torch.from_numpy(cond_frames_2.astype(np.float32)).to(device)
    cond_frames_3 = torch.from_numpy(cond_frames_3.astype(np.float32)).to(device)

    value_dict = {}
    value_dict["cond_frames"] = cond_frames
    value_dict["cond_frames_2"] = cond_frames_2
    value_dict["cond_frames_3"] = cond_frames_3
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

def prepare_model(
    model_config: str,
    device: str = "cuda",
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    model, filter, config = load_model(
        model_config,
        device,
    )
    
    return model, filter, config    

def load_model(
    config: str,
    device: str,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    model = instantiate_from_config(config.model).to(device).eval()
    # model = None

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter, config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RoboCasa sampling experiment."
    )
    parser.add_argument(
        "-c", "--config",
        default="scripts/sampling/configs/svd_xt.yaml",
        help="Path to YAML configuration file."
    )
    args = parser.parse_args()

    # ---- load model & config chosen by the user ----
    model, filter, config = prepare_model(model_config=args.config)

    # dataset = None
    dataset = VideoDataset(
        n_frames=config.data.params.n_frames,
        cond_aug=config.data.params.cond_aug,
        motion_bucket_id=config.data.params.motion_bucket_id,
        fps_id=config.data.params.fps_id,
        frame_width=config.data.params.frame_width,
        frame_height=config.data.params.frame_height,
        tasks=config.data.params.tasks,
        skip_demos=config.data.params.skip_demos,
        sample_fps=config.data.params.sample_fps,
        video_fps=config.data.params.video_fps,
        pred_horizon=config.data.params.pred_horizon,
        aug=config.data.params.aug,
        action_dim=config.data.params.action_dim,
        swap_rgb=config.data.params.swap_rgb,
        mode=config.data.params.mode,
    )

    run_experiment(
        model=model,
        dataset=dataset,
        config=config,
        filter=filter,
    )
    