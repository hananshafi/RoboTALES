import argparse
from omegaconf import OmegaConf
import os
import math
import numpy as np
import torch
from einops import rearrange, repeat

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.util import instantiate_from_config

import open_clip

import sys
sys.path.insert(0, "/home/hanan/dev/robosuite_libero")

from sgm.data.video_libero import VideoDataset
from libero.libero import benchmark
from libero.libero.utils import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import json
from filelock import FileLock
import imageio
from ruamel.yaml import YAML
from termcolor import colored

yaml = YAML()
yaml.preserve_quotes = True

np.set_printoptions(precision=5, suppress=True)

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_10"]()

# map task.name -> id
NAME2ID = {task_suite.get_task(i).name: i for i in range(task_suite.n_tasks)}


def create_environment_data_from_yaml(config, output_json_file):
    data = {"environments": {}}
    for env_name, env_details in config.data.params.tasks.items():
        num_experiments = env_details.get("num_experiments", 0)
        experiments = {f"demo_{i}": {"status": "pending", "success": -1} for i in range(num_experiments)}
        data["environments"][env_name] = {"experiments": experiments}

    with open(output_json_file, "w") as f:
        json.dump(data, f, indent=4)
    print(f"JSON file '{output_json_file}' created successfully.")


def get_earliest_pending_experiments(json_file, max_experiments):
    with open(json_file, "r") as f:
        data = json.load(f)

    for env_name, env_details in data["environments"].items():
        pending = {k: v for k, v in env_details["experiments"].items() if v["status"] == "pending"}
        if pending:
            limited = dict(list(pending.items())[:max_experiments])
            return {"environments": {env_name: {"experiments": limited}}}
    return None


def set_all_status_to_in_progress(data):
    for env in data["environments"].values():
        for experiment in env["experiments"].values():
            experiment["status"] = "in_progress"
    return data


def update_json_file(json_file, updated_data):
    with open(json_file, "r") as f:
        existing = json.load(f)

    def merge_dicts(source, target):
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                merge_dicts(value, target[key])
            else:
                target[key] = value

    merge_dicts(updated_data, existing)

    with open(json_file, "w") as f:
        json.dump(existing, f, indent=4)
    print(f"JSON file '{json_file}' updated successfully.")


def create_libero_env(task_suite_name="libero_10", task_id=0,
                      camera_height=256, camera_width=256,
                      seed=0, init_state_id=0):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_description = task.language

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=camera_height,
        camera_widths=camera_width,
    )
    env.seed(seed)

    obs = env.reset()
    print("OBS KEYS:", sorted(list(obs.keys())))

    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[init_state_id])

    try:
        obs = env.get_observation()
    except Exception:
        pass

    return env, task_description, task_suite, obs


# -------------------------
# 2-view ONLY changes start
# -------------------------

def run_pred(model, value_dict, filter, shape, num_frames, num_pose_frames, action_dim, decoding_t, device):
    """
    Returns:
      action_pred: (num_pose_frames, action_dim) in model space
      vid_hand:    predicted hand-view strip (uint8)
      vid_agent:   predicted agent-view strip (uint8)
    """
    with torch.no_grad():
        with torch.autocast(device):
            # only the second view key is extra now
            extra_keys = ["cond_frames_3"]

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
                "noised_video_input": video_randn,
                "noised_pose_input": pose_randn
            }

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(2, num_frames).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]
            additional_model_inputs["num_pose_frames"] = batch["num_pose_frames"]

            def denoiser(input, sigma, c):
                return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

            samples_output = model.sampler(denoiser, noised_input, cond=c, uc=uc)

            action_pred = samples_output["noised_pose_input"]
            samples_z = samples_output["noised_video_input"]

            model.en_and_decode_n_samples_a_time = decoding_t
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
            samples = filter(samples)

            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255)
                .cpu().numpy().astype(np.uint8)
            )

            # Old code assumed: [0]=cond, [1:13]=viewA, [13:25]=viewB
            # Make it robust: split remaining frames evenly into 2 views.
            remaining = vid.shape[0] - 1
            seg = remaining // 2
            vid_hand = vid[1:1 + seg]
            vid_agent = vid[1 + seg:1 + 2 * seg]

    return action_pred, vid_hand, vid_agent


def convert_observations_2views(dataset, hand_img, agent_img,
                                task_description, cond_aug, motion_bucket_id, fps_id, device):
    """
    Produces only:
      cond_frames     = hand (wrist) view (1,3,H,W)
      cond_frames_3   = agent view (1,3,H,W)
    """
    hand = dataset.convert_frame(frame=hand_img, swap_rgb=dataset.swap_rgb)
    agent = dataset.convert_frame(frame=agent_img, swap_rgb=dataset.swap_rgb)

    cond_frames = np.expand_dims(hand, axis=0)
    cond_frames_3 = np.expand_dims(agent, axis=0)

    cond_frames = cond_frames + cond_aug * np.random.randn(*cond_frames.shape)
    cond_frames_3 = cond_frames_3 + cond_aug * np.random.randn(*cond_frames_3.shape)

    cond_frames = torch.from_numpy(cond_frames.astype(np.float32)).to(device)
    cond_frames_3 = torch.from_numpy(cond_frames_3.astype(np.float32)).to(device)

    value_dict = {
        "cond_frames": cond_frames,
        "cond_frames_3": cond_frames_3,
        "cond_frames_without_noise": task_description,
        "cond_aug": cond_aug,
        "motion_bucket_id": motion_bucket_id,
        "fps_id": fps_id,
    }
    return value_dict

# -------------------------
# 2-view ONLY changes end
# -------------------------


def run_experiment(model, dataset, config, filter, device: str = "cuda"):
    log_folder = f"experiments/{config['log_folder']}"
    os.makedirs(log_folder, exist_ok=True)
    OmegaConf.save(config, f"{log_folder}/inference_config.yaml")

    experiment_record = f"{log_folder}/multi_environment_experiment_record.json"
    experiment_record_lock = f"{experiment_record}.lock"
    lock = FileLock(experiment_record_lock)

    with lock:
        if os.path.exists(experiment_record):
            print("Experiment record already exists.")
        else:
            print("Experiment record does not exist.")
            create_environment_data_from_yaml(config, experiment_record)

        experiments_data = get_earliest_pending_experiments(
            experiment_record, config["number_of_experiments"]
        )
        if experiments_data is None:
            print("All experiments are in progress or completed.")
            return

        experiments_data = set_all_status_to_in_progress(experiments_data)
        update_json_file(experiment_record, experiments_data)

    task_name = list(experiments_data["environments"].keys())[0]
    demos = list(experiments_data["environments"][task_name]["experiments"].keys())

    max_traj_len      = config.max_traj_len
    camera_height     = config.data.params.frame_height
    camera_width      = config.data.params.frame_width
    action_horizon    = config.action_horizon
    cond_aug          = config.data.params.cond_aug
    motion_bucket_id  = config.data.params.motion_bucket_id
    fps_id            = config.data.params.fps_id

    num_frames        = config.model.params.sampler_config.params.guider_config.params.num_frames
    num_pose_frames   = config.model.params.sampler_config.params.guider_config.params.num_pose_frames
    action_dim        = config.data.params.action_dim
    decoding_t        = config.decoding_t
    shape             = (num_frames, 4, camera_width // 8, camera_height // 8)

    task_id = NAME2ID[task_name]

    for demo in demos:
        demo_number = int(demo.replace("demo_", ""))
        init_state_id = demo_number // 10

        env, task_description_str, _task_suite, obs = create_libero_env(
            task_suite_name="libero_10",
            task_id=task_id,
            camera_height=camera_height,
            camera_width=camera_width,
            seed=0,
            init_state_id=init_state_id,
        )

        task_description = open_clip.tokenize([task_description_str])  # (1,77)

        video_path = f"{log_folder}/libero10_task{task_id}_{demo}.mp4"
        video_writer = imageio.get_writer(video_path, fps=30)

        success = 0

        for i in range(int(max_traj_len / action_horizon)):
            if obs is None:
                zero_action = np.zeros(action_dim, dtype=np.float32)
                obs, _r, _d, _info = env.step(zero_action)

            # NOTE: use the keys that your OffScreenRenderEnv actually gives.
            # If your print shows 'agentview_rgb' and 'eye_in_hand_rgb', use those.
            # Your current code uses these:
            agent = obs["agentview_image"]
            hand  = obs["robot0_eye_in_hand_image"]

            # 2-view value_dict only
            value_dict = convert_observations_2views(
                dataset=dataset,
                hand_img=hand,
                agent_img=agent,
                task_description=task_description,
                cond_aug=cond_aug,
                motion_bucket_id=motion_bucket_id,
                fps_id=fps_id,
                device=device,
            )

            action_pred, vid_hand, vid_agent = run_pred(
                model=model,
                value_dict=value_dict,
                filter=filter,
                shape=shape,
                num_frames=num_frames,
                num_pose_frames=num_pose_frames,
                action_dim=action_dim,
                decoding_t=decoding_t,
                device=device,
            )

            action_pred_raw = action_pred.detach().cpu().numpy()
            action_pred_raw = np.clip(action_pred_raw, -1.0, 1.0)

            # unnormalize actions
            action_pred = ((action_pred_raw + 1) / 2) * (dataset.max - dataset.min) + dataset.min
            action_pred = action_pred[:action_horizon]
            action_pred = np.clip(action_pred, dataset.min, dataset.max)

            print(i)

            for step in range(action_pred.shape[0]):
                obs, reward, done, info = env.step(action_pred[step])

                agent = obs["agentview_image"]
                hand  = obs["robot0_eye_in_hand_image"]

                # top row: current obs views (agent | hand)
                top = np.concatenate([agent, hand], axis=1)

                # bottom row: predicted views (agent | hand) aligned with top
                if step < vid_agent.shape[0] and step < vid_hand.shape[0]:
                    bottom = np.concatenate([vid_agent[step], vid_hand[step]], axis=1)
                else:
                    bottom = np.concatenate([vid_agent[-1], vid_hand[-1]], axis=1)

                frame = np.concatenate([top, bottom], axis=0)
                video_writer.append_data(frame)

                step_success = int(bool(env.check_success()))
                if step_success == 1:
                    success = 1
                    break

                if done:
                    break

            if success == 1:
                break

        experiments_data["environments"][task_name]["experiments"][demo]["status"] = "done"
        experiments_data["environments"][task_name]["experiments"][demo]["success"] = int(success)

        print(colored(f"Saved video to {video_path}", "green"))
        video_writer.close()
        env.close()

        with lock:
            update_json_file(experiment_record, experiments_data)


def get_unique_embedder_keys_from_conditioner(conditioner, extra_keys=None):
    unique_keys = list(set([x.input_key for x in conditioner.embedders]))
    if extra_keys:
        unique_keys.extend(extra_keys)
    return unique_keys


def get_batch(keys, value_dict, N, T, T_p, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "cond_aug":
            batch[key] = repeat(torch.tensor([value_dict["cond_aug"]]).to(device), "1 -> b", b=math.prod(N))
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[1])
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
    model, filter, config = load_model(model_config, device)
    return model, filter, config


def load_model(config: str, device: str):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = device

    model = instantiate_from_config(config.model).to(device).eval()
    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LIBERO sampling experiment (2 views).")
    parser.add_argument(
        "-c", "--config",
        default="scripts/sampling/configs/svd_xt.yaml",
        help="Path to YAML configuration file."
    )
    args = parser.parse_args()

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

    run_experiment(
        model=model,
        dataset=dataset,
        config=config,
        filter=filter,
    )
