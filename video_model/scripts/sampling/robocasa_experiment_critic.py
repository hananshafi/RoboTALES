"""RoboCasa rollout that:
- Runs the Video UNet policy in closed loop (same as robocasa_planner.py).
- Scores each per-step generated video (24 frames: 8 per view x 3 views) with LIV
  against the task description, and against each subtask from the planner.
- Saves per-step LIV scores, per-rollout aggregations, env success, and a
  subgoal-progress index, into a flat JSON record for offline analysis.

Usage:
  cd /bigdata/hanan/dev/videopolicy/video_model
  python scripts/sampling/robocasa_experiment_critic.py \
      -c scripts/sampling/configs/svd_xt.yaml \
      --use_planner \
      --planner_api_key $GEMINI_API_KEY
"""
import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import sys

import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from filelock import FileLock
from omegaconf import OmegaConf
from termcolor import colored

import open_clip
import robocasa.utils.dataset_registry  # noqa: F401  (registers tasks)
import robosuite

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.data import VideoDataset, SINGLE_STAGE_TASK_DATASETS
from sgm.util import instantiate_from_config

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from liv_critic import LIVCritic

# Cache-aware planner (reads planner_cache.jsonl directly).
from sgm.data.llm_planner import OnlinePlanner

DEFAULT_PLANNER_CACHE = "/bigdata/hanan/dev/videopolicy/video_model/planner_cache.jsonl"


def get_subtasks(task_description, planner):
    """Cache-first lookup via OnlinePlanner.plan_steps(..., mode='test').

    Mirrors the integration in robocasa_planner_features.py. Logs whether the
    instruction was a cache HIT or MISS. Falls back to [task_description] if
    the planner errors (e.g. cache miss + no API key).
    """
    # Determine hit/miss BEFORE calling plan_steps (which may write back).
    key = hashlib.sha1(f"{planner.model}|{task_description}".encode()).hexdigest()
    in_train = isinstance(planner.records.get(key), dict) and \
               isinstance(planner.records[key].get("steps"), list)
    in_test = isinstance(planner.records_test.get(key), dict) and \
              isinstance(planner.records_test[key].get("steps"), list)
    in_disk = planner.cache.get(key) is not None
    cache_hit = in_train or in_test or in_disk
    src = "train" if in_train else ("test" if in_test else ("disk" if in_disk else "MISS"))

    try:
        result = planner.plan_steps(task_description, mode='test')
        steps = result["steps"] if isinstance(result, dict) else result
        if isinstance(steps, list) and len(steps) > 0:
            tag = colored("HIT", "green") if cache_hit else colored("MISS->API", "yellow")
            print(f"[planner_cache] {tag} ({src}) key={key[:10]} "
                  f"task={task_description!r} -> {len(steps)} steps")
            return list(steps)
    except Exception as e:
        print(colored(f"[planner_cache] MISS ({src}) and API failed: {e} "
                      f"key={key[:10]} task={task_description!r}; using single-task",
                      "red"))
    return [task_description]


np.set_printoptions(precision=5, suppress=True)


# ---------- record-keeping ----------

def build_environment_data_from_yaml(config):
    data = {"environments": {}}
    for env_name, env_details in config.data.params.tasks.items():
        n = env_details.get("num_experiments", 0)
        data["environments"][env_name] = {
            "experiments": {
                f"demo_{i}": {"status": "pending", "success": -1}
                for i in range(n)
            }
        }
    return data


def create_environment_data_from_yaml(config, output_json_file):
    data = build_environment_data_from_yaml(config)
    with open(output_json_file, "w") as f:
        json.dump(data, f, indent=4)


def get_earliest_pending_experiments(json_file, max_experiments):
    with open(json_file, "r") as f:
        data = json.load(f)
    for env_name, env_details in data["environments"].items():
        pending = {
            k: v for k, v in env_details["experiments"].items()
            if v["status"] == "pending"
        }
        if pending:
            limited = dict(list(pending.items())[:max_experiments])
            return {"environments": {env_name: {"experiments": limited}}}
    return None


def get_all_pending_experiments(json_file):
    """Return ALL pending experiments across ALL environments."""
    with open(json_file, "r") as f:
        data = json.load(f)
    out_envs = {}
    for env_name, env_details in data["environments"].items():
        pending = {
            k: v for k, v in env_details["experiments"].items()
            if v["status"] == "pending"
        }
        if pending:
            out_envs[env_name] = {"experiments": pending}
    if not out_envs:
        return None
    return {"environments": out_envs}


def set_all_status_to_in_progress(data):
    for env in data["environments"].values():
        for exp in env["experiments"].values():
            exp["status"] = "in_progress"
    return data


def update_json_file(json_file, updated_data):
    with open(json_file, "r") as f:
        existing = json.load(f)

    def merge(src, tgt):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(tgt.get(k), dict):
                merge(v, tgt[k])
            else:
                tgt[k] = v

    merge(updated_data, existing)
    with open(json_file, "w") as f:
        json.dump(existing, f, indent=4)


# ---------- env / model / sampling ----------

def create_eval_env_modified(env_name, controller_configs, id_selection,
                             camera_names=("robot0_agentview_left",
                                           "robot0_agentview_right",
                                           "robot0_eye_in_hand"),
                             camera_widths=256, camera_heights=256,
                             layout_and_style_ids=((1, 1), (2, 2), (4, 4),
                                                   (6, 9), (7, 10))):
    layout = (layout_and_style_ids[id_selection],)
    return robosuite.make(
        env_name=env_name,
        robots="PandaMobile",
        controller_configs=controller_configs,
        camera_names=list(camera_names),
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=None,
        obj_instance_split="B",
        generative_textures=None,
        randomize_cameras=False,
        layout_and_style_ids=layout,
        translucent_robot=False,
    )


def get_unique_embedder_keys_from_conditioner(conditioner, extra_keys=None):
    keys = list(set(x.input_key for x in conditioner.embedders))
    if extra_keys:
        keys.extend(extra_keys)
    return keys


def get_batch(keys, value_dict, N, T, T_p, device):
    batch, batch_uc = {}, {}
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
    for k in batch:
        if k not in batch_uc and isinstance(batch[k], torch.Tensor):
            batch_uc[k] = torch.clone(batch[k])
    return batch, batch_uc


def convert_observations(dataset, image_0, image_1, image_2, image_3,
                         task_description, cond_aug, motion_bucket_id, fps_id, device):
    images = [dataset.convert_frame(frame=im, swap_rgb=dataset.swap_rgb)
              for im in (image_0, image_1, image_2, image_3)]
    cond = []
    for im in images:
        a = np.expand_dims(im, axis=0)
        a = a + cond_aug * np.random.randn(*a.shape)
        cond.append(torch.from_numpy(a.astype(np.float32)).to(device))

    return {
        "cond_frames": cond[0],
        "cond_frames_2": cond[1],
        "cond_frames_3": cond[2],
        "cond_frames_4": cond[3],
        "cond_frames_without_noise": task_description,
        "cond_aug": cond_aug,
        "motion_bucket_id": motion_bucket_id,
        "fps_id": fps_id,
    }


def run_pred(model, value_dict, filter_, shape, num_frames, num_pose_frames,
             action_dim, decoding_t, device):
    """Returns (action_pred, vid_1, vid_2, vid_3, gen_frames_all_uint8).

    gen_frames_all_uint8 is (24, H, W, 3) RGB uint8 — concatenation of the three
    8-frame view chunks, suitable for LIV scoring.
    """
    with torch.no_grad():
        with torch.autocast(device):
            extra_keys = ["cond_frames_2", "cond_frames_3", "cond_frames_4"]
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner, extra_keys),
                value_dict, [1, num_frames],
                T=num_frames, T_p=num_pose_frames, device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch, batch_uc=batch_uc,
                force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
            )

            video_randn = torch.randn(shape, device=device)
            pose_randn = torch.randn((num_pose_frames, action_dim), device=device)
            noised = {"noised_video_input": video_randn, "noised_pose_input": pose_randn}

            extras = {
                "image_only_indicator": torch.zeros(2, num_frames).to(device),
                "num_video_frames": batch["num_video_frames"],
                "num_pose_frames": batch["num_pose_frames"],
            }

            def denoiser(inp, sigma, c_):
                return model.denoiser(model.model, inp, sigma, c_, **extras)

            out = model.sampler(denoiser, noised, cond=c, uc=uc)
            action_pred = out["noised_pose_input"]
            samples_z = out["noised_video_input"]
            model.en_and_decode_n_samples_a_time = decoding_t
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
            samples = filter_(samples)

            vid = (rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)

    vid_1, vid_2, vid_3 = vid[1:9], vid[9:17], vid[17:25]
    gen_all = np.concatenate([vid_1, vid_2, vid_3], axis=0)
    return action_pred, vid_1, vid_2, vid_3, gen_all


# ---------- rollout w/ critic ----------

def _aggregate(scores_per_step):
    """scores_per_step: list of (24,) arrays. Returns dict of scalars."""
    if not scores_per_step:
        return {"mean": float("nan"), "max": float("nan"),
                "last_step_mean": float("nan"), "last_frame": float("nan"),
                "last_quarter_mean": float("nan")}
    flat = np.concatenate(scores_per_step)
    last_step = scores_per_step[-1]
    n = len(scores_per_step)
    last_q = np.concatenate(scores_per_step[max(0, n - max(1, n // 4)):])
    return {
        "mean": float(flat.mean()),
        "max": float(flat.max()),
        "last_step_mean": float(last_step.mean()),
        "last_frame": float(last_step[-1]),
        "last_quarter_mean": float(last_q.mean()),
    }


def run_experiment_with_critic(model, dataset, config, filter_,
                               device="cuda", use_planner=True,
                               planner_api_key=None,
                               planner_cache_path=DEFAULT_PLANNER_CACHE):
    log_folder = f"experiments/{config['log_folder']}_critic"
    os.makedirs(log_folder, exist_ok=True)
    OmegaConf.save(config, f"{log_folder}/inference_config.yaml")

    record_path = f"{log_folder}/multi_environment_experiment_record.json"
    lock = FileLock(record_path + ".lock")

    with lock:
        if not os.path.exists(record_path):
            create_environment_data_from_yaml(config, record_path)
        experiments_data = get_all_pending_experiments(record_path)
        if experiments_data is None:
            print("All experiments are in progress or completed.")
            return
        experiments_data = set_all_status_to_in_progress(experiments_data)
        update_json_file(record_path, experiments_data)

    task_names = list(experiments_data["environments"].keys())
    print(f"[critic] running {sum(len(v['experiments']) for v in experiments_data['environments'].values())} "
          f"demos across {len(task_names)} tasks")

    # patch human_path absolute prefix (same fix as robocasa_planner.py)
    for env in list(SINGLE_STAGE_TASK_DATASETS.keys()):
        entry = SINGLE_STAGE_TASK_DATASETS.get(env, {})
        if isinstance(entry, dict) and "human_path" in entry and not entry["human_path"].startswith("/"):
            entry["human_path"] = "/home/hanan/dev/videopolicy/video_model/" + entry["human_path"]

    with open(SINGLE_STAGE_TASK_DATASETS["ExampleEnvironmentData"], "rb") as f:
        environment_data = pickle.load(f)

    max_traj_len = config.max_traj_len
    camera_names = environment_data["env_kwargs"]["camera_names"]
    camera_h = config.data.params.frame_height
    camera_w = config.data.params.frame_width
    action_horizon = config.action_horizon
    cond_aug = config.data.params.cond_aug
    motion_bucket_id = config.data.params.motion_bucket_id
    fps_id = config.data.params.fps_id
    num_frames = config.model.params.sampler_config.params.guider_config.params.num_frames
    num_pose_frames = config.model.params.sampler_config.params.guider_config.params.num_pose_frames
    action_dim = config.data.params.action_dim
    decoding_t = config.decoding_t
    shape = (num_frames, 4, camera_w // 8, camera_h // 8)

    # critic
    critic = LIVCritic(model, device=device)

    # cache-aware planner (reads planner_cache.jsonl). Lazy init so the script
    # can still run with --use_planner=False even if google-genai is missing.
    planner = None
    if use_planner:
        try:
            planner = OnlinePlanner(cache_path=planner_cache_path)
            print(f"[planner] OnlinePlanner ready (cache={planner_cache_path})")
        except Exception as e:
            print(f"[planner] OnlinePlanner init failed: {e}; falling back to single-task")

    # critic-results sidecar (one row per finished demo)
    critic_path = f"{log_folder}/critic_records.jsonl"

    for task_name in task_names:
     demos = list(experiments_data["environments"][task_name]["experiments"].keys())
     print(colored(f"\n=== task: {task_name} | {len(demos)} demos ===", "cyan"))
     for demo in demos:
        demo_number = int(demo.replace("demo_", ""))
        try:
            env = create_eval_env_modified(
                env_name=task_name,
                controller_configs=environment_data["env_kwargs"]["controller_configs"],
                id_selection=demo_number // 10,
            )
            env.reset()
        except Exception as e:
            print(colored(f"[{task_name}/{demo}] env creation failed: {e}; skipping", "red"))
            continue
        task_description = env.get_ep_meta()["lang"]

        # Get subtasks via cache-aware OnlinePlanner (mode='test').
        if use_planner and planner is not None:
            subtasks = get_subtasks(task_description, planner)
        else:
            subtasks = [task_description]

        task_tokens = open_clip.tokenize([task_description]).to(device)
        video_path = f"{log_folder}/{task_name}_{demo}.mp4"
        video_writer = imageio.get_writer(video_path, fps=30)

        per_step_task_scores = []           # list of (24,) arrays
        per_step_subtask_scores = []        # list of (24, K) arrays
        per_step_max_subtask_idx = []       # list of int (subgoal progress)
        success = 0

        for i in range(int(max_traj_len / action_horizon)):
            cams = []
            for cam_name in camera_names:
                im = env.sim.render(height=camera_h, width=camera_w, camera_name=cam_name)[::-1]
                cams.append(im)

            value_dict = convert_observations(
                dataset=dataset,
                image_0=cams[2], image_1=cams[2],  # eye_in_hand
                image_2=cams[0], image_3=cams[1],
                task_description=task_tokens,
                cond_aug=cond_aug, motion_bucket_id=motion_bucket_id,
                fps_id=fps_id, device=device,
            )

            action_pred, vid_1, vid_2, vid_3, gen_all = run_pred(
                model=model, value_dict=value_dict, filter_=filter_,
                shape=shape, num_frames=num_frames, num_pose_frames=num_pose_frames,
                action_dim=action_dim, decoding_t=decoding_t, device=device,
            )

            # --- LIV scoring on generated frames (the Video UNet output) ---
            task_scores = critic.score_frames_uint8(gen_all, task_description)  # (24,)
            per_step_task_scores.append(task_scores)

            sub_scores = critic.score_against_many(gen_all, subtasks)  # (24, K)
            per_step_subtask_scores.append(sub_scores)
            # subgoal progress: argmax subtask of the mean-over-frames score
            mean_per_subtask = sub_scores.mean(axis=0)
            per_step_max_subtask_idx.append(int(np.argmax(mean_per_subtask)))

            # --- execute predicted actions ---
            action_pred = ((action_pred.detach().cpu().numpy() + 1) / 2) * (dataset.max - dataset.min) + dataset.min
            action_pred = np.hstack((action_pred, [[0, 0, 0, 0, -1]] * action_pred.shape[0]))
            action_pred = action_pred[0:action_horizon]

            for step in range(action_pred.shape[0]):
                env.step(action_pred[step])
                cams = []
                for cam_name in camera_names:
                    im = env.sim.render(height=camera_h, width=camera_w, camera_name=cam_name)[::-1]
                    cams.append(im)
                row = np.concatenate(cams, axis=1)
                if step < vid_1.shape[0]:
                    new_row = np.concatenate((vid_2[step], vid_3[step], vid_1[step]), axis=1)
                else:
                    new_row = np.concatenate((vid_2[-1], vid_3[-1], vid_1[-1]), axis=1)
                row = np.concatenate((row, new_row), axis=0)
                video_writer.append_data(row)
                if env._check_success():
                    break

            if env._check_success():
                success = 1
                break

        # finalize
        experiments_data["environments"][task_name]["experiments"][demo]["status"] = "done"
        experiments_data["environments"][task_name]["experiments"][demo]["success"] = success

        agg_task = _aggregate(per_step_task_scores)
        subgoal_progress = (max(per_step_max_subtask_idx) + 1) if per_step_max_subtask_idx else 0
        record = {
            "task": task_name,
            "demo": demo,
            "success": success,
            "num_steps": len(per_step_task_scores),
            "subtasks": subtasks,
            "num_subtasks": len(subtasks),
            "subgoal_progress_index": subgoal_progress,            # 1..K
            "subgoal_progress_frac": subgoal_progress / max(1, len(subtasks)),
            "reward_task_mean": agg_task["mean"],
            "reward_task_max": agg_task["max"],
            "reward_task_last_step_mean": agg_task["last_step_mean"],
            "reward_task_last_frame": agg_task["last_frame"],
            "reward_task_last_quarter_mean": agg_task["last_quarter_mean"],
            "per_step_task_scores": [s.tolist() for s in per_step_task_scores],
            "per_step_max_subtask_idx": per_step_max_subtask_idx,
        }
        with open(critic_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        experiments_data["environments"][task_name]["experiments"][demo].update({
            "reward_task_mean": agg_task["mean"],
            "reward_task_max": agg_task["max"],
            "reward_task_last_step_mean": agg_task["last_step_mean"],
            "reward_task_last_frame": agg_task["last_frame"],
            "reward_task_last_quarter_mean": agg_task["last_quarter_mean"],
            "subgoal_progress_index": subgoal_progress,
            "subgoal_progress_frac": subgoal_progress / max(1, len(subtasks)),
            "num_subtasks": len(subtasks),
        })

        print(colored(f"[{task_name}/{demo}] success={success} "
                      f"mean_r={agg_task['mean']:.3f} subgoal={subgoal_progress}/{len(subtasks)}",
                      "green"))
        video_writer.close()

        with lock:
            update_json_file(record_path, experiments_data)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------- model loading ----------

def load_model(config_path, device):
    config = OmegaConf.load(config_path)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[0] \
            .params.open_clip_embedding_config.params.init_device = device
    model = instantiate_from_config(config.model).to(device).eval()
    filter_ = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter_, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoboCasa rollout with LIV critic + planner.")
    parser.add_argument("-c", "--config",
                        default="scripts/sampling/configs/svd_xt.yaml")
    parser.add_argument("--use_planner", action="store_true", default=True)
    parser.add_argument("--planner_api_key", type=str, default=None)
    parser.add_argument("--planner_cache",
                        default=DEFAULT_PLANNER_CACHE,
                        help="Path to planner_cache.jsonl produced by OnlinePlanner.")
    args = parser.parse_args()

    model, filter_, config = load_model(args.config, "cuda")

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

    run_experiment_with_critic(
        model=model, dataset=dataset, config=config, filter_=filter_,
        use_planner=args.use_planner, planner_api_key=args.planner_api_key,
        planner_cache_path=args.planner_cache,
    )
