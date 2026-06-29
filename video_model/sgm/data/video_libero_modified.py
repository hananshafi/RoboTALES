import numpy as np
import cv2

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

import lovely_numpy
import lovely_tensors
from rich import print
lovely_tensors.monkey_patch()
np.set_printoptions(precision=5, suppress=True)

import os
import torch
import open_clip
import random
import glob

import torchvision.transforms as T
import h5py
from collections import OrderedDict
from torch.utils.data.dataloader import default_collate
from einops import rearrange
import json
from itertools import chain
from .llm_planner import OnlinePlanner



def format_task_with_steps_clean(task_description: str, steps: list[str]) -> str:
    lines = [f"Task: {task_description.strip()}"]
    for i, step in enumerate(steps, start=1):
        lines.append(f" Step {i}: {step.strip()}")
    return "\n".join(lines)


def grab_language_from_filename(x: str) -> str:
    if x[0].isupper():  # LIBERO-100 style
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8:].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7:].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".hdf5")
    return language[:en]


LIBERO10_DIR = "/home/hanan/dev/LIBERO/datasets/libero_10"

def _task_name_from_file(fp: str) -> str:
    base = os.path.basename(fp)
    return base.replace("_demo.hdf5", "")

libero_files = sorted(glob.glob(os.path.join(LIBERO10_DIR, "*_demo.hdf5")))

SINGLE_STAGE_TASK_DATASETS = OrderedDict(
    KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it=dict(
        horizon=500,
        human_path="/home/hanan/dev/LIBERO/datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
    ),
    KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it=dict(
        horizon=500,
        human_path="/home/hanan/dev/LIBERO/datasets/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5",
    ),
    KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it=dict(
        horizon=500,
        human_path="/home/hanan/dev/LIBERO/datasets/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5",
    ),
    KITCHEN_SCENE8_put_both_moka_pots_on_the_stove=dict(
        horizon=500,
        human_path="/home/hanan/dev/LIBERO/datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5",
    ),
    LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket=dict(
        horizon=500,
        human_path="/home/hanan/dev/LIBERO/datasets/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5",
    ),
    LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket=dict(
        horizon=500,
        human_path="/home/hanan/dev/LIBERO/datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5",
    ),
    LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket=dict(
        horizon=500,
        human_path="/home/hanan/dev/LIBERO/datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5",
    ),
    LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate=dict(
        horizon=500,
        human_path="/home/hanan/dev/LIBERO/datasets/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5",
    ),
    LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate=dict(
        horizon=500,
        human_path="/home/hanan/dev/LIBERO/datasets/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5",
    ),
    STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy=dict(
        horizon=500,
        human_path="/home/hanan/dev/LIBERO/datasets/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
    ),
)

def get_new_ds_path(task, ds_type, return_info=False):
    if task not in SINGLE_STAGE_TASK_DATASETS:
        raise ValueError(f"Unknown task: {task}")
    ds_config = SINGLE_STAGE_TASK_DATASETS[task]
    if ds_type != "human_im":
        raise ValueError(f"Unsupported ds_type: {ds_type}")
    ds_path = ds_config["human_path"]
    if not return_info:
        return ds_path
    ds_info = {"horizon": ds_config.get("horizon", None)}
    return ds_path, ds_info


class VideoDataset(Dataset):
    """
    2-camera views only:
      - agentview_rgb (external view)
      - eye_in_hand_rgb (wrist camera)

    Packed video sequence layout:
      [cond_frame (wrist t=0)] + [wrist frames (H)] + [agentview frames (H)]
    Total frames per sample: 1 + 2*H
    """
    def __init__(
        self,
        n_frames: int,
        cond_aug: float,
        motion_bucket_id: int,
        fps_id: int,
        frame_width: int,
        frame_height: int,
        tasks: dict,
        skip_demos: dict,
        video_stride: int,
        video_pred_horizon: int,
        aug: dict,
        action_dim: int,
        swap_rgb: bool,
        mode: str,
    ):
        super().__init__()

        if mode not in ["train", "test"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'train' or 'test'.")

        self.mode = mode
        self.n_frames = n_frames
        self.cond_aug = cond_aug
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.video_stride = video_stride
        self.video_pred_horizon = video_pred_horizon  # H
        self.action_dim = action_dim
        self.swap_rgb = swap_rgb
        self.planner = OnlinePlanner()

        if self.mode == "train":
            self.transform_rgb = T.Compose([
                T.ColorJitter(
                    brightness=aug["brightness"],
                    contrast=aug["contrast"],
                    saturation=aug["saturation_rgb"],
                    hue=aug["hue_rgb"],
                ),
            ])
        else:
            self.transform_rgb = T.Compose([])

        if tasks is not None and len(tasks) > 0:
            self.task_list = list(tasks.keys())
        else:
            self.task_list = list(SINGLE_STAGE_TASK_DATASETS.keys())

        self.datasets, self.hdf5_datasets = self.get_dataset_file(self.task_list)

        # Build flat index: (task_name, task_index, demo_key, demo_step, tokenized_task)
        self.indexed_demos = []
        for task_index, task_name in enumerate(self.task_list):
            print("Loading task:", task_name)
            
            task_data = self.datasets[task_index]["data"]
            if self.mode == "test":
                for demo_key in task_data.keys():
                    if (task_name in skip_demos and demo_key in skip_demos[task_name]):
                        continue

                    # task_description = json.loads(self.hdf5_datasets[task_index]['data'][demo_key].attrs['ep_meta'])['lang'] 
                    task_description = grab_language_from_filename(task_name)
                    try:
                        steps = self.planner.plan_steps(task_description, mode=self.mode)["steps"]
                        steps  = format_task_with_steps_clean(task_description, steps)   # ". ".join(steps)
                    except Exception as e:
                        print("Couldn't generate steps, therefore steps dict will be original task instruction, error:", e)
                        steps = task_description
                        task_description_with_steps_tokenized = open_clip.tokenize(task_description)  #open_clip.tokenize(task_description)  # # returns torch.Size([1, 77])

                    demo_steps = range(0, task_data[demo_key]["actions"].shape[0])
                    for demo_step in demo_steps:
                        self.indexed_demos.append((task_name, task_index, demo_key, demo_step, task_description_with_steps_tokenized, [steps], task_description))

            else:  # train
                for demo_key in task_data.keys():
                    if (task_name in skip_demos and demo_key in skip_demos[task_name]):     # skip invalid demos with robot base actions
                        continue

                    # task_description = json.loads(self.hdf5_datasets[task_index]['data'][demo_key].attrs['ep_meta'])['lang']
                    task_description = grab_language_from_filename(task_name)
                    try:
                        steps = self.planner.plan_steps(task_description, mode=self.mode)["steps"]
                        steps  = format_task_with_steps_clean(task_description, steps)  #". ".join(steps)
                    except Exception as e:
                        print("Couldn't generate steps, therefore steps dict will be original task instruction, error:", e)
                        steps = task_description
                    task_description_with_steps_tokenized =  open_clip.tokenize(steps)  #open_clip.tokenize(steps)
                    
                    demo_steps = range(0, task_data[demo_key]['actions'].shape[0])
                    
                    for demo_step in demo_steps:
                        self.indexed_demos.append((task_name, task_index, demo_key, demo_step, task_description_with_steps_tokenized, [steps], task_description))


            # action normalization stats
        all_relative_actions = []
        for task_index, task_name in enumerate(self.task_list):
            task_data = self.datasets[task_index]["data"]
            for demo_key in task_data.keys():
                demo_steps = range(0, int(task_data[demo_key]["actions"].shape[0]))
                for demo_step in demo_steps:
                    all_relative_actions.append(task_data[demo_key]["actions"][demo_step][0:self.action_dim])

        all_relative_actions = np.array(all_relative_actions)
        self.min = np.min(all_relative_actions, axis=0, keepdims=True)
        self.max = np.max(all_relative_actions, axis=0, keepdims=True)

        print(self.min)
        print(self.max)

    def load_hdf5_into_memory(self, h5_file):
        def recursive_load(h5_obj):
            if isinstance(h5_obj, h5py.Group):
                return {key: recursive_load(h5_obj[key]) for key in h5_obj.keys() if key != "obs"}
            elif isinstance(h5_obj, h5py.Dataset):
                return h5_obj[()]
            raise ValueError(f"Unsupported HDF5 object: {type(h5_obj)}")

        with h5py.File(h5_file, "r") as f:
            return recursive_load(f)

    def get_dataset_file(self, task_list):
        datasets = []
        hdf5_datasets = []

        for task in list(SINGLE_STAGE_TASK_DATASETS):
            if task not in task_list:
                continue

            human_path, _ = get_new_ds_path(task=task, ds_type="human_im", return_info=True)
            in_memory_data = self.load_hdf5_into_memory(human_path)
            datasets.append(in_memory_data)

            hdf5_file = h5py.File(human_path, "r")
            hdf5_datasets.append(hdf5_file)

        return datasets, hdf5_datasets

    def convert_frame(self, frame, size=None, swap_rgb=False):
        if size is not None:
            original_height, original_width = frame.shape[:2]
            target_width, target_height = size

            if original_width != target_width or original_height != target_height:
                original_aspect_ratio = original_width / original_height
                target_aspect_ratio = target_width / target_height

                if original_aspect_ratio > target_aspect_ratio:
                    new_width = int(original_height * target_aspect_ratio)
                    crop_start = (original_width - new_width) // 2
                    cropped_image = frame[:, crop_start:crop_start + new_width]
                else:
                    new_height = int(original_width / target_aspect_ratio)
                    crop_start = (original_height - new_height) // 2
                    cropped_image = frame[crop_start:crop_start + new_height, :]

                frame = cv2.resize(cropped_image, size, interpolation=cv2.INTER_LINEAR)

        if swap_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.astype(np.float32) / 255.0
        frame = frame * 2.0 - 1.0
        frame = np.transpose(frame, (2, 0, 1))
        return frame

    def augmentation_transform(self, images, transform):
        transformed_images = []
        seed = random.randint(0, 2**32)
        for frame in images:
            torch.manual_seed(seed)
            transformed_images.append(transform(frame))
        return torch.stack(transformed_images), seed

    def __getitem__(self, i):
        try:
            task_name, task_index, demo_key, demo_step, task_description, task_string, orig_task_string = self.indexed_demos[i]

            # actions: (H*stride, action_dim) then pad, then normalize
            rel = self.hdf5_datasets[task_index]["data"][demo_key]["actions"][
                demo_step : demo_step + self.video_pred_horizon * self.video_stride
            ][:, 0:self.action_dim]

            pad_a = self.video_pred_horizon * self.video_stride - rel.shape[0]
            if pad_a > 0:
                rel = np.concatenate([rel, np.zeros((pad_a, self.action_dim))], axis=0)

            rel_norm = 2 * ((rel - self.min) / (self.max - self.min + 1e-8)) - 1  # avoid div0

            # 2-view frames (H each, strided)
            agentview = self.hdf5_datasets[task_index]["data"][demo_key]["obs"]["agentview_rgb"][
                demo_step : demo_step + self.video_pred_horizon * self.video_stride : self.video_stride
            ]
            eye_in_hand = self.hdf5_datasets[task_index]["data"][demo_key]["obs"]["eye_in_hand_rgb"][
                demo_step : demo_step + self.video_pred_horizon * self.video_stride : self.video_stride
            ]

            # pad frames to H
            H = self.video_pred_horizon
            def pad_to_H(arr):
                if arr.shape[0] >= H:
                    return arr
                last = np.expand_dims(arr[-1], axis=0)
                pad = np.concatenate([last] * (H - arr.shape[0]), axis=0)
                return np.concatenate([arr, pad], axis=0)

            agentview = pad_to_H(agentview)
            eye_in_hand = pad_to_H(eye_in_hand)

            # resize/normalize to [-1,1], CHW
            agentview = np.stack([
                self.convert_frame(frame=f, size=(self.frame_width, self.frame_height), swap_rgb=self.swap_rgb)
                for f in agentview
            ])
            eye_in_hand = np.stack([
                self.convert_frame(frame=f, size=(self.frame_width, self.frame_height), swap_rgb=self.swap_rgb)
                for f in eye_in_hand
            ])

        except Exception as e:
            raise RuntimeError(f"sample retrieve exception: {e}")

        # torch for augmentation
        agentview = torch.tensor(agentview, dtype=torch.float32)
        eye_in_hand = torch.tensor(eye_in_hand, dtype=torch.float32)

        # [-1,1] -> [0,1]
        agentview = (agentview + 1) / 2
        eye_in_hand = (eye_in_hand + 1) / 2

        agentview, _ = self.augmentation_transform(agentview, self.transform_rgb)
        eye_in_hand, _ = self.augmentation_transform(eye_in_hand, self.transform_rgb)

        # [0,1] -> [-1,1]
        agentview = agentview * 2 - 1
        eye_in_hand = eye_in_hand * 2 - 1

        agentview = agentview.numpy()
        eye_in_hand = eye_in_hand.numpy()

        # packed video: [cond wrist0] + [wrist H] + [agent H]
        video_data = np.concatenate((eye_in_hand[0:1], eye_in_hand, agentview), axis=0)

        # conditioning frames: one per view (no duplicates)
        cond_wrist = eye_in_hand[0:1]
        cond_frames_2 = eye_in_hand[0:1]  # kept name to minimize other code edits
        cond_agent = agentview[0:1]

        # noise cond frames
        cond_wrist = cond_wrist + self.cond_aug * np.random.randn(*cond_wrist.shape)
        cond_frames_2 = cond_frames_2 + self.cond_aug * np.random.randn(*cond_frames_2.shape)
        cond_agent = cond_agent + self.cond_aug * np.random.randn(*cond_agent.shape)

        # text conditioning (tokenized) repeated across time
        cond_text = task_description.numpy()
        cond_text = cond_text.repeat(self.n_frames, axis=0)


        cond_aug = np.ones(shape=(self.n_frames,), dtype=np.float32) * self.cond_aug
        motion_bucket_id = np.ones(shape=(self.n_frames,), dtype=np.int32) * self.motion_bucket_id
        fps_id = np.ones(shape=(self.n_frames,), dtype=np.int32) * self.fps_id
        image_only_indicator = np.zeros(shape=(1, self.n_frames), dtype=np.float32)

        # IMPORTANT: keep task_string fields consistent with your trainer usage
        # task_string = grab_language_from_filename(task_name)
        # orig_task_string = task_string

        return {
            "jpg": video_data.astype(np.float32),                    # (1+2H, 3, H, W)
            "cond_frames": cond_wrist.astype(np.float32),            # (1, 3, H, W)
            "cond_frames_2": cond_frames_2.astype(np.float32),      # (1, 3, H, W)  (kept name to minimize other code edits)
            "cond_frames_3": cond_agent.astype(np.float32),          # (1, 3, H, W)  (kept name to minimize other code edits)
            "cond_frames_4": cond_agent.astype(np.float32),          # (1, 3, H, W)  (kept name to minimize other code edits)
            "cond_frames_without_noise": cond_text,                  # (n_frames, 77) if task_tok is (1,77)
            "task_string": task_string,
            "orig_task_string": orig_task_string,
            "cond_aug": cond_aug,
            "motion_bucket_id": motion_bucket_id,
            "fps_id": fps_id,
            "image_only_indicator": image_only_indicator,
            "pose": rel_norm.astype(np.float32),                     # (H*stride, action_dim) padded
            "extra_frames": 1,                                       # because we prepended eye_in_hand[0:1]
        }

    def __len__(self):
        return len(self.indexed_demos)


def _to_str(x):
    if isinstance(x, (list, tuple)):
        return " ".join(map(str, x))
    return str(x)


def collate_fn(example_list):
    # normalize task_string before default_collate so it becomes list[str]
    for ex in example_list:
        if "task_string" in ex:
            ex["task_string"] = _to_str(ex["task_string"])

    collated = default_collate(example_list)

    # infer (B, T) from jpg (B,T,...) expected
    if "jpg" in collated and isinstance(collated["jpg"], torch.Tensor):
        B, T = collated["jpg"].shape[:2]
    else:
        any_key = next(k for k, v in collated.items() if isinstance(v, torch.Tensor) and v.ndim >= 2)
        B, T = collated[any_key].shape[:2]

    batch = {}

    for k, v in collated.items():
        if isinstance(v, torch.Tensor):
            if v.ndim >= 2:
                batch[k] = rearrange(v, "b t ... -> (b t) ...")
            elif v.ndim == 1 and v.shape[0] == B:
                batch[k] = v.repeat_interleave(T, dim=0)
            else:
                batch[k] = v
            continue

        if isinstance(v, np.ndarray):
            vt = torch.from_numpy(v)
            if vt.ndim >= 2:
                batch[k] = rearrange(vt, "b t ... -> (b t) ...")
            elif vt.ndim == 1 and vt.shape[0] == B:
                batch[k] = vt.repeat(T)
            else:
                batch[k] = vt
            continue

        if k == "task_string":
            per_clip = list(map(_to_str, v))
            batch[k] = list(chain.from_iterable([[s] * T for s in per_clip]))
        else:
            batch[k] = v

    batch["num_video_frames"] = int(T)

    if "pose" in collated and isinstance(collated["pose"], torch.Tensor):
        batch["num_pose_frames"] = int(collated["pose"].shape[1])
    else:
        batch["num_pose_frames"] = int(T)

    return batch


class VideoDatasetModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=1, shuffle=True, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dataset = VideoDataset(**kwargs)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    print(SINGLE_STAGE_TASK_DATASETS.keys())
