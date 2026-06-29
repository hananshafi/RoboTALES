import numpy as np
import cv2

from torch.utils.data import DataLoader, Dataset, default_collate
import pytorch_lightning as pl
from einops import rearrange

import lovely_numpy
import lovely_tensors
from lovely_numpy import lo
from rich import print
lovely_tensors.monkey_patch()
np.set_printoptions(precision=5, suppress=True)

import os
import torch
import open_clip
import random

import torchvision.transforms as T
import h5py
import json
from collections import OrderedDict
from .llm_planner import OnlinePlanner


def format_task_with_steps_clean(task_description: str, steps: list[str]) -> str:
    lines = [f"Task: {task_description.strip()}"]
    for i, step in enumerate(steps, start=1):
        lines.append(f" Step {i}: {step.strip()}")
    return "\n".join(lines)

SINGLE_STAGE_TASK_DATASETS = OrderedDict(
    PnPCounterToCab=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab",
    ),
    PnPCabToCounter=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter",
    ),
    PnPCounterToSink=dict(
        horizon=700,
        human_path="datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink",
    ),
    PnPSinkToCounter=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter",
    ),
    PnPCounterToMicrowave=dict(
        horizon=600,
        human_path="datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave",
    ),
    PnPMicrowaveToCounter=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter",
    ),
    PnPCounterToStove=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove",
    ),
    PnPStoveToCounter=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter",
    ),
    OpenSingleDoor=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor",
    ),
    CloseSingleDoor=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor",
    ),
    OpenDoubleDoor=dict(
        horizon=1000,
        human_path="datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor",
    ),
    CloseDoubleDoor=dict(
        horizon=700,
        human_path="datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor",
    ),
    OpenDrawer=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_drawer/OpenDrawer",
    ),
    CloseDrawer=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer",
    ),
    TurnOnSinkFaucet=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet",
    ),
    TurnOffSinkFaucet=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet",
    ),
    TurnSinkSpout=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_sink/TurnSinkSpout",
    ),
    TurnOnStove=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_stove/TurnOnStove",
    ),
    TurnOffStove=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_stove/TurnOffStove",
    ),
    CoffeeSetupMug=dict(
        horizon=600,
        human_path="datasets/v0.1/single_stage/kitchen_coffee/CoffeeSetupMug",
    ),
    CoffeeServeMug=dict(
        horizon=600,
        human_path="datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug",
    ),
    CoffeePressButton=dict(
        horizon=300,
        human_path="datasets/v0.1/single_stage/kitchen_coffee/CoffeePressButton",
    ),
    TurnOnMicrowave=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave",
    ),
    TurnOffMicrowave=dict(
        horizon=500,
        human_path="datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave",
    ),
    ExampleEnvironmentData="datasets/v0.1/single_stage/demo_gentex_im128_randcams_env.pkl"
)
# for env in SINGLE_STAGE_TASK_DATASETS:
#     if env == "ExampleEnvironmentData":
#         SINGLE_STAGE_TASK_DATASETS[env] = "/home/hanan/videopolicy/video_model/" + SINGLE_STAGE_TASK_DATASETS[env]
#     else:
#         SINGLE_STAGE_TASK_DATASETS[env]['human_path'] = "/home/hanan/videopolicy/video_model/" + SINGLE_STAGE_TASK_DATASETS[env]['human_path']


def get_new_ds_path(task, ds_type, return_info=False):
    if task in SINGLE_STAGE_TASK_DATASETS:
        ds_config = SINGLE_STAGE_TASK_DATASETS[task]
    else:
        raise ValueError

    if ds_type == "human_im":
        folder = ds_config["human_path"]
        if task in SINGLE_STAGE_TASK_DATASETS:
            fname = "demo_gentex_im256_randcams.hdf5"
    else:
        raise ValueError

    # if dataset type is not registered, return None
    if folder is None:
        ret = (None, None) if return_info is True else None
        return ret

    ds_path = os.path.join(folder, fname)

    if return_info is False:
        return ds_path

    ds_info = {}
    # ds_info["url"] = ds_config["download_links"][ds_type]
    ds_info["horizon"] = ds_config["horizon"]
    return ds_path, ds_info

class VideoDataset(Dataset):
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

        if mode not in ['train', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'train' or 'test'.")
        
        self.mode = mode
        self.n_frames = n_frames
        self.cond_aug = cond_aug
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.video_stride = video_stride
        self.video_pred_horizon = video_pred_horizon
        self.action_dim = action_dim
        self.swap_rgb = swap_rgb
        self.planner = OnlinePlanner()

        if self.mode == 'train':
            self.transform_rgb = T.Compose([
                T.ColorJitter(brightness=aug['brightness'], contrast=aug['contrast'], saturation=aug['saturation_rgb'], hue=aug['hue_rgb']),  # Random color jitter
            ])
        elif self.mode == 'test':
            self.transform_rgb = T.Compose([
            ])

        self.task_list = list(tasks.keys())
        self.datasets, self.hdf5_datasets = self.get_dataset_file(self.task_list)

        self.indexed_demos = []
        for task_index, task_name in enumerate(self.task_list):
            print("Loading task:", task_name)

            task_data = self.datasets[task_index]['data']
            if self.mode == "test":
                for demo_key in task_data.keys():
                    if (task_name in skip_demos and demo_key in skip_demos[task_name]):     # skip invalid demos with robot base actions
                        continue

                    task_description = json.loads(self.hdf5_datasets[task_index]['data'][demo_key].attrs['ep_meta'])['lang'] 
                    try:
                        steps = self.planner.plan_steps(task_description, mode=self.mode)["steps"]
                        steps  = format_task_with_steps_clean(task_description, steps)   # ". ".join(steps)
                    except Exception as e:
                        print("Couldn't generate steps, therefore steps dict will be original task instruction, error:", e)
                        steps = task_description
                    task_description_with_steps_tokenized = open_clip.tokenize(steps)  #open_clip.tokenize(task_description)  # # returns torch.Size([1, 77])

                    demo_steps = range(0, task_data[demo_key]['actions'].shape[0])
                    for demo_step in demo_steps:
                        self.indexed_demos.append((task_name, task_index, demo_key, demo_step, task_description_with_steps_tokenized, [steps], task_description))
            else:
                for demo_key in task_data.keys():
                    if (task_name in skip_demos and demo_key in skip_demos[task_name]):     # skip invalid demos with robot base actions
                        continue

                    task_description = json.loads(self.hdf5_datasets[task_index]['data'][demo_key].attrs['ep_meta'])['lang']
                    try:
                        steps = self.planner.plan_steps(task_description, mode=self.mode)["steps"]
                        steps  = format_task_with_steps_clean(task_description, steps)  #". ".join(steps)
                    except Exception as e:
                        print("Couldn't generate steps, therefore steps dict will be original task instruction, error:", e)
                        steps = task_description
                    #task_description_with_steps = open_clip.tokenize([steps]) # returns torch.Size([1, 77])
                    # if steps != None:
                    task_description_with_steps_tokenized =  open_clip.tokenize(steps)  #open_clip.tokenize(steps)
                    # else:
                    #     steps_tokenized = steps
                    demo_steps = range(0, task_data[demo_key]['actions'].shape[0])
                    for demo_step in demo_steps:
                        self.indexed_demos.append((task_name, task_index, demo_key, demo_step, task_description_with_steps_tokenized, [steps], task_description))

        
        all_relative_actions = []

        for task_index, task_name in enumerate(self.task_list):
            task_data = self.datasets[task_index]['data']
            for demo_key in task_data.keys():
                demo_steps = range(0, int(task_data[demo_key]['actions'].shape[0]))
                for demo_step in demo_steps:
                    all_relative_actions.append(task_data[demo_key]['actions'][demo_step][0:self.action_dim])

        all_relative_actions = np.array(all_relative_actions)

        self.min = np.min(all_relative_actions, axis=(0,))[None, :]
        self.max = np.max(all_relative_actions, axis=(0,))[None, :]

        # self.min = np.ones((1, 7), dtype=np.float32) * -1
        # self.max = np.ones((1, 7), dtype=np.float32)

        print(self.min)
        print(self.max)

    def load_hdf5_into_memory(self, h5_file):
        """Load an HDF5 file into memory as a dictionary."""
        def recursive_load(h5_obj):
            if isinstance(h5_obj, h5py.Group):
                return {
                        key: recursive_load(h5_obj[key])
                        for key in h5_obj.keys()
                        if key != 'obs'  # Exclude the 'obs' key
                    }
            elif isinstance(h5_obj, h5py.Dataset):
                return h5_obj[()]  # Load dataset into memory as a NumPy array
            else:
                raise ValueError(f"Unsupported HDF5 object: {type(h5_obj)}")

        with h5py.File(h5_file, "r") as f:
            return recursive_load(f)
    
    def get_dataset_file(self, task_list):
        datasets = []
        hdf5_datasets = []

        for task in list(SINGLE_STAGE_TASK_DATASETS):
            if task not in task_list:
                continue

            human_path, ds_meta = get_new_ds_path(task=task, ds_type="human_im", return_info=True)  # human dataset path

            in_memory_data = self.load_hdf5_into_memory(human_path)

            datasets.append(in_memory_data)

            hdf5_file = h5py.File(human_path, 'r')
            hdf5_datasets.append(hdf5_file)

        return datasets, hdf5_datasets

    def convert_frame(self, frame, size=None, swap_rgb=False):
        if size is not None:
            original_height, original_width = frame.shape[:2]
            target_width, target_height = size

            if original_width != target_width or original_height != target_height:
                # Calculate aspect ratios
                original_aspect_ratio = original_width / original_height
                target_aspect_ratio = target_width / target_height

                if original_aspect_ratio > target_aspect_ratio:
                    # Crop width (as in the original code)
                    new_width = int(original_height * target_aspect_ratio)
                    crop_start = (original_width - new_width) // 2
                    cropped_image = frame[:, crop_start:crop_start + new_width]
                else:
                    # Crop height
                    new_height = int(original_width / target_aspect_ratio)
                    crop_start = (original_height - new_height) // 2
                    cropped_image = frame[crop_start:crop_start + new_height, :]
                
                # Resize the cropped image to the target size
                frame = cv2.resize(cropped_image, size, interpolation=cv2.INTER_LINEAR)

        if swap_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.astype(np.float32)
        frame = frame / 255.0
        frame = frame * 2.0 - 1.0
        frame = np.transpose(frame, (2, 0, 1))  # Transpose the frame to have the shape (3, frame_height, frame_width)

        return frame

    def augmentation_transform(self, images, transform):
        transformed_images = []
        seed = random.randint(0, 2**32)
        for frame in images:
            torch.manual_seed(seed)
            transformed_images.append(transform(frame))
        images = torch.stack(transformed_images), seed

        return images

    def __getitem__(self, i):

        try:
            task_name, task_index, demo_key, demo_step, task_description, task_string, orig_task_string = self.indexed_demos[i]
            relative_actions = self.hdf5_datasets[task_index]['data'][demo_key]['actions'][demo_step:demo_step+self.video_pred_horizon*self.video_stride][:, 0:self.action_dim]
            
            pad_size = self.video_pred_horizon*self.video_stride - relative_actions.shape[0]

            if pad_size > 0:
                relative_actions = np.concatenate([relative_actions, np.zeros((pad_size, self.action_dim))], axis=0)

            relative_actions_normalized = 2 * ((relative_actions - self.min) / (self.max - self.min)) - 1

            left_image = self.hdf5_datasets[task_index]['data'][demo_key]['obs']['robot0_agentview_left_image'][demo_step:demo_step+self.video_pred_horizon*self.video_stride:self.video_stride]
            right_image = self.hdf5_datasets[task_index]['data'][demo_key]['obs']['robot0_agentview_right_image'][demo_step:demo_step+self.video_pred_horizon*self.video_stride:self.video_stride]
            gripper_image = self.hdf5_datasets[task_index]['data'][demo_key]['obs']['robot0_eye_in_hand_image'][demo_step:demo_step+self.video_pred_horizon*self.video_stride:self.video_stride]

            pad_size = self.video_pred_horizon - left_image.shape[0]
            if pad_size > 0:
                last_element = np.expand_dims(left_image[-1], axis=0)
                padding = np.concatenate([last_element] * pad_size, axis=0)
                left_image = np.concatenate([left_image, padding], axis=0)

                last_element = np.expand_dims(right_image[-1], axis=0)
                padding = np.concatenate([last_element] * pad_size, axis=0)
                right_image = np.concatenate([right_image, padding], axis=0)

                last_element = np.expand_dims(gripper_image[-1], axis=0)
                padding = np.concatenate([last_element] * pad_size, axis=0)
                gripper_image = np.concatenate([gripper_image, padding], axis=0)

            left_image = np.stack([self.convert_frame(frame=frame, size=(self.frame_width,self.frame_height), swap_rgb=self.swap_rgb) for frame in left_image])
            right_image = np.stack([self.convert_frame(frame=frame, size=(self.frame_width,self.frame_height), swap_rgb=self.swap_rgb) for frame in right_image])
            gripper_image = np.stack([self.convert_frame(frame=frame, size=(self.frame_width,self.frame_height), swap_rgb=self.swap_rgb) for frame in gripper_image])

        except Exception as e:
            print(f'sample retrive exception: {e}')
            
        left_image = torch.tensor(left_image, dtype=torch.float32)
        right_image = torch.tensor(right_image, dtype=torch.float32)
        gripper_image = torch.tensor(gripper_image, dtype=torch.float32)
        relative_actions_normalized = torch.tensor(relative_actions_normalized, dtype=torch.float32)

        # Rescale from [-1, 1] to [0, 1] for transforms
        left_image = (left_image + 1) / 2
        right_image = (right_image + 1) / 2
        gripper_image = (gripper_image + 1) / 2

        left_image, _ = self.augmentation_transform(left_image, self.transform_rgb)
        right_image, _ = self.augmentation_transform(right_image, self.transform_rgb)
        gripper_image, seed = self.augmentation_transform(gripper_image, self.transform_rgb)

        # Rescale back to [-1, 1]
        left_image = left_image * 2 - 1
        right_image = right_image * 2 - 1
        gripper_image = gripper_image * 2 - 1

        left_image = left_image.numpy()
        right_image = right_image.numpy()
        gripper_image = gripper_image.numpy()
        relative_actions_normalized = relative_actions_normalized.numpy()
        
        video_data = np.concatenate((gripper_image[0:1], gripper_image, left_image, right_image), axis=0)
        cond_frames = gripper_image[0:1]
        cond_frames_2 = gripper_image[0:1]
        cond_frames_3 = left_image[0:1]
        cond_frames_4 = right_image[0:1]

        cond_frames = (cond_frames + self.cond_aug * np.random.randn(*cond_frames.shape))
        cond_frames_2 = (cond_frames_2 + self.cond_aug * np.random.randn(*cond_frames_2.shape))
        cond_frames_3 = (cond_frames_3 + self.cond_aug * np.random.randn(*cond_frames_3.shape))
        cond_frames_4 = (cond_frames_4 + self.cond_aug * np.random.randn(*cond_frames_4.shape))

        cond_frames_without_noise = task_description.numpy()
        cond_frames_without_noise = cond_frames_without_noise.repeat(self.n_frames, axis=0)

        # steps_without_noise = steps.numpy()
        # steps_without_noise = steps_without_noise.repeat(self.n_frames, axis=0)

        cond_aug = np.ones(shape=(self.n_frames,)) * self.cond_aug
        motion_bucket_id = np.ones(shape=(self.n_frames,), dtype=np.int32) * self.motion_bucket_id
        fps_id = np.ones(shape=(self.n_frames,), dtype=np.int32) * self.fps_id
        image_only_indicator = np.zeros(shape=(1, self.n_frames,))
        return {
            "jpg": video_data.astype(np.float32),
            "cond_frames": cond_frames.astype(np.float32),
            "cond_frames_2": cond_frames_2.astype(np.float32),
            "cond_frames_3": cond_frames_3.astype(np.float32),
            "cond_frames_4": cond_frames_4.astype(np.float32),
            "cond_frames_without_noise": cond_frames_without_noise,
            "task_string": task_string,
            "orig_task_string": orig_task_string,
            "cond_aug": cond_aug.astype(np.float32),
            "motion_bucket_id": motion_bucket_id,
            "fps_id": fps_id,
            "image_only_indicator": image_only_indicator.astype(np.float32),
            "pose": relative_actions_normalized.astype(np.float32)  # [t, 7]
        }
    
    def __len__(self):
        return len(self.indexed_demos)

# def collate_fn(example_list):
#     collated = default_collate(example_list)
#     batch = {k: rearrange(v, "b t ... -> (b t) ...") for (k, v) in collated.items()}
#     batch["num_video_frames"] = 25
#     batch["num_pose_frames"] = 32
#     return batch



from torch.utils.data.dataloader import default_collate
from einops import rearrange
import torch
import numpy as np
from itertools import chain

def _to_str(x):
    # task_string can be str or list/tuple[str]; normalize to a single string
    if isinstance(x, (list, tuple)):
        return " ".join(map(str, x))
    return str(x)

def collate_fn(example_list):
    # normalize task_string before default_collate so it becomes a simple list[str]
    for ex in example_list:
        if "task_string" in ex:
            ex["task_string"] = _to_str(ex["task_string"])

    collated = default_collate(example_list)

    # infer (B, T) from a time-shaped tensor, prefer "jpg"
    if "jpg" in collated and isinstance(collated["jpg"], torch.Tensor):
        B, T = collated["jpg"].shape[:2]
    else:
        # fallback: try any tensor with ndim>=2
        any_key = next(k for k, v in collated.items()
                       if isinstance(v, torch.Tensor) and v.ndim >= 2)
        B, T = collated[any_key].shape[:2]

    batch = {}

    for k, v in collated.items():
        # tensors
        if isinstance(v, torch.Tensor):
            if v.ndim >= 2:
                # has (B, T, ...) -> flatten to (B*T, ...)
                batch[k] = rearrange(v, "b t ... -> (b t) ...")
            elif v.ndim == 1 and v.shape[0] == B:
                # per-sample metadata -> repeat across time to length B*T
                batch[k] = v.repeat_interleave(T, dim=0)
            else:
                # leave scalars or unexpected shapes alone
                batch[k] = v
            continue

        # numpy arrays
        if isinstance(v, np.ndarray):
            vt = torch.from_numpy(v)
            if vt.ndim >= 2:
                batch[k] = rearrange(vt, "b t ... -> (b t) ...")
            elif vt.ndim == 1 and vt.shape[0] == B:
                batch[k] = vt.repeat(T)
            else:
                batch[k] = vt
            continue

        # strings / lists / tuples
        if k == "task_string":
            # v is a list[str] of length B -> expand to length B*T
            # (["do X", "do Y"] -> ["do X"]*T + ["do Y"]*T)
            per_clip = list(map(_to_str, v))
            batch[k] = list(chain.from_iterable([[s] * T for s in per_clip]))
        else:
            # leave other non-array fields as-is
            batch[k] = v

    batch["num_video_frames"] = int(T)
    # if you need pose frames too:
    if "pose" in collated and isinstance(collated["pose"], torch.Tensor):
        batch["num_pose_frames"] = int(collated["pose"].shape[1])
    else:
        batch["num_pose_frames"] = int(T)

    return batch


class VideoDatasetModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=1, num_workers=1, shuffle=True,
            **kwargs):
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
    print(SINGLE_STAGE_TASK_DATASETS['PnPCounterToCab']['human_path'])