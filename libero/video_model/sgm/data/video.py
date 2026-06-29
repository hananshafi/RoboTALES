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

SINGLE_STAGE_TASK_DATASETS = OrderedDict(
    LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo=dict(
        horizon=0,
        path="../datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5",
    ),
    LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo=dict(
        horizon=0,
        path="../datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5",
    ),
    KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo=dict(
        horizon=0,
        path="../datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
    ),
    KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo=dict(
        horizon=0,
        path="../datasets/libero_10/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5",
    ),
    LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo=dict(
        horizon=0,
        path="../datasets/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5",
    ),
    STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo=dict(
        horizon=0,
        path="../datasets/libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5",
    ),
    LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo=dict(
        horizon=0,
        path="../datasets/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5",
    ),
    LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo=dict(
        horizon=0,
        path="../datasets/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5",
    ),
    KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo=dict(
        horizon=0,
        path="../datasets/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5",
    ),
    KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo=dict(
        horizon=0,
        path="../datasets/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5",
    ),
)

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
        sample_fps: float,
        video_fps: float,
        pred_horizon: int,
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
        self.video_stride = round(video_fps / sample_fps)
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.swap_rgb = swap_rgb

        if self.mode == 'train':
            self.transform_rgb = T.Compose([
                T.ColorJitter(brightness=aug['brightness'], contrast=aug['contrast'], saturation=aug['saturation_rgb'], hue=aug['hue_rgb']),  # Random color jitter
            ])
        elif self.mode == 'test':
            self.transform_rgb = T.Compose([
            ])

        self.task_list = list(tasks.keys())

        if self.mode == 'test':
            # Inference path: only convert_frame and static params are used.
            # Skip hdf5 loading so unseen task suites (zero-shot eval) work.
            self.datasets, self.hdf5_datasets = [], []
            self.indexed_demos = []
        else:
            self.datasets, self.hdf5_datasets = self.get_dataset_file(self.task_list)

            self.indexed_demos = []
            for task_index, task_name in enumerate(self.task_list):

                task_data = self.datasets[task_index]['data']
                for demo_key in task_data.keys():
                    if (task_name in skip_demos and demo_key in skip_demos[task_name]):     # skip invalid demos with robot base actions
                        continue

                    task_description = json.loads(self.hdf5_datasets[task_index]["data"].attrs["problem_info"])["language_instruction"]
                    task_description = open_clip.tokenize([task_description]) # returns torch.Size([1, 77])

                    demo_steps = range(0, task_data[demo_key]['actions'].shape[0])
                    for demo_step in demo_steps:
                        self.indexed_demos.append((task_name, task_index, demo_key, demo_step, task_description))

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

            path = SINGLE_STAGE_TASK_DATASETS[task]['path']

            in_memory_data = self.load_hdf5_into_memory(path)

            datasets.append(in_memory_data)

            hdf5_file = h5py.File(path, 'r')
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
            task_name, task_index, demo_key, demo_step, task_description = self.indexed_demos[i]
            relative_actions_abs = self.hdf5_datasets[task_index]['data'][demo_key]['actions'][demo_step:demo_step+self.pred_horizon*self.video_stride][:, 0:self.action_dim]
            
            pad_size = self.pred_horizon*self.video_stride - relative_actions_abs.shape[0]

            if pad_size > 0:
                relative_actions_abs = np.concatenate([relative_actions_abs, np.zeros((pad_size, self.action_dim))], axis=0)

            relative_actions_abs_normalized = relative_actions_abs

            agentview_rgb = self.hdf5_datasets[task_index]['data'][demo_key]['obs']['agentview_rgb'][demo_step:demo_step+self.pred_horizon*self.video_stride:self.video_stride]
            eye_in_hand_rgb = self.hdf5_datasets[task_index]['data'][demo_key]['obs']['eye_in_hand_rgb'][demo_step:demo_step+self.pred_horizon*self.video_stride:self.video_stride]

            pad_size = self.pred_horizon - agentview_rgb.shape[0]
            if pad_size > 0:
                last_element = np.expand_dims(agentview_rgb[-1], axis=0)
                padding = np.concatenate([last_element] * pad_size, axis=0)
                agentview_rgb = np.concatenate([agentview_rgb, padding], axis=0)

                last_element = np.expand_dims(eye_in_hand_rgb[-1], axis=0)
                padding = np.concatenate([last_element] * pad_size, axis=0)
                eye_in_hand_rgb = np.concatenate([eye_in_hand_rgb, padding], axis=0)

            agentview_rgb = np.stack([self.convert_frame(frame=frame, size=(self.frame_width,self.frame_height), swap_rgb=self.swap_rgb) for frame in agentview_rgb])
            eye_in_hand_rgb = np.stack([self.convert_frame(frame=frame, size=(self.frame_width,self.frame_height), swap_rgb=self.swap_rgb) for frame in eye_in_hand_rgb])

        except Exception as e:
            print(f'sample retrive exception: {e}')
            
        agentview_rgb = torch.tensor(agentview_rgb, dtype=torch.float32)
        eye_in_hand_rgb = torch.tensor(eye_in_hand_rgb, dtype=torch.float32)
        relative_actions_abs_normalized = torch.tensor(relative_actions_abs_normalized, dtype=torch.float32)

        # Rescale from [-1, 1] to [0, 1] for transforms
        agentview_rgb = (agentview_rgb + 1) / 2
        eye_in_hand_rgb = (eye_in_hand_rgb + 1) / 2

        agentview_rgb, _ = self.augmentation_transform(agentview_rgb, self.transform_rgb)
        eye_in_hand_rgb, _ = self.augmentation_transform(eye_in_hand_rgb, self.transform_rgb)

        # Rescale back to [-1, 1]
        agentview_rgb = agentview_rgb * 2 - 1
        eye_in_hand_rgb = eye_in_hand_rgb * 2 - 1

        agentview_rgb = agentview_rgb.numpy()
        eye_in_hand_rgb = eye_in_hand_rgb.numpy()
        relative_actions_abs_normalized = relative_actions_abs_normalized.numpy()
        
        video_data = np.concatenate((eye_in_hand_rgb[0:1], eye_in_hand_rgb, agentview_rgb), axis=0)
        cond_frames = eye_in_hand_rgb[0:1]
        cond_frames_2 = eye_in_hand_rgb[0:1]
        cond_frames_3 = agentview_rgb[0:1]

        cond_frames = (cond_frames + self.cond_aug * np.random.randn(*cond_frames.shape))
        cond_frames_2 = (cond_frames_2 + self.cond_aug * np.random.randn(*cond_frames_2.shape))
        cond_frames_3 = (cond_frames_3 + self.cond_aug * np.random.randn(*cond_frames_3.shape))

        cond_frames_without_noise = task_description.numpy()
        cond_frames_without_noise = cond_frames_without_noise.repeat(self.n_frames-1, axis=0)

        cond_aug = np.ones(shape=(self.n_frames-1,)) * self.cond_aug
        motion_bucket_id = np.ones(shape=(self.n_frames-1,), dtype=np.int32) * self.motion_bucket_id
        fps_id = np.ones(shape=(self.n_frames-1,), dtype=np.int32) * self.fps_id
        image_only_indicator = np.zeros(shape=(1, self.n_frames-1,))

        return {
            "jpg": video_data.astype(np.float32),
            "cond_frames": cond_frames.astype(np.float32),
            "cond_frames_2": cond_frames_2.astype(np.float32),
            "cond_frames_3": cond_frames_3.astype(np.float32),
            "cond_frames_without_noise": cond_frames_without_noise,
            "cond_aug": cond_aug.astype(np.float32),
            "motion_bucket_id": motion_bucket_id,
            "fps_id": fps_id,
            "image_only_indicator": image_only_indicator.astype(np.float32),
            "pose": relative_actions_abs_normalized.astype(np.float32)  # [t, 7]
        }
    
    def __len__(self):
        return len(self.indexed_demos)

def collate_fn(example_list):
    collated = default_collate(example_list)
    batch = {k: rearrange(v, "b t ... -> (b t) ...") for (k, v) in collated.items()}
    batch["num_video_frames"] = 25
    batch["num_pose_frames"] = 36
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

