"""
Feature extraction utility for UMAP / t-SNE analysis of decoder representations.

Hooks the U-Net bottleneck (`middle_block`) and captures the activation each time
the diffusion model is called. The final call (lowest sigma) overwrites prior
captures, so `last_feature` holds the cleanest-noise feature for that rollout.

Task-category mapping below is derived from Table 1 of the ECCV 2026 RoboTALES
submission (paper #11562).
"""

import os
from collections import OrderedDict
import numpy as np
import torch


TASK_TO_CATEGORY = {
    "PnPCabToCounter":       "Pick and Place",
    "PnPCounterToCab":       "Pick and Place",
    "PnPCounterToMicrowave": "Pick and Place",
    "PnPCounterToSink":      "Pick and Place",
    "PnPCounterToStove":     "Pick and Place",
    "PnPMicrowaveToCounter": "Pick and Place",
    "PnPSinkToCounter":      "Pick and Place",
    "PnPStoveToCounter":     "Pick and Place",
    "OpenSingleDoor":        "Doors",
    "OpenDoubleDoor":        "Doors",
    "CloseSingleDoor":       "Doors",
    "CloseDoubleDoor":       "Doors",
    "OpenDrawer":            "Drawers",
    "CloseDrawer":           "Drawers",
    "TurnOnStove":           "Twisting Knobs",
    "TurnOffStove":          "Twisting Knobs",
    "TurnOnSinkFaucet":      "Turning Levers",
    "TurnOffSinkFaucet":     "Turning Levers",
    "TurnSinkSpout":         "Turning Levers",
    "CoffeePressButton":     "Pressing Buttons",
    "TurnOnMicrowave":       "Pressing Buttons",
    "TurnOffMicrowave":      "Pressing Buttons",
    "CoffeeServeMug":        "Insertion",
    "CoffeeSetupMug":        "Insertion",
}


def task_category(task_name: str) -> str:
    return TASK_TO_CATEGORY.get(task_name, "Unknown")


class BottleneckFeatureExtractor:
    """
    Registers a forward hook on the U-Net `middle_block` and stores the most
    recent output. Mean-pools spatial+temporal dims to a 1D vector on demand.

    Usage:
        extractor = BottleneckFeatureExtractor(model)
        # ... run sampling ...
        feat = extractor.pop()  # 1D numpy array, or None if nothing captured
    """

    def __init__(self, model):
        unet = model.model.diffusion_model
        self._handle = unet.middle_block.register_forward_hook(self._hook)
        self._buffer = None

    def _hook(self, module, inputs, output):
        # output shape: (B*T, C, H, W). Detach + move to CPU lazily on pop.
        self._buffer = output.detach()

    def reset(self):
        self._buffer = None

    def pop(self):
        """Return mean-pooled 1D feature (numpy) and clear the buffer."""
        if self._buffer is None:
            return None
        feat = self._buffer.float().mean(dim=(0, 2, 3)).cpu().numpy()
        self._buffer = None
        return feat

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class FeatureLog:
    """Accumulates per-rollout features and metadata, saves to .npz."""

    def __init__(self):
        self.records = []

    def add(self, feature, task_name, demo_id, model_id):
        if feature is None:
            return
        self.records.append({
            "feature": np.asarray(feature, dtype=np.float32),
            "task_name": str(task_name),
            "category": task_category(task_name),
            "demo_id": str(demo_id),
            "model_id": str(model_id),
        })

    def save(self, path):
        if not self.records:
            print(f"[FeatureLog] no new records to save at {path}")
            return
        features = np.stack([r["feature"] for r in self.records], axis=0)
        task_names = np.array([r["task_name"] for r in self.records])
        categories = np.array([r["category"] for r in self.records])
        demo_ids = np.array([r["demo_id"] for r in self.records])
        model_ids = np.array([r["model_id"] for r in self.records])

        # If an existing file is present, merge instead of overwriting so reruns
        # (e.g. after a crash) accumulate features rather than discarding prior ones.
        if os.path.exists(path):
            try:
                prior = np.load(path, allow_pickle=False)
                # De-dup on (model_id, task_name, demo_id) keeping new records.
                new_keys = set(zip(model_ids.tolist(), task_names.tolist(), demo_ids.tolist()))
                prior_models = prior["model_ids"].astype(str)
                prior_tasks = prior["task_names"].astype(str)
                prior_demos = prior["demo_ids"].astype(str)
                keep = np.array([
                    (m, t, d) not in new_keys
                    for m, t, d in zip(prior_models, prior_tasks, prior_demos)
                ])
                if keep.any():
                    features = np.concatenate([prior["features"][keep], features], axis=0)
                    task_names = np.concatenate([prior_tasks[keep], task_names], axis=0)
                    categories = np.concatenate([prior["categories"].astype(str)[keep], categories], axis=0)
                    demo_ids = np.concatenate([prior_demos[keep], demo_ids], axis=0)
                    model_ids = np.concatenate([prior_models[keep], model_ids], axis=0)
                print(f"[FeatureLog] merging with {int(keep.sum())} prior records at {path}")
            except Exception as e:
                print(f"[FeatureLog] could not merge prior file ({e}); overwriting.")

        np.savez(
            path,
            features=features,
            task_names=task_names,
            categories=categories,
            demo_ids=demo_ids,
            model_ids=model_ids,
        )
        print(f"[FeatureLog] saved {len(features)} total records to {path}")
