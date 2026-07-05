import math
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors
from torch.optim.lr_scheduler import LambdaLR

from ..modules import UNCONDITIONAL_CONFIG
from ..modules.autoencoding.temporal_ae import VideoDecoder
from ..modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from ..modules.ema import LitEma
from ..util import (default, disabled_train, get_obj_from_str,
                    instantiate_from_config, log_txt_as_img, append_dims)

import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from einops import rearrange, repeat
from torchvision.transforms import ToTensor

from scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark

import open_clip
import torch.distributed as dist

import pdb
from .helper import *
from .ddpo_helpers import *
try:
    from liv import load_liv
except Exception:
    load_liv = None
try:
    from liv.models.clip import clip as liv_clip
except Exception:
    try:
        import clip as liv_clip
    except Exception:
        liv_clip = None
try:
    from ..modules.critic_model.llava_critic import LlavaBertCritic
except Exception:
    LlavaBertCritic = None
import torch.nn.functional as F

TensorTree = Union[torch.Tensor, Dict[str, Any], List[Any], Tuple[Any, ...]]




class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        network_wrapper: Union[None, str] = None,
        ckpt_path: Union[None, str, Dict, ListConfig, OmegaConf] = None,
        use_ema: bool = False,
        ema_decay_rate: float = 0.9999,
        scale_factor: float = 1.0,
        disable_first_stage_autocast=False,
        input_key: list[str] = ["jpg", "pose"],
        log_keys: Union[List, None] = None,
        no_cond_log: bool = False,
        compile_model: bool = False,
        en_and_decode_n_samples_a_time: Optional[int] = None,
        vision_encoder_lr_scale: float = 1.0,
        pose_decoder_lr_scale: float = 1.0,
        grad_config: Dict = None,
        use_ddpo: bool = True,
        ddpo_lambda: float = 0.1,
        ddpo_steps: int = 2,
        ddpo_frames_for_reward: int = 3,
        ddpo_eta: float = 1.0,
        ddpo_every: int = 4,
        ddpo_alpha: float = 0.05,
        num_views: int = 3,
        frames_per_view: int = 8,
        num_video_frames: int = 25,
        rollout_seq_mb: int = 2,
        ddpo_seq_mb: int = 2,
        reward_decode_mb: int = 4,
        reward_down_hw: int = 224,
        reward_keyframe_index: int = 3,
        ddpo_amp_dtype: str = "fp16",
        reward_use_pil_pre: bool = False,
        reward_offload_after_score: bool = False,
        critic_type: str = "liv",
        llava_model_id: str = "llava-hf/llava-1.5-7b-hf",
        llava_question: str = "Describe the robot manipulation action taking place in this video.",
        val_cycle_enabled: bool = True,
        val_cycle_every: int = 50,
        val_cycle_max_seqs: int = 1,
    ):
        super().__init__()
        self.log_keys = log_keys
        self.input_key = input_key
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW", "phase":'stage_1'}
        )
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        self.scheduler_config = scheduler_config
        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model, decay=ema_decay_rate)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        self.vision_encoder_lr_scale = vision_encoder_lr_scale
        self.pose_decoder_lr_scale = pose_decoder_lr_scale
        self.grad_config = grad_config
        # Reward/critic model for DDPO steering, selectable via `critic_type`:
        #   "liv"   -> LIV image-language value model (per-frame cosine reward)
        #   "llava" -> LLaVA-1.5 + BERTScore critic (per-frame VLM reward)
        # Kept outside nn.Module registration so DDP/NCCL does not broadcast
        # their buffers during _sync_buffers.
        self.critic_type = str(critic_type).lower()
        self.llava_question = str(llava_question)
        object.__setattr__(self, "liv_model", None)
        object.__setattr__(self, "llava_critic", None)
        if self.critic_type == "liv":
            if load_liv is not None:
                try:
                    liv_model = load_liv()
                    liv_model.eval()
                    for p in liv_model.parameters():
                        p.requires_grad = False
                    object.__setattr__(self, "liv_model", liv_model)
                except Exception as exc:
                    print(f"[warn] Failed to initialize LIV reward model: {exc}")
        elif self.critic_type == "llava":
            if LlavaBertCritic is not None:
                try:
                    object.__setattr__(
                        self,
                        "llava_critic",
                        LlavaBertCritic(model_id=llava_model_id, num_frames=int(num_video_frames)),
                    )
                except Exception as exc:
                    print(f"[warn] Failed to initialize LLaVA critic: {exc}")
            else:
                print("[warn] critic_type='llava' but LlavaBertCritic import failed.")
        else:
            raise ValueError(
                f"Unknown critic_type={critic_type!r}; expected 'liv' or 'llava'."
            )
        self.use_ddpo = bool(use_ddpo)
        self.ddpo_lambda = float(ddpo_lambda)
        self.ddpo_steps = int(ddpo_steps)
        self.ddpo_frames_for_reward = int(ddpo_frames_for_reward)
        self.ddpo_eta = float(ddpo_eta)
        self.ddpo_every = int(ddpo_every)
        self.adv_ema = EMA(beta=0.9)         # running baseline for advantages
        self.ddpo_alpha = float(ddpo_alpha)


        self.num_views = int(num_views)
        self.frames_per_view = int(frames_per_view)
        self.num_video_frames = int(num_video_frames)   # ensure dataloader packs B*T divisible by this

        self.rollout_seq_mb = int(rollout_seq_mb)      # sequences per call during rollout
        self.ddpo_seq_mb = int(ddpo_seq_mb)            # sequences per call during DDPO
        self.reward_decode_mb = int(reward_decode_mb)  # frames decoded per reward chunk
        self.reward_down_hw = int(reward_down_hw)      # reward image resize
        self.reward_keyframe_index = int(reward_keyframe_index)
        self.ddpo_amp_dtype = str(ddpo_amp_dtype)      # "fp16" or "bf16"
        self.reward_use_pil_pre = bool(reward_use_pil_pre)
        self.reward_offload_after_score = bool(reward_offload_after_score)

        # lightweight validation-time consistency metric controls
        self.val_cycle_enabled = bool(val_cycle_enabled)
        self.val_cycle_every = int(val_cycle_every)
        self.val_cycle_max_seqs = int(val_cycle_max_seqs)


    # def _ensure_cyclereward_device(self, target_device: torch.device) -> torch.device:
    #     """
    #     Make the CycleReward model live on the same device as the video UNet inputs,
    #     unless you explicitly set a static device via self.reward_device_policy = "static".
    #     """
    #     policy = getattr(self, "reward_device_policy", "match_input")  # "match_input" | "static"
    #     if policy == "match_input":
    #         rm_device = target_device
    #     else:
    #         rm_device = torch.device(getattr(self, "reward_device", str(target_device)))

    #     # Move model; keep it in fp32 for numerical stability
    #     self.cyclereward_model.to(device=rm_device, dtype=torch.float32)
    #     # Some wrappers look at a `.device` attribute; set if present
    #     try:
    #         self.cyclereward_model.device = rm_device
    #     except Exception:
    #         pass
    #     return rm_device


    def grad_stats_for_loss(self, loss, model, scale: float = 1.0, retain_graph: bool = True):
        """
        Computes gradient stats for (scale * loss) w.r.t. model parameters,
        without writing into p.grad.

        Returns:
            total_norm: L2 norm over all params
            mean_abs:  mean of |grad| over all params
            max_abs:   max of |grad| over all params
        """
        params = [p for p in model.parameters() if p.requires_grad]
        scaled_loss = scale * loss

        grads = torch.autograd.grad(
            scaled_loss,
            params,
            retain_graph=retain_graph,
            allow_unused=True,
        )

        # Filter out Nones (params not used by this loss)
        grads = [g for g in grads if g is not None]

        if len(grads) == 0:
            return 0.0, 0.0, 0.0

        total_norm_sq = torch.tensor(0.0, device=grads[0].device)
        abs_means = []
        abs_maxes = []

        for g in grads:
            g_abs = g.detach().abs()
            total_norm_sq += (g_abs ** 2).sum()
            abs_means.append(g_abs.mean())
            abs_maxes.append(g_abs.max())

        total_norm = total_norm_sq.sqrt()
        mean_abs = torch.stack(abs_means).mean()
        max_abs = torch.stack(abs_maxes).max()

        return total_norm.item(), mean_abs.item(), max_abs.item()

    def _ensure_cycle_rm_device(self, target):
        # Backward-compatible alias used by older reward code paths.
        return self._cr_move_to(target)


    def _load_checkpoint_state_dict(self, path: str) -> Dict[str, torch.Tensor]:
        if path.endswith("ckpt"):
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                for key in ("state_dict", "model_state_dict", "model"):
                    v = obj.get(key, None)
                    if isinstance(v, dict):
                        return {k: t for k, t in v.items() if torch.is_tensor(t)}
                if all(isinstance(k, str) for k in obj.keys()) and any(torch.is_tensor(v) for v in obj.values()):
                    return {k: t for k, t in obj.items() if torch.is_tensor(t)}
            raise TypeError(f"Unsupported checkpoint payload from {path}: {type(obj)}")
        if path.endswith("safetensors"):
            return load_safetensors(path)
        raise NotImplementedError(f"Unsupported checkpoint extension for {path}")

    def _filter_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Keep compatibility with checkpoints carrying OpenCLIP keys we do not restore here.
        return {
            key: value
            for key, value in state_dict.items()
            if "conditioner.embedders.0.open_clip" not in key
        }

    def _load_state_dict_with_report(
        self, state_dict: Dict[str, torch.Tensor], source: str, strict: bool = False
    ) -> Tuple[List[str], List[str]]:
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        print(
            f"Restored from {source} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        return missing, unexpected

    def _extract_prefixed_state(
        self, state_dict: Dict[str, torch.Tensor], prefixes: List[str]
    ) -> Dict[str, torch.Tensor]:
        return {
            key: value
            for key, value in state_dict.items()
            if any(key.startswith(prefix) for prefix in prefixes)
        }

    def init_from_ckpt(
        self,
        path: Union[str, Dict, ListConfig, OmegaConf],
    ) -> None:
        if OmegaConf.is_config(path):
            path = OmegaConf.to_container(path, resolve=True)

        if isinstance(path, dict):
            # Dual-source init supported:
            # - video_unet_ckpt / stage1_ckpt / base_ckpt
            # - action_unet_ckpt / stage2_ckpt
            # If only stage2 is provided, default to full stage2 load.
            video_ckpt = path.get("video_unet_ckpt") or path.get("stage1_ckpt") or path.get("base_ckpt")
            stage2_ckpt = path.get("action_unet_ckpt") or path.get("stage2_ckpt")
            action_prefixes = path.get(
                "action_prefixes",
                [
                    "model.diffusion_model.pose_pred_net",
                    "model.diffusion_model.vision_encoder",
                ],
            )
            if isinstance(action_prefixes, (str, bytes)):
                action_prefixes = [str(action_prefixes)]
            else:
                action_prefixes = [str(x) for x in action_prefixes]

            # Backward compatible knob: load only action branches from stage2.
            # For full stage2 initialization (including VideoUNet), set True.
            stage2_full_load = bool(path.get("stage2_full_load", False))
            if ("stage2_full_load" not in path) and (stage2_ckpt is not None) and (video_ckpt is None):
                stage2_full_load = True

            # Stage-1 convenience: initialize only VideoUNet keys from a stage2 checkpoint.
            stage2_video_unet_only = bool(path.get("stage2_video_unet_only", False))
            video_prefixes = path.get("video_prefixes", ["model.diffusion_model"])
            if isinstance(video_prefixes, (str, bytes)):
                video_prefixes = [str(video_prefixes)]
            else:
                video_prefixes = [str(x) for x in video_prefixes]
            video_exclude_patterns = path.get(
                "video_exclude_patterns", ["pose_pred_net", "vision_encoder"]
            )
            if isinstance(video_exclude_patterns, (str, bytes)):
                video_exclude_patterns = [str(video_exclude_patterns)]
            else:
                video_exclude_patterns = [str(x) for x in video_exclude_patterns]

            if video_ckpt is None and stage2_ckpt is None:
                raise ValueError(
                    "ckpt_path dict must include at least one of "
                    "'video_unet_ckpt'/'stage1_ckpt'/'base_ckpt' or "
                    "'action_unet_ckpt'/'stage2_ckpt'."
                )

            if video_ckpt is not None:
                video_sd = self._filter_state_dict(self._load_checkpoint_state_dict(str(video_ckpt)))
                self._load_state_dict_with_report(video_sd, f"{video_ckpt} [base/video]")

            if stage2_ckpt is not None:
                stage2_sd_all = self._filter_state_dict(
                    self._load_checkpoint_state_dict(str(stage2_ckpt))
                )

                if stage2_video_unet_only:
                    stage2_video_sd = {
                        key: value
                        for key, value in stage2_sd_all.items()
                        if any(key.startswith(prefix) for prefix in video_prefixes)
                        and not any(pat in key for pat in video_exclude_patterns)
                    }
                    if len(stage2_video_sd) == 0:
                        raise RuntimeError(
                            "stage2_video_unet_only=True but no keys matched "
                            f"video_prefixes={video_prefixes} with "
                            f"video_exclude_patterns={video_exclude_patterns} in {stage2_ckpt}"
                        )
                    self._load_state_dict_with_report(
                        stage2_video_sd, f"{stage2_ckpt} [stage2 video-only]"
                    )
                elif stage2_full_load:
                    self._load_state_dict_with_report(
                        stage2_sd_all, f"{stage2_ckpt} [stage2 full]"
                    )
                else:
                    action_sd = self._extract_prefixed_state(stage2_sd_all, action_prefixes)
                    if len(action_sd) == 0:
                        # If prefixes miss, fall back to full stage2 so VideoUNet keys are not dropped.
                        print(
                            f"No keys matched prefixes {action_prefixes} in {stage2_ckpt}; "
                            "falling back to full stage2 load."
                        )
                        self._load_state_dict_with_report(
                            stage2_sd_all, f"{stage2_ckpt} [stage2 full fallback]"
                        )
                    else:
                        self._load_state_dict_with_report(
                            action_sd, f"{stage2_ckpt} [action overlay]"
                        )
            return

        if not isinstance(path, str):
            raise TypeError(f"Unsupported ckpt_path type: {type(path)}")

        sd = self._filter_state_dict(self._load_checkpoint_state_dict(path))
        self._load_state_dict_with_report(sd, path)

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in bchw format
        return {
        'video': batch[self.input_key[0]],
        'pose': batch[self.input_key[1]]
    }

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        use_cuda_amp = (z.device.type == "cuda") and (not self.disable_first_stage_autocast)
        amp_ctx = torch.autocast("cuda", enabled=use_cuda_amp) if z.device.type == "cuda" else nullcontext()
        with amp_ctx:
            for n in range(n_rounds):
                if isinstance(self.first_stage_model.decoder, VideoDecoder):
                    kwargs = {"timesteps": len(z[n * n_samples : (n + 1) * n_samples])}
                else:
                    kwargs = {}
                out = self.first_stage_model.decode(
                    z[n * n_samples : (n + 1) * n_samples], **kwargs
                )
                all_out.append(out)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        use_cuda_amp = (x.device.type == "cuda") and (not self.disable_first_stage_autocast)
        amp_ctx = torch.autocast("cuda", enabled=use_cuda_amp) if x.device.type == "cuda" else nullcontext()
        with amp_ctx:
            for n in range(n_rounds):
                out = self.first_stage_model.encode(
                    x[n * n_samples : (n + 1) * n_samples]
                )
                all_out.append(out)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z
        return z
    
    # def _init_cycle_reward(self, device):
    #     self.cyclereward_model, self.cyclereward_pre = cyclereward(device=device, model_type="CycleReward-Combo")

    def _decode_to_cpu_in_chunks(self, latents_bt, B, T, down_hw: int | None, chunk: int = 8):
        """
        Decode (B*T, C, H, W) latents -> CPU float tensor (B, T, C, H, W) in [0,1],
        using small GPU batches and freeing GPU memory each step.
        """
        cpu_chunks = []
        with torch.no_grad():
            for i in range(0, latents_bt.shape[0], chunk):
                z = latents_bt[i:i+chunk]                    # on GPU
                img = self.decode_first_stage(z)            # (mb, C, H, W) on GPU
                img = img.clamp(0, 1)
                if down_hw is not None:
                    img = F.interpolate(img, size=(down_hw, down_hw),
                                        mode="bilinear", align_corners=False)
                cpu_chunks.append(img.to("cpu", non_blocking=True))
                # free GPU ASAP
                del z, img
        imgs_cpu = torch.cat(cpu_chunks, dim=0)  # (B*T, C, H, W) on CPU
        C, H, W = imgs_cpu.shape[1:]
        return imgs_cpu.view(B, T, C, H, W)      # CPU



    # ----------------------------- small utils -----------------------------
    def _append_dims(self, x: torch.Tensor, target_ndim: int) -> torch.Tensor:
        while x.ndim < target_ndim:
            x = x.unsqueeze(-1)
        return x


    def _move_tree_to_device_dtype(self, obj: TensorTree, device: torch.device, x_dtype: torch.dtype) -> TensorTree:
        if torch.is_tensor(obj):
            dtype = x_dtype if torch.is_floating_point(obj) else obj.dtype
            return obj.to(device=device, dtype=dtype, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_tree_to_device_dtype(v, device, x_dtype) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_tree_to_device_dtype(v, device, x_dtype) for v in obj)
        return obj


    def _slice_bxt(self,obj: TensorTree, sl: slice, total: int) -> TensorTree:
        if torch.is_tensor(obj):
            return obj[sl] if (obj.dim() > 0 and obj.size(0) == total) else obj
        if isinstance(obj, dict):
            return {k: self._slice_bxt(v, sl, total) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)) and len(obj) == total and torch.is_tensor(obj[0]):
            return type(obj)(obj[sl])
        return obj


    def _slice_b_by(self,obj: TensorTree, b_sl: slice, B: int, device: torch.device, x_dtype: torch.dtype) -> TensorTree:
        # Slice objects whose first dim == B (per-sequence)
        if torch.is_tensor(obj):
            t = obj[b_sl] if (obj.dim() > 0 and obj.size(0) == B) else obj
            dt = x_dtype if torch.is_floating_point(t) else t.dtype
            return t.to(device=device, dtype=dt, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._slice_b_by(v, b_sl, B, device, x_dtype) for k, v in obj.items()}
        return obj


    def _slice_kwargs_by(self, kwargs_full: Dict[str, Any],
                        b_sl: slice,
                        bt_sl: slice,
                        B: int,
                        BxT: int,
                        device: torch.device,
                        x_dtype: torch.dtype) -> Dict[str, Any]:
        out = {}
        for k, v in kwargs_full.items():
            if k in ("num_video_frames", "num_pose_frames"):
                out[k] = int(v)  # scalar
            else:
                if torch.is_tensor(v):
                    if v.dim() > 0 and v.size(0) == B:
                        vv = v[b_sl]
                    elif v.dim() > 0 and v.size(0) == BxT:
                        vv = v[bt_sl]
                    else:
                        vv = v
                    dt = x_dtype if torch.is_floating_point(vv) else vv.dtype
                    out[k] = vv.to(device=device, dtype=dt, non_blocking=True)
                else:
                    out[k] = v
        return out


    @torch.no_grad()
    def _euler_ancestral_step(self, x_t, sigma_t, sigma_next, x0_pred, noise, eta: float = 1.0):
        d = (x_t - x0_pred) / (sigma_t + 1e-8)
        x_mean = x_t + (sigma_next - sigma_t) * d
        if eta != 0.0:
            var = (sigma_next**2 - sigma_t**2).clamp_min(0.0)
            x_next = x_mean + (eta * var.sqrt()) * noise
        else:
            x_next = x_mean
        return x_next, x_mean, d


    def _build_denoising_kwargs(self, add_inputs: dict, device: torch.device, x_dtype: torch.dtype) -> dict:
        # Keep ONLY: y, time_context, num_video_frames, num_pose_frames, image_only_indicator
        allowed = {"y", "time_context", "num_video_frames", "num_pose_frames", "image_only_indicator"}
        out = {}
        for k in allowed:
            if k not in add_inputs:
                continue
            v = add_inputs[k]
            if k in ("num_video_frames", "num_pose_frames"):
                out[k] = int(v)
            else:
                if torch.is_tensor(v):
                    dtype = x_dtype if torch.is_floating_point(v) else v.dtype
                    out[k] = v.to(device=device, dtype=dtype, non_blocking=True)
                else:
                    out[k] = v
        return out


    def _extract_prompts(self, d: dict, B: int, BxT: int, T: int) -> List[str]:
        if isinstance(d, dict):
            for key in ('orig_task_string', 'caption', 'text', 'prompt'):
                if key in d:
                    p = d[key]
                    if isinstance(p, (list, tuple)):
                        if len(p) == BxT:
                            p = p[::T]
                        elif len(p) != B:
                            p = list(p) + [p[-1]] * (B - len(p))
                    else:
                        p = [p] * B
                    return p
        return [""] * B

    def _to_unit_range(self, img: torch.Tensor) -> torch.Tensor:
        # Decoder outputs can be either [-1,1] or [0,1] depending on config.
        img_min = float(img.detach().amin().item())
        img_max = float(img.detach().amax().item())
        if img_min < 0.0 or img_max > 1.0:
            return (img.clamp(-1.0, 1.0) + 1.0) * 0.5
        return img.clamp(0.0, 1.0)

    def _is_global_rank_zero(self) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return True
        return dist.get_rank() == 0

    def _get_sequence_prompts(self, batch: Dict[str, Any], B: int, T_v: int) -> List[str]:
        prompts = batch.get("orig_task_string", "")
        if isinstance(prompts, (list, tuple)):
            if len(prompts) == B:
                return [str(x) for x in prompts]
            if len(prompts) == B * T_v:
                return [str(prompts[i * T_v]) for i in range(B)]
            if len(prompts) == 0:
                return [""] * B
            p2 = list(prompts)
            if len(p2) < B:
                p2 = p2 + [p2[-1]] * (B - len(p2))
            return [str(p2[i]) if i < len(p2) else "" for i in range(B)]
        return [str(prompts)] * B
    
    def _mk_sigmas(self, sig_scalar: torch.Tensor, m: int, ndim: int, device, dtype):
        sig1d = sig_scalar.expand(m).to(device=device, dtype=dtype).contiguous()  # shape [m]
        sigbd = self._append_dims(sig1d, ndim)                                        # shape [m,1,1,1,...]
        return sig1d, sigbd


    def _cr_move_to(self, target) -> torch.device:
        """
        Move LIV to the requested device and keep its internal `device` flag synced.
        """
        dev = target if isinstance(target, torch.device) else torch.device(target)
        if self.liv_model is None:
            raise RuntimeError(
                "LIV reward requested but LIV is not available. "
                "Install with `pip install LIV-robotics` and ensure imports succeed."
            )

        self.liv_model.to(device=dev)
        self.liv_model.eval()
        device_str = "cpu" if dev.type == "cpu" else f"cuda:{dev.index if dev.index is not None else torch.cuda.current_device()}"
        for obj in (self.liv_model, getattr(self.liv_model, "module", None)):
            if obj is None:
                continue
            try:
                setattr(obj, "device", device_str)
            except Exception:
                pass
        return dev

    def _cr_offload_cpu(self):
        """No-op: LIV is kept on GPU for training throughput."""
        return


    def _cr_score(self, imgs_pre: torch.Tensor, caps: list[str]):
        """Dispatch to the configured critic (``critic_type``): LIV or LLaVA.

        Returns one reward per input image, shape ``(N,)``, so every DDPO reward
        call-site stays agnostic to which critic is used.
        """
        if len(caps) != imgs_pre.shape[0]:
            raise ValueError(
                f"Expected one caption per image, got {len(caps)} captions for {imgs_pre.shape[0]} images."
            )
        if getattr(self, "critic_type", "liv") == "llava":
            return self._llava_score(imgs_pre, caps)
        return self._liv_score(imgs_pre, caps)

    def _liv_score(self, imgs_pre: torch.Tensor, caps: list[str]):
        """
        LIV reward: cosine similarity between per-image vision embeddings and prompt text embeddings.
        """
        if self.liv_model is None or liv_clip is None:
            raise RuntimeError(
                "LIV reward requested but LIV imports failed. "
                "Install with `pip install LIV-robotics`."
            )
        dev = imgs_pre.device
        self._cr_move_to(dev)
        imgs = self._to_unit_range(imgs_pre).float()
        caps = [str(c) for c in caps]

        with torch.inference_mode():
            tokenized = liv_clip.tokenize(caps).to(dev, non_blocking=True)
            img_embedding = self.liv_model(input=imgs, modality="vision")
            text_embedding = self.liv_model(input=tokenized, modality="text")
            core = self.liv_model.module if hasattr(self.liv_model, "module") else self.liv_model
            score = core.sim(img_embedding, text_embedding)
        return score.to(device=dev, dtype=torch.float32).view(-1)

    def _llava_score(self, imgs_pre: torch.Tensor, caps: list[str]):
        """
        LLaVA critic reward: LLaVA-1.5 answers a fixed question about each frame and
        the answer is compared to the task instruction (``caps``) via BERTScore F1.
        Returns one reward per image, shape ``(N,)``, matching the LIV interface.
        Note: this runs a LLaVA generation per frame and is much slower than LIV.
        """
        if self.llava_critic is None:
            raise RuntimeError(
                "LLaVA reward requested but LlavaBertCritic failed to initialize. "
                "Check the transformers/bert_score install and llava_model_id."
            )
        dev = imgs_pre.device
        imgs = self._to_unit_range(imgs_pre).float().clamp(0.0, 1.0)  # (N,C,H,W) in [0,1]
        imgs_np = imgs.permute(0, 2, 3, 1).detach().cpu().numpy()      # (N,H,W,3) float [0,1]
        scores = []
        for i in range(imgs_np.shape[0]):
            out = self.llava_critic.score_frames(
                frames=imgs_np[i][None],          # single frame as a 1-frame clip
                question=self.llava_question,
                reference=str(caps[i]),
            )
            scores.append(float(out["reward"]))
        return torch.tensor(scores, device=dev, dtype=torch.float32).view(-1)

    @torch.no_grad()
    def _compute_cyclereward_consistency_metric(
        self, latents_bt: torch.Tensor, batch: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        if not torch.is_tensor(latents_bt) or latents_bt.ndim < 4:
            return None

        if self.liv_model is None:
            return None

        device = latents_bt.device
        total = int(latents_bt.shape[0])
        T_v = int(batch.get("num_video_frames", getattr(self, "num_video_frames", 1)))
        if T_v <= 0 or total % T_v != 0:
            return None

        V = int(getattr(self, "num_views", 3))
        Fv = int(getattr(self, "frames_per_view", 8))
        mid_idx = int(getattr(self, "reward_keyframe_index", 3))
        if V * Fv > T_v:
            Fv = max(1, T_v // V)
        mid_idx = max(0, min(mid_idx, Fv - 1))

        B = total // T_v
        B_eval = min(B, max(1, int(getattr(self, "val_cycle_max_seqs", 1))))
        if B_eval <= 0:
            return None

        self._cr_move_to(device)
        decode_mb = max(1, int(getattr(self, "reward_decode_mb", 1)))
        down_hw = int(getattr(self, "reward_down_hw", 224))

        decode_indices: list[int] = []
        owners_b: list[int] = []
        prompts = self._get_sequence_prompts(batch, B, T_v)[:B_eval]
        for b in range(B_eval):
            base = b * T_v
            for v in range(V):
                t = v * Fv + mid_idx
                if t >= T_v:
                    continue
                decode_indices.append(base + t)
                owners_b.append(b)

        if len(decode_indices) == 0:
            return None

        score_sum = torch.zeros(B_eval, device=device, dtype=torch.float32)
        count = torch.zeros(B_eval, device=device, dtype=torch.float32)

        for s in range(0, len(decode_indices), decode_mb):
            e = min(s + decode_mb, len(decode_indices))
            idx = decode_indices[s:e]
            z = latents_bt[idx]
            img = self._to_unit_range(self.decode_first_stage(z))
            if down_hw is not None:
                img = F.interpolate(img, size=(down_hw, down_hw), mode="bilinear", align_corners=False)

            caps = [prompts[owners_b[i]] for i in range(s, e)]
            sc = self._cr_score(img, caps).to(device=device, dtype=torch.float32).view(-1)
            owners = torch.as_tensor(owners_b[s:e], device=device, dtype=torch.long)
            score_sum.index_add_(0, owners, sc)
            count.index_add_(0, owners, torch.ones_like(sc))
            del z, img, sc, owners

        return (score_sum / count.clamp_min(1.0)).mean()



    def _slice_cond_by(self,
                   cond_obj,
                   b_sl: slice,          # slice over sequences (B)
                   bt_sl_v: slice,       # slice over frames for VIDEO (B*T_v)
                   B: int,
                   BxT_v: int,
                   device: torch.device,
                   x_dtype: torch.dtype):
        """
        Recursively slice a conditioning tree so that any tensor whose first
        dimension equals B is sliced by `b_sl`, and any tensor whose first dimension
        equals BxT_v is sliced by `bt_sl_v`. Other tensors are passed through.
        """
        import torch
        if torch.is_tensor(cond_obj):
            t = cond_obj
            if t.dim() > 0:
                if t.size(0) == BxT_v:
                    t = t[bt_sl_v]
                elif t.size(0) == B:
                    t = t[b_sl]
            dt = x_dtype if torch.is_floating_point(t) else t.dtype
            return t.to(device=device, dtype=dt, non_blocking=True)
        if isinstance(cond_obj, dict):
            return {k: self._slice_cond_by(v, b_sl, bt_sl_v, B, BxT_v, device, x_dtype)
                    for k, v in cond_obj.items()}
        if isinstance(cond_obj, (list, tuple)):
            return type(cond_obj)(
                self._slice_cond_by(v, b_sl, bt_sl_v, B, BxT_v, device, x_dtype)
                for v in cond_obj
            )
        return cond_obj


    def get_cyclereward_loss_modified(
        self,
        loss_value: torch.Tensor,
        network,
        denoiser,
        conditioner,
        input: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        sigmas: Dict[str, torch.Tensor],
        noised_input: Dict[str, torch.Tensor],
        model_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns ONLY the DDPO term (caller does: total = supervised + returned_ddpo).

        Fixes:
        - rollout uses true Euler-Ancestral (sigma_down/sigma_up) for decreasing sigmas
        - DDPO logp matches the *same* kernel used in rollout (mean uses sigma_down, std uses sigma_up)
        - reward accumulators use float32 (index_add_ dtype safe)
        - no dependency on _HAS_PIL
        - baseline keys align with orig_task_string
        - returns ddpo_alpha * ddpo_loss
        """
        import math
        import torch
        import torch.nn.functional as F

        # ------------------- local helpers -------------------
        def mk_sigmas(sig_scalar: torch.Tensor, m: int, ndim: int, device, dtype):
            sig1d = sig_scalar.expand(m).to(device=device, dtype=dtype).contiguous()  # [m]
            sigbd = self._append_dims(sig1d, ndim)                                    # [m,1,1,1,...]
            return sig1d, sigbd

        @torch.no_grad()
        def euler_ancestral_step_fixed(x_t, sigma_t_bd, sigma_next_bd, x0_pred, noise, eta: float):
            """
            EDM-style Euler-Ancestral step for decreasing sigmas:
            sigma_next^2 = sigma_down^2 + sigma_up^2
            x_mean = x_t + (sigma_down - sigma_t) * d
            x_next = x_mean + sigma_up * noise
            """
            d = (x_t - x0_pred) / (sigma_t_bd + 1e-8)

            if eta == 0.0:
                x_next = x_t + (sigma_next_bd - sigma_t_bd) * d
                return x_next, x_next, d

            sigma_t = sigma_t_bd
            sigma_next = sigma_next_bd

            sigma_up = eta * torch.sqrt(torch.clamp(
                sigma_next**2 * (sigma_t**2 - sigma_next**2) / (sigma_t**2 + 1e-12),
                min=0.0
            ))
            sigma_down = torch.sqrt(torch.clamp(sigma_next**2 - sigma_up**2, min=0.0))


            x_mean = x_t + (sigma_down - sigma_t) * d
            x_next = x_mean + sigma_up * noise
            return x_next, x_mean, d

        def extract_prompt_keys_for_baseline(add_inputs: Dict[str, Any], B: int, BxT_v: int, T_v: int) -> list[str]:
            # Prefer the same text used for reward captions
            p = add_inputs.get("orig_task_string", None)
            if p is None:
                return self._extract_prompts(add_inputs, B, BxT_v, T_v)
            if isinstance(p, (list, tuple)):
                if len(p) == BxT_v:
                    return [str(p[i * T_v]) for i in range(B)]
                if len(p) == B:
                    return [str(x) for x in p]
                p2 = list(p)
                if len(p2) < B and len(p2) > 0:
                    p2 = p2 + [p2[-1]] * (B - len(p2))
                return [str(p2[i]) if i < len(p2) else "" for i in range(B)]
            return [str(p)] * B

        # ------------------- basics / config -------------------
        device = input["video"].device
        dtype = input["video"].dtype

        # move CycleReward model once up-front (also sets internal flags in your helper)
        self._cr_move_to(device)

        add_inputs = batch if isinstance(batch, dict) else {}

        V = int(getattr(self, "num_views", 3))
        Fv = int(getattr(self, "frames_per_view", 8))
        E = int(add_inputs.get("extra_frames", 1))
        T_v_decl = int(add_inputs.get("num_video_frames", getattr(self, "num_video_frames", V * Fv + E)))

        steps = int(getattr(self, "ddpo_steps", 1))
        eta = float(getattr(self, "ddpo_eta", 1.0))
        ddpo_alpha = float(getattr(self, "ddpo_alpha", 0.1))

        seq_mb_rollout = max(1, int(getattr(self, "rollout_seq_mb", 2)))
        seq_mb_ddpo = max(1, int(getattr(self, "ddpo_seq_mb", 2)))
        decode_mb = max(1, int(getattr(self, "reward_decode_mb", 6)))
        down_hw = int(getattr(self, "reward_down_hw", 224))
        mid_idx = int(getattr(self, "reward_keyframe_index", 3))

        amp_pref = str(getattr(self, "ddpo_amp_dtype", "fp16"))
        amp_dtype = torch.float16 if amp_pref == "fp16" else torch.bfloat16

        # ------------------- infer B, T_video -------------------
        BxT_v = int(input["video"].shape[0])
        if BxT_v % T_v_decl != 0:
            raise RuntimeError(
                f"[CR/DDPO] Packed video len {BxT_v} not divisible by declared T_video={T_v_decl}. "
                f"Fix dataloader or set batch['num_video_frames'] correctly."
            )
        T_v = T_v_decl
        B = BxT_v // T_v

        # reconcile V/Fv indexing for reward
        if V * Fv > T_v:
            Fv_eff = max(1, T_v // V)
            E = max(0, T_v - V * Fv_eff)
            Fv = Fv_eff
        mid_idx = max(0, min(mid_idx, Fv - 1))

        # ------------------- sigmas & inputs -------------------
        video_sigmas: torch.Tensor = sigmas["video_sigmas"].to(device=device)
        pose_sigmas: Optional[torch.Tensor] = sigmas.get("pose_sigmas", None)

        x_t: torch.Tensor = noised_input["noised_video_input"].to(device=device, dtype=dtype)
        noised_pose = noised_input.get("noised_pose_input", None)

        # ------------------- infer T_pose independently -------------------
        BxT_p = None
        if pose_sigmas is not None and torch.is_tensor(pose_sigmas) and pose_sigmas.dim() > 0:
            if pose_sigmas.shape[0] % B == 0:
                BxT_p = int(pose_sigmas.shape[0])
        if BxT_p is None and (noised_pose is not None) and torch.is_tensor(noised_pose):
            if noised_pose.shape[0] % B == 0:
                BxT_p = int(noised_pose.shape[0])
        if BxT_p is None:
            BxT_p = BxT_v
        T_p = BxT_p // B

        # Ensure pose tensors exist & are on device
        if noised_pose is None or (torch.is_tensor(noised_pose) and noised_pose.shape[0] != BxT_p):
            noised_pose = torch.zeros(BxT_p, *x_t.shape[1:], device=device, dtype=dtype)
        else:
            noised_pose = self._move_tree_to_device_dtype(noised_pose, device=device, x_dtype=dtype)

        if pose_sigmas is None or (torch.is_tensor(pose_sigmas) and pose_sigmas.shape[0] != BxT_p):
            pose_sigmas = video_sigmas
            if pose_sigmas.shape[0] != BxT_p:
                if pose_sigmas.shape[0] == BxT_v and BxT_v != BxT_p:
                    repeat_factor = math.ceil(BxT_p / BxT_v)
                    pose_sigmas = pose_sigmas.repeat(repeat_factor)[:BxT_p]
                else:
                    pose_sigmas = pose_sigmas[:BxT_p]
        pose_sigmas = pose_sigmas.to(device=device)

        # conditioner (CUDA)
        cond = conditioner(batch)
        cond = self._move_tree_to_device_dtype(cond, device=device, x_dtype=dtype)

        # kwargs
        denoiser_kwargs_full = self._build_denoising_kwargs(add_inputs, device, dtype)
        denoiser_kwargs_full["num_video_frames"] = int(T_v)
        denoiser_kwargs_full["num_pose_frames"] = int(T_p)

        # sigma schedule (decreasing)
        s_max = float(video_sigmas.max().item())
        sig_sched = torch.linspace(s_max, 0.0, steps + 1, device=device, dtype=dtype)

        # ------------------- rollout (sequence-aligned; kernel matches logp) -------------------
        traj = []
        with torch.no_grad():
            xt = x_t.detach()
            for k in range(steps):
                xt_next_full = torch.empty_like(xt)
                b0 = 0
                while b0 < B:
                    b1 = min(b0 + seq_mb_rollout, B)

                    i_v, j_v = b0 * T_v, b1 * T_v
                    i_p, j_p = b0 * T_p, b1 * T_p
                    m_v = j_v - i_v

                    sig1d_t_v, sig_t_v_bd = mk_sigmas(sig_sched[k],   m_v, xt[i_v:j_v].ndim, device, dtype)
                    sig1d_n_v, sig_n_v_bd = mk_sigmas(sig_sched[k+1], m_v, xt[i_v:j_v].ndim, device, dtype)

                    b_sl = slice(b0, b1)
                    bt_sl_v = slice(i_v, j_v)
                    bt_sl_p = slice(i_p, j_p)

                    cond_chunk = self._slice_cond_by(cond, b_sl, bt_sl_v, B, BxT_v, device, xt.dtype)
                    kwargs_chunk = self._slice_kwargs_by(denoiser_kwargs_full, b_sl, bt_sl_v, B, BxT_v, device, xt.dtype)

                    out_k = denoiser(
                        network,
                        {
                            "noised_video_input": xt[i_v:j_v],
                            "noised_pose_input": self._slice_bxt(noised_pose, bt_sl_p, BxT_p),
                        },
                        {
                            "video_sigmas": sig1d_t_v,
                            "pose_sigmas": self._slice_bxt(pose_sigmas, bt_sl_p, BxT_p),
                        },
                        cond_chunk,
                        **kwargs_chunk,
                    )
                    x0_pred = out_k["video_output"] if isinstance(out_k, dict) else out_k

                    eps = torch.randn_like(xt[i_v:j_v])
                    x_next, _, _ = euler_ancestral_step_fixed(
                        xt[i_v:j_v], sig_t_v_bd, sig_n_v_bd, x0_pred, noise=eps, eta=eta
                    )
                    xt_next_full[i_v:j_v] = x_next
                    b0 = b1

                traj.append(
                    {
                        "xt_cpu": xt.detach().to("cpu"),
                        "xtm1_cpu": xt_next_full.detach().to("cpu"),
                        "sigma_t_scalar": float(sig_sched[k].item()),
                        "sigma_next_scalar": float(sig_sched[k + 1].item()),
                    }
                )
                xt = xt_next_full

        # ------------------- reward: 1 keyframe/view on CUDA -------------------
        self._cr_move_to(device)

        decode_indices, owners_b, owners_v = [], [], []
        for b in range(B):
            base = b * T_v
            for v in range(V):
                t = v * Fv + mid_idx
                if t >= T_v:
                    continue
                decode_indices.append(base + t)
                owners_b.append(b)
                owners_v.append(v)

        # float32 accumulators (index_add safe)
        rewards_sum_flat = torch.zeros(B * V, device=device, dtype=torch.float32)
        counts_flat = torch.zeros(B * V, device=device, dtype=torch.float32)

        with torch.inference_mode():
            for s in range(0, len(decode_indices), decode_mb):
                e = min(s + decode_mb, len(decode_indices))
                idx = decode_indices[s:e]

                z = xt[idx]
                img = self._to_unit_range(self.decode_first_stage(z))
                if down_hw is not None:
                    img = F.interpolate(img, size=(down_hw, down_hw), mode="bilinear", align_corners=False)

                if isinstance(batch.get("orig_task_string", None), (list, tuple)):
                    caps = [batch["orig_task_string"][owners_b[i]] for i in range(s, e)]
                else:
                    caps = [batch.get("orig_task_string", "")] * (e - s)

                sc = self._cr_score(img, caps).to(device=device, dtype=torch.float32).view(-1)

                b_chunk = torch.as_tensor(owners_b[s:e], device=device, dtype=torch.long)
                v_chunk = torch.as_tensor(owners_v[s:e], device=device, dtype=torch.long)
                flat_idx = b_chunk * V + v_chunk

                rewards_sum_flat.index_add_(0, flat_idx, sc)
                counts_flat.index_add_(0, flat_idx, torch.ones_like(sc))

                del z, img, sc, b_chunk, v_chunk, flat_idx

        rewards_per_view = (rewards_sum_flat / counts_flat.clamp_min(1.0)).view(B, V)
        reward = rewards_per_view.mean(dim=1)  # (B,)

        # ------------------- per-prompt normalization -> advantages -------------------
        prompts = extract_prompt_keys_for_baseline(add_inputs, B, BxT_v, T_v)
        if not hasattr(self, "_reward_stats"):
            self._reward_stats = {}

        adv = torch.empty(B, device=device, dtype=torch.float16)
        with torch.no_grad():
            for b in range(B):
                key = str(prompts[b])
                st = self._reward_stats.get(key, {"mean": 0.0, "M2": 0.0, "n": 0})
                n1 = st["n"] + 1
                rb = float(reward[b].item())
                delta = rb - st["mean"]
                mean1 = st["mean"] + delta / n1
                M21 = st["M2"] + delta * (rb - mean1)
                st = {"mean": mean1, "M2": M21, "n": n1}
                self._reward_stats[key] = st
                std = math.sqrt(M21 / (n1 - 1)) if n1 > 1 else 1.0
                adv[b] = (reward[b] - mean1) / max(std, 1e-6)

            adv.clamp_(-3.0, 3.0)
        # ------------------- DDPO: logp matches rollout kernel -------------------
        ddpo_loss = torch.zeros((), device=device, dtype=loss_value.dtype)

        for rec in traj:
            s_t = torch.tensor(rec["sigma_t_scalar"], device=device, dtype=dtype)
            s_n = torch.tensor(rec["sigma_next_scalar"], device=device, dtype=dtype)

            logp_sum_B = torch.zeros(B, device=device, dtype=loss_value.dtype)
            cnt_B = torch.zeros(B, device=device, dtype=loss_value.dtype)

            b0 = 0
            while b0 < B:
                b1 = min(b0 + seq_mb_ddpo, B)
                i_v, j_v = b0 * T_v, b1 * T_v
                i_p, j_p = b0 * T_p, b1 * T_p
                m_v = j_v - i_v

                xt = rec["xt_cpu"][i_v:j_v].to(device=device, non_blocking=True)
                xtm1 = rec["xtm1_cpu"][i_v:j_v].to(device=device, non_blocking=True)

                sig1d_t_v, sig_t_v_bd = mk_sigmas(s_t, m_v, xt.ndim, device, xt.dtype)
                sig1d_n_v, sig_n_v_bd = mk_sigmas(s_n, m_v, xt.ndim, device, xt.dtype)

                b_sl = slice(b0, b1)
                bt_sl_v = slice(i_v, j_v)
                bt_sl_p = slice(i_p, j_p)

                cond_chunk = self._slice_cond_by(cond, b_sl, bt_sl_v, B, BxT_v, device, xt.dtype)
                kwargs_chunk = self._slice_kwargs_by(denoiser_kwargs_full, b_sl, bt_sl_v, B, BxT_v, device, xt.dtype)

                pose_sig_chunk = self._slice_bxt(pose_sigmas, bt_sl_p, BxT_p)
                pose_chunk = self._slice_bxt(noised_pose, bt_sl_p, BxT_p).to(device=device, dtype=xt.dtype, non_blocking=True)

                with (torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda"
                    else nullcontext()):
                    out = denoiser(
                        network,
                        {"noised_video_input": xt, "noised_pose_input": pose_chunk},
                        {"video_sigmas": sig1d_t_v, "pose_sigmas": pose_sig_chunk},
                        cond_chunk,
                        **kwargs_chunk,
                    )
                    x0_pred = out["video_output"] if isinstance(out, dict) else out

                # logp of the *same* Euler-Ancestral kernel used in rollout
                xt32 = xt.float()
                xtm1_32 = xtm1.float()
                sigma_t = sig_t_v_bd.float()
                sigma_next = sig_n_v_bd.float()

                d32 = (xt32 - x0_pred.float()) / (sigma_t + 1e-8)

                if eta == 0.0:
                    logp_flat = torch.zeros(m_v, device=device, dtype=loss_value.dtype)
                else:
                    sigma_up = eta * torch.sqrt(torch.clamp(
                        sigma_next**2 * (sigma_t**2 - sigma_next**2) / (sigma_t**2 + 1e-12),
                        min=0.0
                    ))
                    sigma_down = torch.sqrt(torch.clamp(sigma_next**2 - sigma_up**2, min=0.0))

                    x_mean32 = xt32 + (sigma_down - sigma_t) * d32

                    sigma_up_safe = sigma_up.clamp_min(1e-6)  # clamp only if you also clamp in rollout
                    resid = (xtm1_32 - x_mean32) / sigma_up_safe
                    resid = resid.clamp_(-10.0, 10.0)

                    resid2_mean = resid.view(m_v, -1).pow(2).mean(dim=1)
                    log_sigma = torch.log(sigma_up_safe.reshape(m_v) + 1e-12)
                    logp_flat = (-0.5 * resid2_mean - log_sigma).to(loss_value.dtype)

                logp_seq = logp_flat.view(b1 - b0, T_v).mean(dim=1)

                idx = torch.arange(b0, b1, device=device)
                logp_sum_B.index_add_(0, idx, logp_seq)
                cnt_B.index_add_(0, idx, torch.ones_like(logp_seq))

                del xt, xtm1, sig1d_t_v, sig1d_n_v, sig_t_v_bd, sig_n_v_bd, out, x0_pred
                del cond_chunk, kwargs_chunk, pose_sig_chunk, pose_chunk
                del xt32, xtm1_32, sigma_t, sigma_next, d32, logp_flat, logp_seq
                b0 = b1

            logp_B = logp_sum_B / cnt_B.clamp_min(1.0)
            ddpo_loss = ddpo_loss + (-(adv.detach().to(loss_value.dtype) * logp_B).mean())

        del cond
        return ddpo_loss


    # --------------------------- main public function ---------------------------
    def get_cyclereward_loss(
        self,
        loss_value: torch.Tensor,
        network,
        denoiser,
        conditioner,
        input: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        sigmas: Dict[str, torch.Tensor],
        noised_input: Dict[str, torch.Tensor],
        model_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns ONLY the DDPO term (so the caller can do: total = supervised + returned_ddpo).
        Key change vs your version: video and pose can have DIFFERENT T (T_video != T_pose).
        We infer both and pass the correct num_*_frames + slices for each stream.
        """
        import torch
        import torch.nn.functional as F

        # ------------------- local helpers -------------------
        def mk_sigmas(sig_scalar: torch.Tensor, m: int, ndim: int, device, dtype):
            sig1d = sig_scalar.expand(m).to(device=device, dtype=dtype).contiguous()  # [m]
            sigbd = self._append_dims(sig1d, ndim)                                    # [m,1,1,1,...]
            return sig1d, sigbd

        def build_kwargs_chunk(kwargs_full: Dict[str, Any],
                            b_sl: slice,
                            bt_sl_v: slice, BxT_v: int,
                            bt_sl_p: slice, BxT_p: int,
                            B: int, device, x_dtype):
            """
            Slice kwargs by B, BxT_video, or BxT_pose based on their leading dim.
            """
            out = {}
            for k, v in kwargs_full.items():
                if k == "num_video_frames":
                    out[k] = int(T_v)
                elif k == "num_pose_frames":
                    out[k] = int(T_p)
                else:
                    if torch.is_tensor(v):
                        if v.dim() > 0:
                            if v.size(0) == B:
                                vv = v[b_sl]
                            elif v.size(0) == BxT_v:
                                vv = v[bt_sl_v]
                            elif v.size(0) == BxT_p:
                                vv = v[bt_sl_p]
                            else:
                                vv = v
                        else:
                            vv = v
                        dt = x_dtype if torch.is_floating_point(vv) else vv.dtype
                        out[k] = vv.to(device=device, dtype=dt, non_blocking=True)
                    else:
                        out[k] = v
            return out

        # ------------------- basics / config -------------------
        device = input['video'].device
        dtype= input['video'].dtype

        # dtype  = input['video'].dtype
        self._cr_move_to(device)
        add_inputs = batch if isinstance(batch, dict) else {}

        V   = int(getattr(self, "num_views", 3))
        Fv  = int(getattr(self, "frames_per_view", 8))
        E   = int(add_inputs.get("extra_frames", 1))
        T_v_decl = int(add_inputs.get("num_video_frames", getattr(self, "num_video_frames", V*Fv + E)))
        steps   = int(getattr(self, "ddpo_steps", 1))
        eta     = float(getattr(self, "ddpo_eta", 1.0))
        ddpo_alpha = float(getattr(self, "ddpo_alpha", 0.1))

        seq_mb_rollout = max(1, int(getattr(self, "rollout_seq_mb", 2)))
        seq_mb_ddpo    = max(1, int(getattr(self, "ddpo_seq_mb", 2)))
        decode_mb      = max(1, int(getattr(self, "reward_decode_mb", 6)))
        down_hw        = int(getattr(self, "reward_down_hw", 224))
        mid_idx        = int(getattr(self, "reward_keyframe_index", 3))
        amp_pref       = str(getattr(self, "ddpo_amp_dtype", "fp16"))
        amp_dtype      = torch.float16 if amp_pref == "fp16" else torch.bfloat16

        # ------------------- infer B, T_video -------------------
        BxT_v = int(input['video'].shape[0])
        if BxT_v % T_v_decl != 0:
            raise RuntimeError(f"[CR/DDPO] Packed video len {BxT_v} not divisible by declared T_video={T_v_decl}. "
                            f"Fix dataloader or set batch['num_video_frames'] correctly.")
        T_v = T_v_decl
        B   = BxT_v // T_v

        # views/frames reconciliation for reward indexing
        if V * Fv > T_v:
            Fv_eff = max(1, T_v // V)
            E = max(0, T_v - V * Fv_eff)
            Fv = Fv_eff
        mid_idx = max(0, min(mid_idx, Fv - 1))

        # ------------------- sigmas & inputs -------------------
        video_sigmas: torch.Tensor = sigmas['video_sigmas'].to(device=device)
        pose_sigmas:  Optional[torch.Tensor] = sigmas.get('pose_sigmas', None)
        x_t: torch.Tensor = noised_input['noised_video_input'].to(device=device, dtype=dtype)
        noised_pose = noised_input.get('noised_pose_input', None)

        # ------------------- infer T_pose independently -------------------
        # Prefer pose_sigmas length; else noised_pose; else fallback to video
        BxT_p = None
        if pose_sigmas is not None and torch.is_tensor(pose_sigmas) and pose_sigmas.dim() > 0:
            if pose_sigmas.shape[0] % B == 0:
                BxT_p = int(pose_sigmas.shape[0])
        if BxT_p is None and (noised_pose is not None) and torch.is_tensor(noised_pose):
            if noised_pose.shape[0] % B == 0:
                BxT_p = int(noised_pose.shape[0])
        if BxT_p is None:
            BxT_p = BxT_v  # fallback: tie pose to video
        T_p = BxT_p // B

        # Make sure pose tensors exist & are on device
        if noised_pose is None or (torch.is_tensor(noised_pose) and noised_pose.shape[0] != BxT_p):
            # synthesize dummy pose that matches (B, T_p)
            noised_pose = torch.zeros(BxT_p, *x_t.shape[1:], device=device, dtype=dtype)
        else:
            noised_pose = self._move_tree_to_device_dtype(noised_pose, device=device, x_dtype=dtype)

        if pose_sigmas is None or (torch.is_tensor(pose_sigmas) and pose_sigmas.shape[0] != BxT_p):
            pose_sigmas = video_sigmas
            if pose_sigmas.shape[0] != BxT_p:
                # last resort: expand or slice to match length
                if pose_sigmas.shape[0] == BxT_v and BxT_v != BxT_p:
                    # simple proportional repeat/truncate to match; keeps device/dtype
                    repeat_factor = math.ceil(BxT_p / BxT_v)
                    pose_sigmas = pose_sigmas.repeat(repeat_factor)[:BxT_p]
                else:
                    pose_sigmas = pose_sigmas[:BxT_p]
        pose_sigmas = pose_sigmas.to(device=device)

        # conditioner (CUDA)
        cond = conditioner(batch)
        cond = self._move_tree_to_device_dtype(cond, device=device, x_dtype=dtype)

        # kwargs (set correct frame counts for BOTH streams)
        denoiser_kwargs_full = self._build_denoising_kwargs(add_inputs, device, dtype)
        denoiser_kwargs_full['num_video_frames'] = int(T_v)
        denoiser_kwargs_full['num_pose_frames']  = int(T_p)

        # schedule for video
        s_max = float(video_sigmas.max().item()); s_min = 0.0
        sig_sched = torch.linspace(s_max, s_min, steps + 1, device=device, dtype=dtype)

        # ------------------- rollout (sequence-aligned; video/pose decoupled) -------------------
        traj = []
        with torch.no_grad():
            xt = x_t.detach()
            for k in range(steps):
                xt_next_full = torch.empty_like(xt)
                b0 = 0
                while b0 < B:
                    b1 = min(b0 + seq_mb_rollout, B)

                    # indices for video & pose in this chunk
                    i_v, j_v = b0 * T_v, b1 * T_v
                    i_p, j_p = b0 * T_p, b1 * T_p
                    m_v = j_v - i_v
                    m_p = j_p - i_p
                    assert (m_v % T_v) == 0, f"rollout video chunk {m_v} not divisible by T_v={T_v}"
                    assert (m_p % T_p) == 0, f"rollout pose  chunk {m_p} not divisible by T_p={T_p}"

                    sig1d_t_v, sig_t_v = mk_sigmas(sig_sched[k],   m_v, xt[i_v:j_v].ndim, device, dtype)
                    sig1d_n_v, sig_n_v = mk_sigmas(sig_sched[k+1], m_v, xt[i_v:j_v].ndim, device, dtype)

                    # Build cond/kwargs slices
                    b_sl         = slice(b0, b1)
                    bt_sl_v      = slice(i_v, j_v)
                    bt_sl_p      = slice(i_p, j_p)
                    # cond_chunk   = self._slice_b_by(cond, b_sl, B, device, xt.dtype)
                    # kwargs_chunk = build_kwargs_chunk(denoiser_kwargs_full, b_sl, bt_sl_v, BxT_v, bt_sl_p, BxT_p, B, device, xt.dtype)

                    b_sl    = slice(b0, b1)
                    vi = b0*T_v
                    vj = b1*T_v
                    bt_sl_v = slice(vi, vj)   # vi = b0*T_v, vj = b1*T_v

                    cond_chunk = self._slice_cond_by(cond, b_sl, bt_sl_v, B, BxT_v, device, xt.dtype)

                    # If your kwargs can be per‑pose too, use your build_kwargs_chunk that accepts BxT_v and BxT_p.
                    # Otherwise keep your existing kwargs slicing for video-aligned tensors:
                    kwargs_chunk = self._slice_kwargs_by(denoiser_kwargs_full, b_sl, bt_sl_v, B, BxT_v, device, xt.dtype)


                    out_k = denoiser(
                        network,
                        {'noised_video_input': xt[i_v:j_v],
                        'noised_pose_input' : self._slice_bxt(noised_pose, bt_sl_p, BxT_p)},
                        {'video_sigmas': sig1d_t_v,
                        'pose_sigmas' : self._slice_bxt(pose_sigmas, bt_sl_p, BxT_p)},
                        cond_chunk,
                        **kwargs_chunk
                    )
                    x0_pred = out_k['video_output'] if isinstance(out_k, dict) else out_k

                    eps = torch.randn_like(xt[i_v:j_v])
                    x_next, _, _ = self._euler_ancestral_step(xt[i_v:j_v], sig_t_v, sig_n_v, x0_pred, noise=eps, eta=eta)
                    xt_next_full[i_v:j_v] = x_next
                    b0 = b1

                traj.append({
                    'xt_cpu'           : xt.detach().to('cpu'),
                    'xtm1_cpu'         : xt_next_full.detach().to('cpu'),
                    'sigma_t_scalar'   : float(sig_sched[k].item()),
                    'sigma_next_scalar': float(sig_sched[k+1].item()),
                })
                xt = xt_next_full

        # ------------------- reward: 1 keyframe/view on CUDA -------------------
        self._cr_move_to(device)

        # decode indices use video timeline (T_v)
        decode_indices, owners_b, owners_v = [], [], []
        for b in range(B):
            base = b * T_v
            for v in range(V):
                t = v * Fv + mid_idx
                if t >= T_v:  # safe-guard
                    continue
                decode_indices.append(base + t)
                owners_b.append(b); owners_v.append(v)

        rewards_sum_flat = torch.zeros(B * V, device=device, dtype=torch.float32)
        counts_flat      = torch.zeros(B * V, device=device, dtype=torch.float32)

        with torch.inference_mode():
            for s in range(0, len(decode_indices), decode_mb):
                e = min(s + decode_mb, len(decode_indices))
                idx = decode_indices[s:e]

                z   = xt[idx]                          # (mb, C, H, W) CUDA
                img = self._to_unit_range(self.decode_first_stage(z))
                if down_hw is not None:
                    img = F.interpolate(img, size=(down_hw, down_hw), mode="bilinear", align_corners=False)

                if isinstance(batch.get('orig_task_string', None), (list, tuple)):
                    caps = [batch['orig_task_string'][owners_b[i]] for i in range(s, e)]
                else:
                    caps = [batch.get('orig_task_string', "")] * (e - s)

                sc = self._cr_score(img, caps).to(device=device, dtype=torch.float32).view(-1)

                b_chunk = torch.as_tensor(owners_b[s:e], device=device, dtype=torch.long)
                v_chunk = torch.as_tensor(owners_v[s:e], device=device, dtype=torch.long)
                flat_idx = b_chunk * V + v_chunk
                rewards_sum_flat.index_add_(0, flat_idx, sc)
                counts_flat.index_add_(0, flat_idx, torch.ones_like(sc))

                del z, img, sc, b_chunk, v_chunk, flat_idx

        rewards_per_view = (rewards_sum_flat / counts_flat.clamp_min(1.0)).view(B, V)
        reward = rewards_per_view.mean(dim=1)  # (B,)
        del rewards_sum_flat, counts_flat, decode_indices, owners_b, owners_v
        # ------------------- per-prompt normalization -> advantages -------------------
        prompts = self._extract_prompts(add_inputs, B, BxT_v, T_v)
        if not hasattr(self, "_reward_stats"):
            self._reward_stats = {}

        adv = torch.empty_like(reward, dtype=torch.float16, device=device)
        with torch.no_grad():
            for b in range(B):
                key = str(prompts[b])
                st = self._reward_stats.get(key, {"mean": 0.0, "M2": 0.0, "n": 0})
                n1 = st["n"] + 1
                delta = float(reward[b].item()) - st["mean"]
                mean1 = st["mean"] + delta / n1
                M21 = st["M2"] + delta * (float(reward[b].item()) - mean1)
                st = {"mean": mean1, "M2": M21, "n": n1}
                self._reward_stats[key] = st
                std = math.sqrt(M21 / (n1 - 1)) if n1 > 1 else 1.0
                adv[b] = (reward[b] - mean1) / max(std, 1e-6)
        
        ################################################
        with torch.no_grad():
            #adv = (adv - adv.mean()) / (adv.std() + 1e-6)  # center + scale
            adv = adv.clamp_(-3.0, 3.0)                    # keep it bounded



        # ------------------- DDPO: sequence-aligned (video/pose decoupled) -------------------
        ddpo_loss = torch.zeros((), device=device, dtype=loss_value.dtype)

        for rec in traj:
            s_t = torch.tensor(rec['sigma_t_scalar'],   device=device, dtype=dtype)
            s_n = torch.tensor(rec['sigma_next_scalar'], device=device, dtype=dtype)

            logp_sum_B = torch.zeros(B, device=device, dtype=loss_value.dtype)
            cnt_B      = torch.zeros(B, device=device, dtype=loss_value.dtype)

            b0 = 0
            while b0 < B:
                b1 = min(b0 + seq_mb_ddpo, B)
                i_v, j_v = b0 * T_v, b1 * T_v
                i_p, j_p = b0 * T_p, b1 * T_p
                m_v = j_v - i_v
                m_p = j_p - i_p
                assert (m_v % T_v) == 0, f"ddpo video chunk {m_v} not divisible by T_v={T_v}"
                assert (m_p % T_p) == 0, f"ddpo pose  chunk {m_p} not divisible by T_p={T_p}"

                xt   = rec['xt_cpu'][i_v:j_v].to(device=device, non_blocking=True)
                xtm1 = rec['xtm1_cpu'][i_v:j_v].to(device=device, non_blocking=True)

                sig1d_t_v, sig_t_v = mk_sigmas(s_t, m_v, xt.ndim, device, xt.dtype)
                sig1d_n_v, sig_n_v = mk_sigmas(s_n, m_v, xt.ndim, device, xt.dtype)

                b_sl         = slice(b0, b1)
                bt_sl_v      = slice(i_v, j_v)
                bt_sl_p      = slice(i_p, j_p)

                cond_chunk = self._slice_cond_by(cond, b_sl, bt_sl_v, B, BxT_v, device, xt.dtype)
                kwargs_chunk = self._slice_kwargs_by(denoiser_kwargs_full, b_sl, bt_sl_v, B, BxT_v, device, xt.dtype)



                pose_sig_chunk = self._slice_bxt(pose_sigmas, bt_sl_p, BxT_p)
                pose_chunk     = self._slice_bxt(noised_pose, bt_sl_p, BxT_p).to(device=device, dtype=xt.dtype, non_blocking=True)

                with (torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda"
                    else nullcontext()):
                    out = denoiser(
                        network,
                        {'noised_video_input': xt,
                        'noised_pose_input' : pose_chunk},
                        {'video_sigmas': sig1d_t_v,
                        'pose_sigmas' : pose_sig_chunk},
                        cond_chunk, **kwargs_chunk
                    )
                    x0_pred = out['video_output'] if isinstance(out, dict) else out

                    d = (xt - x0_pred) / (sig_t_v + 1e-8)
                    x_mean = xt + (sig_n_v - sig_t_v) * d

                    #############################################
                    # --- compute in float32 for stability ---
                    xt32      = xt.float()
                    xtm1_32   = xtm1.float()
                    sig_t_v32 = sig_t_v.float()
                    sig_n_v32 = sig_n_v.float()

                    # variance of the Euler-Ancestral kernel
                    # var = (sig_n_v32**2 - sig_t_v32**2).clamp_min(0.0)

                    # # make the kernel floor *meaningful* (avoid tiny σ)
                    # sigma_kernel = (eta * var.sqrt()).clamp_min(1e-3)  # <-- was 1e-8


                    var = (sig_t_v32**2 - sig_n_v32**2).clamp_min(0.0)
                    sigma_kernel = (eta * var.sqrt()).clamp_min(1e-3)


                    # residual in float32, with mild clipping
                    resid = (xtm1_32 - (xt32 + (sig_n_v32 - sig_t_v32) * ((xt32 - x0_pred.float()) / (sig_t_v32 + 1e-8)) )) / sigma_kernel
                    resid = resid.clamp_(-10.0, 10.0)

                    # number of elements per sample (per frame)
                    m_v = xt.shape[0]                # already defined in your loop
                    numel_per = resid[0].numel()     # C*H*W  (float)


                    resid2 = resid.pow(2).view(m_v, -1).sum(dim=1)          # sum over pixels
                    numel  = resid[0].numel()
                    log_sigma_sum = torch.log(sigma_kernel.view(m_v, -1) + 1e-12).sum(dim=1)
                    logp_flat = (-0.5 * resid2 - log_sigma_sum).to(loss_value.dtype) / numel


                    # log_sigma = torch.log(sigma_kernel.view(m_v, -1) + 1e-12).mean(dim=1)
                    # logp_flat = (-0.5 * resid.pow(2).view(m_v, -1).mean(dim=1) - log_sigma).to(loss_value.dtype)

                    # per-sequence average over frames
                    logp_seq = logp_flat.view(b1 - b0, T_v).mean(dim=1)


                    # var = (sig_n_v**2 - sig_t_v**2).clamp_min(0.0)
                    # sigma_kernel = (eta * var.sqrt()).clamp_min(1e-8)

                    # resid = (xtm1 - x_mean) / sigma_kernel
                    # logp_flat = (-0.5 * resid.pow(2).flatten(1).sum(dim=1)).to(loss_value.dtype)  # (m_v,)
                    # logp_seq  = logp_flat.view(b1 - b0, T_v).mean(dim=1)                          # (seq_mb,)

                idx = torch.arange(b0, b1, device=device)
                logp_sum_B.index_add_(0, idx, logp_seq)
                cnt_B.index_add_(0, idx, torch.ones_like(logp_seq))

                del xt, xtm1, sig1d_t_v, sig1d_n_v, sig_t_v, sig_n_v, out, x0_pred, d, x_mean, var, sigma_kernel, resid, logp_flat, logp_seq, cond_chunk, kwargs_chunk, pose_sig_chunk, pose_chunk
                b0 = b1

            logp_B = logp_sum_B / cnt_B.clamp_min(1.0)
            ddpo_loss = ddpo_loss + (-(adv.detach().to(loss_value.dtype) * logp_B).mean())
        del cond

        return ddpo_loss

    def forward(self, x, batch):
        loss, sigmas, noised_input, model_output = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()     # tensor[25] -> tensor[1]
        cycle_reward_loss = torch.zeros_like(loss_mean)
        ddpo_every = max(1, int(getattr(self, "ddpo_every", 1)))
        run_ddpo = bool(getattr(self, "use_ddpo", True)) and self.training and (int(self.global_step) % ddpo_every == 0)
        if run_ddpo and (self.liv_model is not None):
            cycle_reward_loss = self.ddpo_alpha * self.get_cyclereward_loss_modified(
                loss_mean, self.model, self.denoiser, self.conditioner, x, batch, sigmas, noised_input, model_output
            )
        loss_total = loss_mean + cycle_reward_loss
        action_only = bool(getattr(self.loss_fn, "action_only_loss", False))
        if action_only:
            loss_dict = {"loss": loss_total, "loss_action_sft": loss_mean, "loss_ddpo": cycle_reward_loss}
        else:
            loss_dict = {"loss": loss_total, "loss_diffusion": loss_mean, "loss_ddpo": cycle_reward_loss}
        return loss_total, loss_dict

    def shared_step(self, batch: Dict) -> Any:
        x = self.get_input(batch)
        x['video'] = self.encode_first_stage(x['video'])  # return tensor[25, 4, 40, 56] x∈[-5.454, 5.389] μ=0.100 σ=1.124
        batch["global_step"] = self.global_step
        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        detached = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
        if "loss_ddpo" in detached:
            ddpo_val = detached.pop("loss_ddpo")
            self.log(
                "loss_ddpo",
                ddpo_val,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )
        self.log_dict(detached, prog_bar=True, logger=True, on_step=True, on_epoch=False)


        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        x["video"] = self.encode_first_stage(x["video"])
        loss, sigmas, noised_input, model_output = self.loss_fn(
            self.model, self.denoiser, self.conditioner, x, batch
        )

        loss_mean = loss.mean()
        sync_dist_flag = bool(dist.is_available() and dist.is_initialized())
        if sync_dist_flag:
            try:
                # NCCL cannot all-reduce CPU tensors.
                if dist.get_backend() == "nccl" and loss_mean.device.type != "cuda":
                    sync_dist_flag = False
            except Exception:
                pass
        action_only = bool(getattr(self.loss_fn, "action_only_loss", False))
        val_key = "val_loss_action_sft" if action_only else "val_loss_diffusion"
        self.log(
            val_key,
            loss_mean.detach(),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=sync_dist_flag,
        )

        run_cycle = bool(getattr(self, "val_cycle_enabled", True))
        val_every = max(1, int(getattr(self, "val_cycle_every", 50)))
        if run_cycle and (batch_idx % val_every == 0) and self._is_global_rank_zero():
            latents = model_output["video_output"] if isinstance(model_output, dict) else model_output
            score = self._compute_cyclereward_consistency_metric(latents.detach(), batch)
            if score is not None:
                self.log(
                    "val_liv_consistency",
                    score.detach(),
                    prog_bar=True,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
                self.log(
                    "val_cyclereward_consistency",
                    score.detach(),
                    prog_bar=True,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )
        return loss_mean

    def on_train_start(self, *args, **kwargs):
        if self.sampler is None or self.loss_fn is None:
            raise ValueError("Sampler and loss function need to be set for training.")
        if bool(getattr(self, "use_ddpo", False)) and self._is_global_rank_zero():
            _active_critic = (
                self.llava_critic if getattr(self, "critic_type", "liv") == "llava"
                else self.liv_model
            )
            if _active_critic is None:
                print(
                    f"[warn] use_ddpo=True but the '{getattr(self, 'critic_type', 'liv')}' "
                    "reward model is unavailable; loss_ddpo will stay zero. "
                    "Check the critic install (LIV-robotics for 'liv'; transformers/bert_score for 'llava')."
                )

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate

        if self.optimizer_config.phase == "stage_1":
            param_groups = []

            # 0) Freeze all params first, then selectively unfreeze only:
            #    - full blocks at indices [9, 14, 17, 20, 23]
            #    - VideoUNet cross-attention params
            for _, p in self.model.named_parameters():
                p.requires_grad = False

            vu = find_videounet(self.model)
            blocks = list(vu.input_blocks) + list(vu.output_blocks)
            stage1_block_indices = [9, 14, 17, 20, 23]
            seen = set()

            # 1) Full params from selected VideoUNet blocks.
            if self.grad_config.get("train_video_unet", True):
                selected_block_params = []
                for idx in stage1_block_indices:
                    if idx >= len(blocks):
                        print(f"[warn] stage1 block index {idx} out of range ({len(blocks)} blocks)")
                        continue
                    for p in blocks[idx].parameters():
                        if id(p) in seen:
                            continue
                        p.requires_grad = True
                        selected_block_params.append(p)
                        seen.add(id(p))
                if selected_block_params:
                    param_groups.append({"params": selected_block_params, "lr": lr})

            # 2) Cross-attention params across VideoUNet blocks.
            def iter_cross_attn_params(module):
                import torch.nn as nn
                for name, sub in module.named_modules():
                    lname = name.lower()
                    # canonical BasicTransformerBlock.attn2
                    if hasattr(sub, "attn2") and isinstance(sub.attn2, nn.Module):
                        for p in sub.attn2.parameters():
                            yield p
                    # name patterns used by various repos
                    if any(tag in lname for tag in ("attn2", "cross_attn", "crossattn", "xattn")):
                        for p in sub.parameters():
                            yield p
                    # CrossAttention-like modules (to_q/to_k/to_v/to_out)
                    if all(hasattr(sub, k) for k in ("to_q", "to_k", "to_v")) and hasattr(sub, "to_out"):
                        for p in sub.parameters():
                            yield p
                    # minimal glue so grads can flow even if cross-attn is conditionally skipped
                    if any(tag in lname for tag in ("to_out", "out_proj", "proj_out", "ln", "norm")):
                        for p in sub.parameters():
                            yield p

            if self.grad_config.get("train_video_unet_cross_attn", True):
                cross_attn_params = []
                for idx in range(len(blocks)):
                    for p in iter_cross_attn_params(blocks[idx]):
                        pid = id(p)
                        if pid in seen:
                            continue
                        p.requires_grad = True
                        cross_attn_params.append(p)
                        seen.add(pid)

                assert cross_attn_params, "No cross-attn params matched; check names/indices."
                param_groups.append({"params": cross_attn_params, "lr": lr})

        else:

            param_groups = []

            # Hard-freeze everything first, then selectively unfreeze.
            for _, p in self.model.named_parameters():
                p.requires_grad = False

            # Add parameters for specific parts of the model with different learning rates.
            for name, param in self.model.named_parameters():
                if "vision_encoder" in name:
                    if self.grad_config['train_vision_encoder']:
                        param.requires_grad = True
                        param_groups.append({"params": param, "lr": lr * self.vision_encoder_lr_scale})
                elif "pose_pred_net" in name:
                    if self.grad_config['train_pose_pred_net']:
                        param.requires_grad = True
                        param_groups.append({"params": param, "lr": lr * self.pose_decoder_lr_scale})
                else:
                    if self.grad_config['train_video_unet']:
                        param.requires_grad = True
                        param_groups.append({"params": param, "lr": lr})

        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                # with open('logs/params.txt', "a") as file:
                #     file.write(f"embedder added" + "\n")
                param_groups.append({"params": list(embedder.parameters()), "lr": lr})
        
        # for g in param_groups:
        #     g["params"] = [p for p in g["params"] if p.requires_grad]
        #param_groups = [g for g in param_groups if g["params"]]

        # Instantiate the optimizer with parameter groups
        opt = self.instantiate_optimizer_from_config(param_groups, lr, self.optimizer_config)

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(self.device)
        denoiser = lambda input, sigma, c: self.denoiser(
            self.model, input, sigma, c, **kwargs
        )
        samples = self.sampler(denoiser, randn, cond, uc=uc)
        return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key[0]].shape[2:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        xc = x
                        # raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_images(
        self,
        batch: Dict,
        N: int = 8,
        sample: bool = True,
        ucg_keys: List[str] = None,
        **kwargs,
    ) -> Dict:
        conditioner_input_keys = [e.input_key for e in self.conditioner.embedders]
        if ucg_keys:
            assert all(map(lambda x: x in conditioner_input_keys, ucg_keys)), (
                "Each defined ucg key for sampling must be in the provided conditioner input keys,"
                f"but we have {ucg_keys} vs. {conditioner_input_keys}"
            )
        else:
            ucg_keys = conditioner_input_keys
        log = dict()

        x = self.get_input(batch)

        c, uc = self.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=ucg_keys
            if len(self.conditioner.embedders) > 0
            else [],
        )

        # sampling_kwargs = {}
        sampling_kwargs = {
            key: batch[key] for key in ['num_video_frames', 'image_only_indicator']
        }

        N = min(x.shape[0], N)
        x = x.to(self.device)[:N]
        log["inputs"] = x
        z = self.encode_first_stage(x)
        log["reconstructions"] = self.decode_first_stage(z)
        # log.update(self.log_conditionings(batch, N))
        
        return log
