#!/usr/bin/env python3
"""Launch RoboCasa action-generator SFT using GT actions only."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def str2bool(v: str) -> bool:
    return v.lower() in {"1", "true", "t", "yes", "y"}


def _parse_device_ids(spec: str) -> Optional[List[int]]:
    s = spec.strip()
    if not s:
        return None
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if "," not in s:
        return None
    toks = [t.strip() for t in s.split(",") if t.strip()]
    if not toks:
        return None
    if not all(t.isdigit() for t in toks):
        return None
    return [int(t) for t in toks]


def _format_devices_for_main(device_ids: List[int]) -> str:
    if len(device_ids) == 1:
        # Preserve legacy single-GPU string style expected by main.py config.
        return f"{device_ids[0]},"
    return ",".join(str(i) for i in device_ids)


def _resolve_devices(parser: argparse.ArgumentParser, devices: Optional[str], num_gpus: Optional[int]) -> str:
    if devices is not None and num_gpus is not None:
        parser.error("Use either --devices or --num-gpus, not both.")

    if num_gpus is not None:
        if num_gpus < 1:
            parser.error("--num-gpus must be >= 1.")
        return _format_devices_for_main(list(range(num_gpus)))

    if devices is None:
        return "1"

    s = devices.strip()
    if not s:
        parser.error("--devices cannot be empty.")

    # Accept legacy "0," format and normalize to list.
    parsed_ids = _parse_device_ids(s)
    if parsed_ids is not None:
        return _format_devices_for_main(parsed_ids)

    # Integer string is interpreted as number of GPUs.
    if s.isdigit():
        if int(s) < 1:
            parser.error("--devices as integer must be >= 1.")
        return _format_devices_for_main(list(range(int(s))))

    # Pass through for advanced cases.
    return s


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SFT only action generator (vision_encoder + pose_pred_net) on RoboCasa GT actions."
    )
    parser.add_argument(
        "--base-config",
        default="configs/stage_2_action_decoder_training_sft_action_only_robocasa.yaml",
        help="Base config relative to video_model/.",
    )
    parser.add_argument("--name", default="sft_action_generator_robocasa")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--wandb", type=str2bool, default=True)
    parser.add_argument(
        "--devices",
        default=None,
        help='Lightning device spec: count (e.g. "4") or IDs (e.g. "0,1,2,3" or "[0,1,2,3]").',
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use. Alternative to --devices.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help='Optional CUDA_VISIBLE_DEVICES, e.g. "0" or "0,1,2,3".',
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--ckpt",
        default=None,
        help=(
            "Single checkpoint containing both VideoUNet + action branches. "
            "When set, overrides model.params.ckpt_path directly."
        ),
    )
    parser.add_argument(
        "--video-unet-ckpt",
        default=None,
        help="Optional override for model.params.ckpt_path.video_unet_ckpt.",
    )
    parser.add_argument(
        "--action-unet-ckpt",
        default=None,
        help="Optional override for model.params.ckpt_path.action_unet_ckpt.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args, extra_overrides = parser.parse_known_args()
    resolved_devices = _resolve_devices(parser, args.devices, args.num_gpus)

    video_model_dir = Path(__file__).resolve().parents[2]
    base_config = args.base_config
    if not os.path.isabs(base_config):
        base_config = str(video_model_dir / base_config)

    cmd = [
        sys.executable,
        "-u",
        "main.py",
        f"--base={base_config}",
        f"--name={args.name}",
        f"--seed={args.seed}",
        f"--num_nodes={args.num_nodes}",
        f"--wandb={1 if args.wandb else 0}",
        f"lightning.trainer.devices={resolved_devices}",
        "data.target=sgm.data.video.VideoDatasetModule",
        "data.params.mode=train",
        "model.params.optimizer_config.phase=stage_2",
        "model.params.grad_config.train_vision_encoder=true",
        "model.params.grad_config.train_pose_pred_net=true",
        "model.params.grad_config.train_video_unet=false",
        "model.params.grad_config.train_video_unet_cross_attn=false",
        "model.params.loss_fn_config.params.use_action_loss=true",
        "model.params.loss_fn_config.params.action_only_loss=true",
    ]

    if args.ckpt and (args.video_unet_ckpt or args.action_unet_ckpt):
        parser.error("--ckpt cannot be combined with --video-unet-ckpt / --action-unet-ckpt.")

    if args.ckpt:
        cmd.append(f"model.params.ckpt_path={args.ckpt}")

    if args.batch_size is not None:
        cmd.append(f"data.params.batch_size={args.batch_size}")
    if args.max_epochs is not None:
        cmd.append(f"lightning.trainer.max_epochs={args.max_epochs}")
    if (not args.ckpt) and args.video_unet_ckpt:
        cmd.append(f"model.params.ckpt_path.video_unet_ckpt={args.video_unet_ckpt}")
    if (not args.ckpt) and args.action_unet_ckpt:
        cmd.append(f"model.params.ckpt_path.action_unet_ckpt={args.action_unet_ckpt}")

    cmd.extend(extra_overrides)

    env = os.environ.copy()
    base_pythonpath = ".:..:../packages/robocasa:../packages/robosuite:../packages/robomimic"
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = f"{base_pythonpath}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = base_pythonpath
    env.setdefault("PYTHONBREAKPOINT", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    printable_cmd = " ".join(shlex.quote(c) for c in cmd)
    print(f"[sft_action_generator_robocasa] cwd={video_model_dir}")
    if args.cuda_visible_devices is not None:
        print(f"[sft_action_generator_robocasa] CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
    print(f"[sft_action_generator_robocasa] cmd={printable_cmd}")

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=str(video_model_dir), env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
