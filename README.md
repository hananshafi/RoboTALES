# RoboTALES: Learning Reasoning-Guided Robot Policies via Task-Aligned Simulated Futures

### ECCV 2026
### [Project Page](TODO) | [Paper](TODO) | [arXiv](TODO)

> Anonymous ECCV 2026 Submission — Paper ID #11562

<p align="center">
  <a href="assets/iccv-2026-teaser.pdf"><img src="assets/iccv-2026-teaser.png" width="80%"></a>
</p>

## 📖 Overview

Pretrained video generative models are promising backbones for visuomotor control, but their
imagined futures often drift from task intent and are not reliably action-conditional, which makes
them hard to use for planning or action extraction. **RoboTALES** is a **single-stage** framework
that learns *task-aligned* simulated futures and uses them to train robot policies. It introduces
two key ideas:

1. **Hierarchical LLM Planner.** A reasoning LLM decomposes a complex task instruction into an
   ordered sequence of sub-goals that condition the video model's imagination, turning
   undifferentiated prediction into structured, milestone-driven simulation.
2. **VLM-based Critic (reward-guided steering).** A frozen vision-language critic evaluates the
   "imagined" futures against the task instruction and feeds reward-based signal back into the video
   generator's hidden states via **differentiable policy optimization (DDPO)**, keeping the model's
   internal representations focused on the goal.

By anchoring the video generator in abstract reasoning and steering its representations with the
critic, RoboTALES produces temporally consistent rollouts and more coherent actions. Crucially, the
video generator and the action policy are optimized **jointly in a single stage**, so action-level
gradients flow back into the video generator's decoder layers — the world model learns to "imagine
for acting" while the policy learns to "act from imagination."

We evaluate on diverse manipulation tasks from **RoboCasa** and **LIBERO-10**, where RoboTALES
consistently outperforms existing methods, especially on long-horizon tasks (e.g. 48% mean success
on challenging RoboCasa Pick-and-Place, and 64% / 96% on multi-step turning / pressing).

### Method at a glance

RoboTALES couples four components (see Figure 2 in the paper):

| Component | Role | Where in the code |
|---|---|---|
| **LLM Planner** `F_P` | Decomposes instruction `τ` into `K∈[2,5]` sub-goals → augmented plan `C*` | `video_model/videopolicy_planner.py`, `sgm/data/llm_planner.py` |
| **Video Generator** `G_θ` | Stable Video Diffusion backbone; predicts short-horizon future latents conditioned on the plan | `sgm/models/diffusion.py`, `sgm/modules/diffusionmodules/video_model.py` |
| **Reward Critic** `F_R` | Scores imagined rollouts; reward drives DDPO steering of `G_θ`. Selectable via `critic_type`: **LIV** (default) or **LLaVA-1.5 + BERTScore** | dispatch in `sgm/models/diffusion.py`; critics in `sgm/modules/critic_model/llava_critic.py`, `llava_cycle_critic.py` |
| **Action Policy** `π_φ` | 1D action diffusion UNet decoding executable actions from `G_θ` features | `pose_net` in the network config; `sgm/models/diffusion.py` |

## 🗂️ Repository Structure

```
.
├── README.md
├── requirements.txt
├── assets/                     # teaser media
├── packages/                   # simulator deps are cloned here (robomimic/robosuite/robocasa)
├── src/sdata/                  # data pipeline (sdata)
├── video_model/                # ← main RoboCasa code (run all commands from here)
│   ├── main.py                 # training entry point
│   ├── eval_script.py          # aggregates eval results → success rates
│   ├── videopolicy_planner.py  # hierarchical LLM planner
│   ├── configs/                # training configs (single-stage / two-stage / ablations)
│   ├── scripts/sampling/       # inference / evaluation
│   │   ├── robocasa_experiment.py        # closed-loop RoboCasa evaluation
│   │   └── configs/svd_xt*.yaml          # inference configs
│   └── sgm/                     # model library
│       ├── models/
│       │   ├── diffusion.py             # main engine: planner cond. + DDPO VLM-critic steering
│       │   ├── diffusion_sbert.py       # critic variant: SBERT-based reward
│       │   └── diffusion_modified_cycle.py  # critic variant: LLaVA CycleReward
│       └── modules/critic_model/        # VLM critic implementations
└── libero/                     # self-contained LIBERO-10 training/eval release
```

> **Critic choice.** `diffusion.py` is the canonical engine used by all shipped configs (DDPO is
> built in) and supports **two reward critics**, selected with `model.params.critic_type`:
> - `liv` — LIV image-language value model, per-frame cosine reward (**default**).
> - `llava` — LLaVA-1.5 + BERTScore VLM critic (`sgm/modules/critic_model/llava_critic.py`).
>
> Override from the CLI, e.g.:
> ```bash
> ... --base=configs/joint_training.yaml ... model.params.critic_type=llava
> ```
> (LLaVA runs a generation per reward call and is much slower than LIV.) The separate
> `diffusion_sbert.py` (SBERT reward) and `diffusion_modified_cycle.py` (LLaVA CycleReward) engines
> swap in other reward signals — point a config's `model.target` at them to use those instead.

## 🛠️ Installation

Create the environment:
```bash
git clone <REPO_URL>           # TODO: anonymized repo URL
cd robotales
conda create -n robotales python=3.10
conda activate robotales
```

Install the simulation environment (cloned into `packages/`):
```bash
cd packages && \
git clone -b robocasa https://github.com/ARISE-Initiative/robomimic && pip install -e robomimic && \
git clone https://github.com/ARISE-Initiative/robosuite && pip install -e robosuite && \
git clone https://github.com/robocasa/robocasa && pip install -e robocasa && \
python robocasa/robocasa/scripts/download_kitchen_assets.py && \
python robocasa/robocasa/scripts/setup_macros.py
cd ..
```

Install the Python packages:
```bash
pip install -r requirements.txt
```

Tested with **Python 3.10, PyTorch 2.1.0, CUDA 11.8**, and `xformers` for memory-efficient attention.

## 🧾 Checkpoints and Datasets

Pretrained checkpoints and the simulation datasets are **not** included in this repo (see
`video_model/CHECKPOINTS.md` and `video_model/datasets/README.md`).

```bash
# TODO: public download URLs to be released
wget <CHECKPOINTS_URL>   # → place extracted checkpoints/ under video_model/
wget <DATASETS_URL>      # → place extracted datasets/ under video_model/
```

Expected layout:
```
video_model/
├── checkpoints/
└── datasets/v0.1/...
```

## 🚀 Training

All training is launched with `main.py` from inside the `video_model/` folder. The general form is:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --base=configs/<CONFIG>.yaml \
    --name=<RUN_NAME> \
    --seed=24 \
    --num_nodes=1 \
    --wandb=1 \
    lightning.trainer.devices="0,1,2,3,4,5,6,7"
```

Useful flags (see `main.py` for the full list): `--base` (config, required), `--name` (run name),
`--seed`, `--num_nodes`, `--wandb` (`0`/`1`), `--resume` / `--resume_from_checkpoint` (continue a run),
`--logdir` (output dir). Any `key=value` after the flags overrides a config field
(e.g. `lightning.trainer.devices`, `data.params.batch_size`, `model.params.ckpt_path`).

### RoboTALES (single-stage joint training) — main method

This jointly optimizes the planner-conditioned video generator and the action policy, with the
VLM critic steering the world model via DDPO:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --base=configs/joint_training.yaml --name=robotales --seed=24 --num_nodes=1 --wandb=1 \
    lightning.trainer.devices="0,1,2,3,4,5,6,7"
```

### Decoupled two-stage training (baseline / ablation)

The paper compares against a decoupled regime where the video generator is trained first and the
action policy is trained afterward on frozen features:

```bash
# Stage 1 — video model (planner + DDPO critic)
PYTHONPATH=. python main.py --base=configs/stage_1_video_model_training.yaml --name=stage1 ...

# Stage 2 — action decoder on the frozen video model
#   (set model.params.ckpt_path to your Stage 1 checkpoint)
PYTHONPATH=. python main.py --base=configs/stage_2_action_decoder_training.yaml --name=stage2 ...
```

### Config reference

| Config | Role |
|---|---|
| `joint_training.yaml` | **RoboTALES single-stage** joint training (main method) |
| `stage_1_video_model_training.yaml` | Decoupled baseline — stage 1 video model (planner + critic) |
| `stage_2_action_decoder_training.yaml` | Decoupled baseline — stage 2 action decoder, video model frozen |
| `stage_1_video_model_train_only_ddpo_no_planner.yaml` | Ablation — DDPO critic **without** the LLM planner (stage 1) |
| `stage_2_action_with_ddpo_no_planner.yaml` | Ablation — DDPO critic **without** the LLM planner (stage 2) |
| `stage_2_action_decoder_training_sft_action_only_robocasa.yaml` | Ablation — supervised action-only policy (no critic) |
| `stage_2_action_decoder_training_sft_action_only_policy12_robocasa.yaml` | Action-only ablation (policy-12 variant) |
| `stage_1_video_model_training_libero.yaml` | Stage 1 on LIBERO (see also the `libero/` release) |

> **Hardware.** Configs target an **8× GPU** node with **80 GB** VRAM each. An overall batch size of
> **32** works well; larger batch sizes tend to help. Adjust `lightning.trainer.devices` and
> `data.params.batch_size` for your setup.

## 🖥️ Inference / Evaluation

Closed-loop evaluation on RoboCasa is run from the `video_model/` folder:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/sampling/robocasa_experiment.py \
    -c scripts/sampling/configs/svd_xt.yaml
```

Before running, edit the inference config (`scripts/sampling/configs/svd_xt.yaml`):

- **`model.params.ckpt_path`** — your trained RoboTALES checkpoint (the shipped value is a placeholder
  local path and **must** be changed).
- **`log_folder`** — results are written to `experiments/<log_folder>/`.
- **`number_of_experiments`** — demos attempted per launch.
- **`max_traj_len`**, **`action_horizon`**, **`decoding_t`** — rollout length, actions executed per
  step, decoded frames.
- **`data.params.tasks`** — the RoboCasa tasks (24 by default), each with `num_experiments`.

**Multi-GPU.** Each launch claims the next pending task/demo via a file lock on
`experiments/<log_folder>/multi_environment_experiment_record.json`, so you can parallelize by running
the same command on different GPUs:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/sampling/robocasa_experiment.py -c scripts/sampling/configs/svd_xt.yaml &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/sampling/robocasa_experiment.py -c scripts/sampling/configs/svd_xt.yaml &
# ... one per available GPU
```

### Inference config variants

| Config | Description |
|---|---|
| `svd_xt_ours_sbert_newckpt.yaml` | RoboTALES with the SBERT-guided planner/critic |
| `svd_xt.yaml` | Base policy evaluation |
| `svd_xt_newckpt_baseline.yaml` | Baseline policy (with planner) |
| `svd_xt_newckpt.yaml` | RoboTALES on the newer checkpoint |
| `svd_xt_sbert_newckpt_a100*.yaml` | A100 SBERT configs (`*_umap` also dumps UMAP features) |

### Computing success rates

```bash
python eval_script.py experiments/<log_folder>/multi_environment_experiment_record.json
```

This prints per-task and overall mean success rates from the experiment record.

## 🤖 LIBERO-10

The `libero/` folder is a **self-contained** release for the LIBERO-10 benchmark with its own
`video_model/`, `sgm/`, configs, and scripts:

```bash
cd libero/video_model

# Training (see configs/)
PYTHONPATH=. python main.py --base=configs/stage_1_video_model_training.yaml --name=libero_stage1 ...
PYTHONPATH=. python main.py --base=configs/stage_2_action_decoder_training.yaml --name=libero_stage2 ...

# Evaluation
PYTHONPATH=. python scripts/sampling/libero_experiment.py   # closed-loop LIBERO eval
PYTHONPATH=. python scripts/sampling/libero_planner.py       # planner-conditioned eval
```

See `libero/video_model/scripts/sampling/run_libero_original_split4.sh` and
`run_libero_planner_split.sh` for example launch commands and task splits.

## 🙏 Acknowledgement

<!-- This repository builds on [Stable Video Diffusion / generative-models](https://github.com/Stability-AI/generative-models)
and the `sdata` data pipeline. We thank the authors for publicly releasing their code. -->

## 📚 Citation

<!-- ```bibtex
@inproceedings{robotales2026,
  title     = {RoboTALES: Learning Reasoning-Guided Robot Policies via Task-Aligned Simulated Futures},
  author    = {Anonymous ECCV 2026 Submission},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
``` -->
