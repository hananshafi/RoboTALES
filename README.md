# RoboTALES: Learning Reasoning-Guided Robot Policies via Task-Aligned Simulated Futures

**Authors:** [Hanan Gani](https://hananshafi.github.io/), [Tejal Kulkarni](https://scholar.google.com/citations?view_op=search_authors&mauthors=Tejal+Kulkarni), [Madhoolika Chodavarapu](https://www.linkedin.com/in/madhoolika-chodavarapu/), [Nicklas Hansen](https://www.nicklashansen.com/), [Manmohan Chandraker](https://cseweb.ucsd.edu/~mkchandraker/)  
**Affiliation:** University of California, San Diego

### ECCV 2026
### [**Project Page**](https://hananshafi.github.io/RoboTALES/) **|** [**Paper**](https://arxiv.org/pdf/2607.06018) **|** [**arXiv**](https://arxiv.org/abs/2607.06018)

<p align="center">
  <a href="assets/iccv-2026-teaser.pdf"><img src="assets/iccv-2026-teaser.png" width="80%"></a>
</p>

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
│   ├── configs/                # joint-training configs
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

### Trained RoboTALES checkpoints (Hugging Face)

The trained checkpoints are hosted at **[hanangani/robotales-ckpts](https://huggingface.co/hanangani/robotales-ckpts)**:

| Benchmark | File |
|---|---|
| RoboCasa | `robocasa_ckpt/robotales-trained-robocasa.ckpt` (~14 GB) |
| LIBERO-10 | `libero10_ckpt/robotales-trained-libero10.ckpt` (~12 GB) |

Download into `video_model/checkpoints/` (run from `video_model/`):
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download hanangani/robotales-ckpts --local-dir checkpoints
# or in Python:
#   from huggingface_hub import snapshot_download
#   snapshot_download("hanangani/robotales-ckpts", local_dir="checkpoints")
```

Then point a config at the checkpoint you want (training or eval), e.g.:
```bash
model.params.ckpt_path=checkpoints/robocasa_ckpt/robotales-trained-robocasa.ckpt
```

### Other required assets (not in this repo)

- **OpenCLIP ViT-H-14 weights** — the conditioner loads `checkpoints/open_clip_pytorch_model.bin`
  (`laion2b_s32b_b79k`). Grab it from
  [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
  and place it under `checkpoints/`.
- **RoboCasa demo datasets** — needed for closed-loop RoboCasa eval (reset states). Download the
  simulation dataset and place the extracted `datasets/` folder under `video_model/`:
  ```bash
  cd video_model
  wget https://videopolicy.cs.columbia.edu/assets/datasets.zip
  unzip datasets.zip
  ```

Expected layout (under `video_model/`):
```
video_model/
├── checkpoints/
│   ├── robocasa_ckpt/robotales-trained-robocasa.ckpt
│   ├── libero10_ckpt/robotales-trained-libero10.ckpt
│   └── open_clip_pytorch_model.bin
└── datasets/v0.1/...
```

## 🧠 LLM Planner

The hierarchical planner calls a **closed-source LLM** — either **Google Gemini** or **OpenAI**.
Bring **your own key** through the environment; **no key is bundled**. The provider is auto-detected
from the model name (`gemini-*` → Gemini, `gpt-*` / `o*` → OpenAI):

```bash
# Gemini (default, e.g. gemini-2.5-pro)
pip install -U google-genai
export GEMINI_API_KEY="<your-gemini-key>"

# — or — OpenAI (e.g. gpt-4o)
pip install -U openai
export OPENAI_API_KEY="<your-openai-key>"
```
Pick the model per config/CLI (e.g. `--model gpt-4o`, or `model="gpt-4o"` in a planner call); pass
`provider="gemini"|"openai"` to override the auto-detection.

**Reproducing without a key.** A **plan cache** ships with the repo
(`video_model/sgm/data/planner_cache.jsonl` for RoboCasa,
`libero/video_model/sgm/data/planner_cache_libero.jsonl` for LIBERO). Plans for the benchmark task
instructions are pre-computed there, so the shipped experiments run **without** an API key — the
planner serves cached plans by instruction lookup. On a cache **miss** with no key set, it raises a
clear error rather than failing silently; set `GEMINI_API_KEY` or `OPENAI_API_KEY` (or add the
instruction to the cache) to plan new tasks.

## 🚀 Training

### RoboCasa Single-Stage Joint Training

Run the main RoboTALES training config from `video_model/`:

```bash
cd video_model
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --base=configs/joint_training_robocasa.yaml --name=robotales --seed=24 --num_nodes=1 --wandb=1 \
    lightning.trainer.devices="0,1,2,3,4,5,6,7"
```

### Joint-Training Configs

| Config | Role |
|---|---|
| `video_model/configs/joint_training_robocasa.yaml` | RoboCasa single-stage joint training |
| `libero/video_model/configs/joint_training_libero.yaml` | LIBERO-10 single-stage joint training |

> **Hardware.** Training requires GPUs with **80 GB** VRAM.

## 🖥️ Inference / Evaluation

Closed-loop evaluation on RoboCasa is run from the `video_model/` folder:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/sampling/robocasa_experiment.py \
    -c scripts/sampling/configs/svd_xt_newckpt.yaml
```

Before running, set only the checkpoint/logging paths in `scripts/sampling/configs/svd_xt_newckpt.yaml`:

- **`model.params.ckpt_path`** — your trained RoboTALES checkpoint.
- **`log_folder`** — results are written to `experiments/<log_folder>/`.

The paper reproducibility settings are already the defaults in this config and should be kept unchanged:

- **`number_of_experiments: 50`** — demos attempted per launch.
- **`max_traj_len: 1000`**, **`action_horizon: 16`**, **`decoding_t: 25`** — rollout length, actions executed per step, and decoded frames.
- **`data.params.tasks`** — the 24 RoboCasa tasks, each with **`num_experiments: 50`**.

**Multi-GPU.** Each launch claims the next pending task/demo via a file lock on
`experiments/<log_folder>/multi_environment_experiment_record.json`, so you can parallelize by running
the same command on different GPUs:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/sampling/robocasa_experiment.py -c scripts/sampling/configs/svd_xt_newckpt.yaml &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/sampling/robocasa_experiment.py -c scripts/sampling/configs/svd_xt_newckpt.yaml &
# ... one per available GPU
```

### Computing success rates

```bash
python eval_script.py experiments/<log_folder>/multi_environment_experiment_record.json
```

This prints per-task and overall mean success rates from the experiment record.

## 🤖 LIBERO-10

The `libero/` folder is a self-contained release for the LIBERO-10 benchmark. Dataset setup paths below are from the repo root; training and evaluation commands run from `libero/video_model/`.

### Dataset

The loaders expect LIBERO-10 HDF5 files under `libero/datasets/libero_10/`. Download the VideoPolicy LIBERO release archive and keep only the dataset directory:

```bash
cd libero
wget https://videopolicy.cs.columbia.edu/assets/libero_release.zip
unzip libero_release.zip
# If the zip extracts into a wrapper directory, move only its datasets/ folder here.
```

Expected layout:

```text
libero/
└── datasets/libero_10/*.hdf5
```

Run training and evaluation from:

```bash
cd libero/video_model
```

Closed-loop evaluation imports the **LIBERO benchmark** and the **robosuite fork**, so clone those
separately and add them to `PYTHONPATH` (training does not need them):

```bash
export LIBERO_PP="<path-to>/LIBERO:<path-to>/robosuite_libero"
```

> Checkpoint paths in the shipped configs are placeholders — set `model.params.ckpt_path` (training)
> and the sampling config's `ckpt_path` (evaluation) to your own checkpoints.

### Training (single-stage joint) — main method

Run the main LIBERO-10 joint-training config from `libero/video_model/`:

```bash
PYTHONPATH=. python main.py --base=configs/joint_training_libero.yaml \
    --name=libero_joint --seed=24 --wandb=1
```

### Evaluation

**RoboTALES (planner-conditioned)**:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$LIBERO_PP:." \
    python scripts/sampling/libero_planner.py \
    --config=scripts/sampling/configs/svd_xt_modified.yaml --use_planner
```

**Multi-GPU splits.** Two helper scripts fan the tasks across GPUs (they auto-generate the per-GPU
split configs). Export `LIBERO_PP` (as above) so the scripts find your `LIBERO` / `robosuite_libero`
clones:

```bash
bash scripts/sampling/run_libero_planner_split.sh [GPU_A GPU_B]
```

Per-task rollouts and success results are written under `experiments/<log_folder>/`.

## 📚 Citation

If you find RoboTALES useful, please cite:

```bibtex
@inproceedings{gani2026robotales,
  title     = {RoboTALES: Learning Reasoning-Guided Robot Policies via Task-Aligned Simulated Futures},
  author    = {Gani, Hanan and Kulkarni, Tejal and Chodavarapu, Madhoolika and Hansen, Nicklas and Chandraker, Manmohan},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
