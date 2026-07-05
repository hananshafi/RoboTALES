#!/bin/bash
# ---------------------------------------------------------------
# run_libero_original_split4.sh
#
# Runs libero_experiment.py (no planner) on the *original*
# videopolicy checkpoint across 4 GPUs, splitting the 15 tasks
# from svd_xt_libero_90_original.yaml roughly evenly (4/4/4/3).
#
# Usage:
#   bash scripts/sampling/run_libero_original_split4.sh [GPU_A GPU_B GPU_C GPU_D]
#
# Defaults to GPUs 3 4 5 6.
# ---------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIDEO_MODEL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIBERO_PP="${LIBERO_PP:-/path/to/LIBERO:/path/to/robosuite_libero}"

GPU_A="${1:-3}"
GPU_B="${2:-4}"
GPU_C="${3:-5}"
GPU_D="${4:-6}"

CONFIG_DIR="$SCRIPT_DIR/configs"
BASE_CFG="$CONFIG_DIR/svd_xt_libero_90_original.yaml"

echo "===== Splitting libero_90 (original ckpt) across GPUs $GPU_A $GPU_B $GPU_C $GPU_D ====="
echo "Base config: $BASE_CFG"
echo ""

# ---------- Generate the four split configs from the base config ----------
python3 - "$BASE_CFG" "$CONFIG_DIR" <<'PYEOF'
import sys, copy
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True

base_path = sys.argv[1]
out_dir   = sys.argv[2]

with open(base_path) as f:
    cfg = yaml.load(f)

tasks = list(cfg["data"]["params"]["tasks"].keys())
n = len(tasks)
# Roughly even 4-way split: chunk i gets tasks[i*n//4 : (i+1)*n//4]
chunks = [tasks[i * n // 4:(i + 1) * n // 4] for i in range(4)]

base_log = cfg["log_folder"]

def write_split(task_list, suffix):
    c = copy.deepcopy(cfg)
    c["data"]["params"]["tasks"] = {t: cfg["data"]["params"]["tasks"][t] for t in task_list}
    c["log_folder"] = f"{base_log}_{suffix}"
    out_path = f"{out_dir}/svd_xt_libero_90_original_{suffix}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(c, f)
    print(f"  Wrote {out_path}  ({len(task_list)} tasks)")

write_split(chunks[0], "gpu0")
write_split(chunks[1], "gpu1")
write_split(chunks[2], "gpu2")
write_split(chunks[3], "gpu3")
PYEOF

echo ""

launch () {
    local gpu="$1"
    local cfg_suffix="$2"
    local log_name="libero_original_gpu${gpu}.log"
    echo "===== Launching GPU $gpu  ($cfg_suffix) ====="
    (
      cd "$VIDEO_MODEL_DIR" && \
      CUDA_VISIBLE_DEVICES=$gpu \
      PYTHONPATH="${LIBERO_PP}:$VIDEO_MODEL_DIR" \
      nohup python scripts/sampling/libero_experiment.py \
        --config="scripts/sampling/configs/svd_xt_libero_90_original_${cfg_suffix}.yaml" \
      > "$VIDEO_MODEL_DIR/experiments/${log_name}" 2>&1 &
      echo "  PID=$!  log=experiments/${log_name}"
    )
}

launch "$GPU_A" "gpu0"
launch "$GPU_B" "gpu1"
launch "$GPU_C" "gpu2"
launch "$GPU_D" "gpu3"

echo ""
echo "All four jobs launched. Monitor with:"
echo "  tail -f $VIDEO_MODEL_DIR/experiments/libero_original_gpu${GPU_A}.log"
echo "  tail -f $VIDEO_MODEL_DIR/experiments/libero_original_gpu${GPU_B}.log"
echo "  tail -f $VIDEO_MODEL_DIR/experiments/libero_original_gpu${GPU_C}.log"
echo "  tail -f $VIDEO_MODEL_DIR/experiments/libero_original_gpu${GPU_D}.log"
echo ""
echo "Output dirs (one per GPU):"
echo "  experiments/example_inference_libero_90_zeroshot_original_gpu0/"
echo "  experiments/example_inference_libero_90_zeroshot_original_gpu1/"
echo "  experiments/example_inference_libero_90_zeroshot_original_gpu2/"
echo "  experiments/example_inference_libero_90_zeroshot_original_gpu3/"
