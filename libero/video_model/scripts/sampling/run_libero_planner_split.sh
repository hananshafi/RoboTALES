#!/bin/bash
# ---------------------------------------------------------------
# run_libero_planner_split.sh
#
# Launches the LIBERO planner experiment across 2 GPUs:
#   GPU 0  →  tasks 1-5   (config: svd_xt_modified_gpu0.yaml)
#   GPU 1  →  tasks 6-10  (config: svd_xt_modified_gpu1.yaml)
#
# Usage:
#   bash scripts/sampling/run_libero_planner_split.sh [GPU_A] [GPU_B]
#
# Defaults to GPUs 4 and 5 if not specified.
# ---------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIDEO_MODEL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU_A="${1:-4}"
GPU_B="${2:-5}"

CONFIG_DIR="$SCRIPT_DIR/configs"

echo "===== Splitting LIBERO tasks across GPU $GPU_A and GPU $GPU_B ====="
echo "Config dir: $CONFIG_DIR"
echo ""

# ---------- Generate the two split configs from the base config ----------
python3 - "$CONFIG_DIR/svd_xt_modified.yaml" "$CONFIG_DIR" <<'PYEOF'
import sys, copy
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True

base_path = sys.argv[1]
out_dir    = sys.argv[2]

with open(base_path) as f:
    cfg = yaml.load(f)

tasks = list(cfg["data"]["params"]["tasks"].keys())
mid = len(tasks) // 2
tasks_gpu0 = tasks[:mid]
tasks_gpu1 = tasks[mid:]

def write_split(task_list, suffix, log_suffix):
    c = copy.deepcopy(cfg)
    c["data"]["params"]["tasks"] = {t: cfg["data"]["params"]["tasks"][t] for t in task_list}
    c["log_folder"] = cfg["log_folder"] + log_suffix
    out_path = f"{out_dir}/svd_xt_modified_{suffix}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(c, f)
    print(f"  Wrote {out_path}  ({len(task_list)} tasks)")

write_split(tasks_gpu0, "gpu0", "_gpu0")
write_split(tasks_gpu1, "gpu1", "_gpu1")
PYEOF

echo ""
echo "===== Launching GPU $GPU_A (tasks 1-5) ====="
CUDA_VISIBLE_DEVICES=$GPU_A \
  PYTHONPATH=/home/hanan/dev/LIBERO \
  nohup python "$SCRIPT_DIR/libero_planner.py" \
    --config="$CONFIG_DIR/svd_xt_modified_gpu0.yaml" \
    --use_planner \
  > "$VIDEO_MODEL_DIR/experiments/libero_planner_gpu${GPU_A}.log" 2>&1 &
PID_A=$!
echo "  PID=$PID_A  log=experiments/libero_planner_gpu${GPU_A}.log"

echo "===== Launching GPU $GPU_B (tasks 6-10) ====="
CUDA_VISIBLE_DEVICES=$GPU_B \
  PYTHONPATH=/home/hanan/dev/LIBERO \
  nohup python "$SCRIPT_DIR/libero_planner.py" \
    --config="$CONFIG_DIR/svd_xt_modified_gpu1.yaml" \
    --use_planner \
  > "$VIDEO_MODEL_DIR/experiments/libero_planner_gpu${GPU_B}.log" 2>&1 &
PID_B=$!
echo "  PID=$PID_B  log=experiments/libero_planner_gpu${GPU_B}.log"

echo ""
echo "Both jobs launched. Monitor with:"
echo "  tail -f experiments/libero_planner_gpu${GPU_A}.log"
echo "  tail -f experiments/libero_planner_gpu${GPU_B}.log"
echo ""
echo "To stop:  kill $PID_A $PID_B"
