#!/usr/bin/env bash
set -u

cd /home/hanan/dev/videopolicy/video_model || exit 1

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate videopolicy

export PYTHONPATH=.:..:../packages/robocasa:../packages/robosuite:../packages/robomimic
export PYTHONBREAKPOINT=0
export PYTHONUNBUFFERED=1

for i in $(seq 1 20); do
  echo "===== Run $i/15 @ $(date) ====="
  python -u scripts/sampling/robocasa_experiment.py --config=scripts/sampling/configs/svd_xt.yaml \
    2>&1 | tee -a "logs/run_${i}.log"
done