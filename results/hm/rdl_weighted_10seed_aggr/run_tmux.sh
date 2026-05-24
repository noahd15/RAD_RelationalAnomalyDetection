#!/usr/bin/env bash
set -euo pipefail
cd <REPO_ROOT>
source <REPO_ROOT>/rad/bin/activate

echo "[wait] waiting for label_sensitivity sweep to finish on gpu1"
while pgrep -f "run_label_sensitivity_rad_vs_rdl.py" >/dev/null; do
  date '+[wait] %F %T label_sensitivity still running'
  sleep 120
done

mkdir -p <REPO_ROOT>/results/hm/rdl_weighted_10seed_aggr
CUDA_VISIBLE_DEVICES=1 \
DATASET=hm \
WEIGHTED_LOSS=1 \
START_SEED=0 \
N_SEEDS=10 \
ATRISK_TAU_A=0.08 \
ATRISK_TAU_B=6 \
ATRISK_TAU_ALPHA=0.05 \
ATRISK_TAU_MULTIPLIER=0.6 \
TARGET_ANOM_RATE=0.015 \
python <REPO_ROOT>/scripts/active/run_10seed_supervised_rdl_weighted.py 2>&1 | tee <REPO_ROOT>/results/hm/rdl_weighted_10seed_aggr/tmux.log

echo "[done] exit=$?"
exec bash
