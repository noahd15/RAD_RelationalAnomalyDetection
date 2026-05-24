#!/usr/bin/env bash
set -euo pipefail
cd <REPO_ROOT>
source <REPO_ROOT>/rad/bin/activate

echo "[wait] waiting for amazon gae_norules to finish"
while pgrep -f "run_10seed_gae_norules_amazon.py" >/dev/null; do
  date '+[wait] %F %T amazon gae_norules still running'
  sleep 90
done

mkdir -p <REPO_ROOT>/results/amazon/rdl_weighted_10seed_aggr
CUDA_VISIBLE_DEVICES=0 \
DATASET=amazon \
WEIGHTED_LOSS=1 \
START_SEED=0 \
N_SEEDS=10 \
ATRISK_TAU_A=0.08 \
ATRISK_TAU_B=6 \
ATRISK_TAU_ALPHA=0.05 \
ATRISK_TAU_MULTIPLIER=0.6 \
TARGET_ANOM_RATE=0.015 \
python <REPO_ROOT>/scripts/active/run_10seed_supervised_rdl_weighted.py 2>&1 | tee <REPO_ROOT>/results/amazon/rdl_weighted_10seed_aggr/tmux.log

echo "[done] exit=$?"
exec bash
