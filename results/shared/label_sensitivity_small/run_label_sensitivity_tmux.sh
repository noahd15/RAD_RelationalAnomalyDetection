#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="<REPO_ROOT>"
OUT_DIR="<REPO_ROOT>/results/shared/label_sensitivity_small"
LOG_FILE="<REPO_ROOT>/results/shared/label_sensitivity_small/label_sensitivity_tmux.log"
WAIT_FOR_EXISTING="0"
DATASETS="amazon,hm"
TARGET_ANOM_RATES="0.01,0.015,0.03"
ATRISK_TAU_MULTIPLIERS="0.6,1.0"
SEEDS="0,1"
POS_DOWNSAMPLE_MODE="subtle"
ATRISK_TAU_A="0.08"
ATRISK_TAU_B="6"
ATRISK_TAU_ALPHA="0.05"
CUDA_DEVICE_INDEX="1"

export ROOT_DIR OUT_DIR LOG_FILE WAIT_FOR_EXISTING
export DATASETS TARGET_ANOM_RATES ATRISK_TAU_MULTIPLIERS SEEDS
export POS_DOWNSAMPLE_MODE ATRISK_TAU_A ATRISK_TAU_B ATRISK_TAU_ALPHA CUDA_DEVICE_INDEX

cd "<REPO_ROOT>"
source "<REPO_ROOT>/rad/bin/activate"
mkdir -p "<REPO_ROOT>/results/shared/label_sensitivity_small"

if [[ "0" == "1" ]]; then
  echo "[wait] waiting for full-ablation jobs to finish before starting label sweep"
  while pgrep -f 'run_full_ablation_10seed.py|run_full_ablation_hm_10seed.py' >/dev/null; do
    date '+[wait] %F %T full-ablation still running'
    sleep 120
  done
fi

echo "[start] 2026-05-18 12:32:44 label-sensitivity RAD-vs-RDL sweep"
echo "DATASETS=amazon,hm"
echo "TARGET_ANOM_RATES=0.01,0.015,0.03"
echo "ATRISK_TAU_MULTIPLIERS=0.6,1.0"
echo "SEEDS=0,1"
python "<REPO_ROOT>/scripts/active/run_label_sensitivity_rad_vs_rdl.py" 2>&1 | tee "<REPO_ROOT>/results/shared/label_sensitivity_small/label_sensitivity_tmux.log"
echo "[done] 2026-05-18 12:32:44"
