#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source rad/bin/activate

mkdir -p results/rad_ablation
SUMMARY_CSV="results/rad_ablation/summary.csv"

echo "run_id,tune_metric,lambda_struct,lambda_rank,alpha_feat,alpha_struct,latent_dim,auprc_filtered,auroc_filtered,auprc_original,auroc_original" > "$SUMMARY_CSV"

run_id=0
for tune_metric in roc_auc auprc; do
  for lambda_struct in 0.2 0.5 1.0; do
    for lambda_rank in 0.5 1.0 2.0; do
      for alpha_pair in "1.0 1.0" "1.0 2.0" "2.0 1.0"; do
        run_id=$((run_id + 1))
        alpha_feat=$(echo "$alpha_pair" | awk '{print $1}')
        alpha_struct=$(echo "$alpha_pair" | awk '{print $2}')

        out_csv="results/rad_ablation/run_${run_id}.csv"
        out_log="results/rad_ablation/run_${run_id}.log"

        echo "[$(date '+%F %T')] START run=${run_id} metric=${tune_metric} ls=${lambda_struct} lr=${lambda_rank} af=${alpha_feat} as=${alpha_struct}" | tee -a "$out_log"

        CUDA_VISIBLE_DEVICES=1 \
        LATENT_DIMS=32 \
        TUNE_METRIC="$tune_metric" \
        LAMBDA_STRUCT="$lambda_struct" \
        LAMBDA_RANK="$lambda_rank" \
        ALPHA_FEAT="$alpha_feat" \
        ALPHA_STRUCT="$alpha_struct" \
        OUT_CSV="$out_csv" \
        python scripts/sweep_rad_latent_dims.py 2>&1 | tee -a "$out_log"

        tail -n 1 "$out_csv" | awk -F',' -v id="$run_id" '{print id "," $0}' >> "$SUMMARY_CSV"
        echo "[$(date '+%F %T')] END   run=${run_id}" | tee -a "$out_log"
      done
    done
  done
done

echo "Ablation complete. Summary: $SUMMARY_CSV"
