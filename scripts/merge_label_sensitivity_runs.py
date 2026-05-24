#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("usage: merge_label_sensitivity_runs.py OUT_DIR INPUT_CSV...")

    out_dir = Path(sys.argv[1])
    input_csvs = [Path(p) for p in sys.argv[2:]]
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for csv_path in input_csvs:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                frames.append(df)

    if not frames:
        raise SystemExit("no input per_run_metrics.csv files found")

    per_run = pd.concat(frames, ignore_index=True)
    per_run = per_run.drop_duplicates(subset=["dataset", "target_anom_rate", "tau_multiplier", "seed"], keep="last")
    per_run_path = out_dir / "per_run_metrics.csv"
    per_run.to_csv(per_run_path, index=False)

    ok = per_run[per_run["status"] == "ok"].copy()
    if ok.empty:
        raise SystemExit("merged file has no successful runs")

    metric_cols = [
        "orig_rate",
        "filtered_rate",
        "sup_auprc",
        "sup_auroc",
        "sup_p_at_k",
        "sup_lift",
        "rad_auprc",
        "rad_auroc",
        "rad_p_at_k",
        "rad_lift",
        "auprc_gap_rad_minus_rdl",
        "auroc_gap_rad_minus_rdl",
        "p_at_k_gap_rad_minus_rdl",
    ]

    summary = (
        ok.groupby(["dataset", "target_anom_rate", "tau_multiplier"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        c if isinstance(c, str) else (c[0] if c[1] == "" else f"{c[0]}_{c[1]}")
        for c in summary.columns
    ]
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False)

    ratio_card = (
        ok.groupby(["dataset", "target_anom_rate", "tau_multiplier"], as_index=False)
        .agg(
            raw_positive_rate=("orig_rate", "mean"),
            unexpected_positive_rate=("filtered_rate", "mean"),
            raw_users=("orig_users", "mean"),
            raw_anomalies=("orig_anomalies", "mean"),
            unexpected_users=("filtered_users", "mean"),
            unexpected_anomalies=("filtered_anomalies", "mean"),
            n_success=("status", "count"),
        )
        .sort_values(["dataset", "target_anom_rate", "tau_multiplier"])
    )
    ratio_card.to_csv(out_dir / "ratio_card.csv", index=False)

    print(f"[done] wrote {per_run_path}")
    print(f"[done] wrote {out_dir / 'summary_mean_std.csv'}")
    print(f"[done] wrote {out_dir / 'ratio_card.csv'}")


if __name__ == "__main__":
    main()
