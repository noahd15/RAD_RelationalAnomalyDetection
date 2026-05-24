#!/usr/bin/env python3
"""Render advisor-ready label-sensitivity curves with percentage+count labels.

This script builds a dual y-axis figure for one or more datasets:
- Left y-axis: one performance metric for supervised RDL and RAD
- Right y-axis: a second performance metric for supervised RDL and RAD
- X-axis labels: percentage plus count (e.g., "1.5% (n=1590)")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("results/shared/label_sensitivity_small/per_run_metrics.csv")
DEFAULT_OUTPUT = Path("results/shared/label_sensitivity_small/label_sensitivity_dual_axis.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to per_run_metrics.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output figure path")
    parser.add_argument("--tau", type=float, default=1.0, help="Tau multiplier to plot")
    parser.add_argument(
        "--datasets",
        type=str,
        default="amazon,hm",
        help="Comma-separated dataset names to plot (e.g., amazon,hm)",
    )
    parser.add_argument(
        "--left-metric",
        type=str,
        default="auroc",
        choices=["auprc", "auroc", "p_at_k"],
        help="Performance metric for left y-axis",
    )
    parser.add_argument(
        "--right-metric",
        type=str,
        default="auprc",
        choices=["auprc", "auroc", "p_at_k"],
        help="Performance metric for right y-axis",
    )
    return parser.parse_args()


def metric_cols(metric: str) -> tuple[str, str, str]:
    if metric == "auprc":
        return "sup_auprc", "rad_auprc", "AUPRC"
    if metric == "auroc":
        return "sup_auroc", "rad_auroc", "AUROC"
    return "sup_p_at_k", "rad_p_at_k", "P@K"


def dataset_label(dataset: str) -> str:
    labels = {
        "amazon": "Amazon",
        "hm": "H&M",
    }
    return labels.get(dataset, dataset)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    needed = {
        "dataset",
        "target_anom_rate",
        "tau_multiplier",
        "status",
        "filtered_anomalies",
        "sup_auprc",
        "rad_auprc",
        "sup_auroc",
        "rad_auroc",
        "sup_p_at_k",
        "rad_p_at_k",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input csv: {sorted(missing)}")

    left_sup_col, left_rad_col, left_metric_label = metric_cols(args.left_metric)
    right_sup_col, right_rad_col, right_metric_label = metric_cols(args.right_metric)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    work = df[(df["status"] == "ok") & (df["tau_multiplier"] == args.tau) & (df["dataset"].isin(datasets))].copy()
    if work.empty:
        raise RuntimeError(
            "No rows found after filtering. "
            f"Try another tau or dataset list. tau={args.tau}, datasets={datasets}"
        )

    grouped = (
        work.groupby(["dataset", "target_anom_rate"], as_index=False)
        .agg(
            left_sup_mean=(left_sup_col, "mean"),
            left_rad_mean=(left_rad_col, "mean"),
            right_sup_mean=(right_sup_col, "mean"),
            right_rad_mean=(right_rad_col, "mean"),
            n_anom=("filtered_anomalies", "mean"),
            n_runs=("seed", "count"),
        )
        .sort_values(["dataset", "target_anom_rate"])
    )

    n_panels = len(datasets)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5), squeeze=False)
    axes = axes[0]

    for ax, ds in zip(axes, datasets):
        d = grouped[grouped["dataset"] == ds].copy()
        if d.empty:
            ax.set_axis_off()
            ax.set_title(f"{dataset_label(ds)} (no data)")
            continue

        x = list(range(len(d)))
        labels = [f"{r * 100:.1f}% (n={int(round(c))})" for r, c in zip(d["target_anom_rate"], d["n_anom"])]

        l1 = ax.plot(
            x,
            d["left_sup_mean"],
            marker="o",
            linewidth=2.2,
            color="#1f77b4",
            label=f"Supervised RDL ({left_metric_label})",
        )
        l2 = ax.plot(
            x,
            d["left_rad_mean"],
            marker="s",
            linewidth=2.2,
            color="#d62728",
            label=f"RAD ({left_metric_label})",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel("Target anomaly labeling rate (with count)")
        ax.set_ylabel(f"Mean {left_metric_label} (solid)")
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        l3 = ax2.plot(
            x,
            d["right_sup_mean"],
            marker="^",
            linestyle=":",
            color="#1f77b4",
            linewidth=1.8,
            label=f"Supervised RDL ({right_metric_label})",
        )
        l4 = ax2.plot(
            x,
            d["right_rad_mean"],
            marker="D",
            linestyle=":",
            color="#d62728",
            linewidth=1.8,
            label=f"RAD ({right_metric_label})",
        )
        ax2.set_ylabel(f"Mean {right_metric_label} (dotted)")

        lines = l1 + l2 + l3 + l4
        labels_l = [line.get_label() for line in lines]
        ax.legend(lines, labels_l, loc="best")
        ax.set_title(dataset_label(ds))

    fig.suptitle(
        f"Label Sensitivity ({left_metric_label} left, {right_metric_label} right) with Percentages + Counts",
        y=1.02,
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
