#!/usr/bin/env python3
"""Render label-sensitivity sweep results as broken-axis grouped bar charts.

The output mirrors the style of results/shared/render.py:
- columns are datasets
- x-axis groups are metrics
- legend entries are ablation variants (target anomaly rates)
- a broken y-axis separates AUROC from lower-scale metrics

By default this script renders two filtered-label figures:
- one for supervised RDL
- one for RAD
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("results/shared/label_sensitivity_actualexp_20260520/per_run_metrics.csv")
DEFAULT_OUTPUT_DIR = Path("results/shared/label_sensitivity_actualexp_20260520")
DATASET_ORDER = ["amazon", "hm"]
METRICS = ["AUROC", "AUPRC", "P@k"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to per_run_metrics.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for rendered figures")
    parser.add_argument("--tau", type=float, default=1.0, help="Tau multiplier to include")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASET_ORDER),
        help="Comma-separated datasets to include",
    )
    return parser.parse_args()


def _rate_label(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _dataset_label(dataset: str) -> str:
    return {"amazon": "Amazon", "hm": "H&M"}.get(dataset, dataset)


def _aggregate(df: pd.DataFrame, model_prefix: str) -> dict[str, dict[str, list[tuple[float, float]]]]:
    grouped = (
        df.groupby(["dataset", "target_anom_rate"], as_index=False)
        .agg(
            auroc_mean=(f"{model_prefix}_auroc", "mean"),
            auroc_std=(f"{model_prefix}_auroc", "std"),
            auprc_mean=(f"{model_prefix}_auprc", "mean"),
            auprc_std=(f"{model_prefix}_auprc", "std"),
            p_at_k_mean=(f"{model_prefix}_p_at_k", "mean"),
            p_at_k_std=(f"{model_prefix}_p_at_k", "std"),
        )
        .sort_values(["dataset", "target_anom_rate"])
    )

    grouped = grouped.fillna(0.0)
    rates = list(grouped["target_anom_rate"].drop_duplicates().sort_values())
    data: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for dataset in grouped["dataset"].drop_duplicates():
        subset = grouped[grouped["dataset"] == dataset].set_index("target_anom_rate")
        data[dataset] = {
            "AUROC": [(float(subset.loc[rate, "auroc_mean"]), float(subset.loc[rate, "auroc_std"])) for rate in rates],
            "AUPRC": [(float(subset.loc[rate, "auprc_mean"]), float(subset.loc[rate, "auprc_std"])) for rate in rates],
            "P@k": [(float(subset.loc[rate, "p_at_k_mean"]), float(subset.loc[rate, "p_at_k_std"])) for rate in rates],
        }
    return data


def _y_ranges(dataset_data: dict[str, list[tuple[float, float]]]) -> tuple[tuple[float, float], tuple[float, float]]:
    auroc_vals = np.array([mean for mean, _ in dataset_data["AUROC"]], dtype=float)
    lower_vals = np.array(
        [mean for metric in ["AUPRC", "P@k"] for mean, _ in dataset_data[metric]],
        dtype=float,
    )
    lower_errs = np.array(
        [std for metric in ["AUPRC", "P@k"] for _, std in dataset_data[metric]],
        dtype=float,
    )
    auroc_errs = np.array([std for _, std in dataset_data["AUROC"]], dtype=float)

    top_min = max(0.0, float(np.min(auroc_vals - auroc_errs)) - 0.02)
    top_max = min(1.02, float(np.max(auroc_vals + auroc_errs)) + 0.02)
    bot_min = max(0.0, float(np.min(lower_vals - lower_errs)) - 0.01)
    bot_max = float(np.max(lower_vals + lower_errs)) + 0.01

    if top_max - top_min < 0.08:
        pad = (0.08 - (top_max - top_min)) / 2
        top_min = max(0.0, top_min - pad)
        top_max = min(1.02, top_max + pad)
    if bot_max - bot_min < 0.05:
        pad = (0.05 - (bot_max - bot_min)) / 2
        bot_min = max(0.0, bot_min - pad)
        bot_max = bot_max + pad

    return (top_min, top_max), (bot_min, bot_max)


def render_figure(
    data: dict[str, dict[str, list[tuple[float, float]]]],
    rates: list[float],
    datasets: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        len(datasets),
        figsize=(4.4 * len(datasets), 4.8),
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.06, "wspace": 0.25},
    )

    if len(datasets) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    bar_width = min(0.14, 0.76 / max(len(rates), 1))
    x = np.arange(len(METRICS))
    offsets = (np.arange(len(rates)) - (len(rates) - 1) / 2.0) * bar_width

    for col, dataset in enumerate(datasets):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        dataset_data = data[dataset]

        for idx, rate in enumerate(rates):
            means = [dataset_data[metric][idx][0] for metric in METRICS]
            stds = [dataset_data[metric][idx][1] for metric in METRICS]
            label = _rate_label(rate)

            ax_top.bar(
                x + offsets[idx],
                means,
                bar_width,
                yerr=stds,
                capsize=2,
                label=label,
                edgecolor="black",
                linewidth=0.4,
            )
            ax_bot.bar(
                x + offsets[idx],
                means,
                bar_width,
                yerr=stds,
                capsize=2,
                edgecolor="black",
                linewidth=0.4,
            )

        top_range, bot_range = _y_ranges(dataset_data)
        ax_top.set_ylim(*top_range)
        ax_bot.set_ylim(*bot_range)

        ax_top.set_title(_dataset_label(dataset), fontweight="bold")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(METRICS)

        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False, bottom=False)
        ax_bot.tick_params(top=False)

        d = 0.012
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=0.8)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        kwargs = dict(transform=ax_bot.transAxes, color="k", clip_on=False, linewidth=0.8)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(rates), 5),
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        title="Target rate",
    )
    fig.suptitle(title, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.18, wspace=0.25, hspace=0.06)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    requested_datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    df = pd.read_csv(args.input)
    df = df[(df["status"] == "ok") & (df["tau_multiplier"] == args.tau)].copy()
    df = df[df["dataset"].isin(requested_datasets)]
    if df.empty:
        raise RuntimeError("No rows found after applying dataset/tau filters")

    datasets = [dataset for dataset in DATASET_ORDER if dataset in set(df["dataset"])]
    rates = list(df["target_anom_rate"].drop_duplicates().sort_values())

    sup_data = _aggregate(df, model_prefix="sup")
    rad_data = _aggregate(df, model_prefix="rad")

    render_figure(
        sup_data,
        rates,
        datasets,
        title="Label Sensitivity Ablation — Supervised RDL",
        output_path=args.output_dir / "label_sensitivity_barplot_supervised_filtered.png",
    )
    render_figure(
        rad_data,
        rates,
        datasets,
        title="Label Sensitivity Ablation — RAD",
        output_path=args.output_dir / "label_sensitivity_barplot_rad_filtered.png",
    )

    print(f"Saved {args.output_dir / 'label_sensitivity_barplot_supervised_filtered.png'}")
    print(f"Saved {args.output_dir / 'label_sensitivity_barplot_rad_filtered.png'}")


if __name__ == "__main__":
    main()