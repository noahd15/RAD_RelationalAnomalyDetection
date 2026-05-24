#!/usr/bin/env python3
"""Render unified user-churn ablation tables as broken-axis grouped bar charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RENDER_DIR = Path("results/shared/user_churn_unified_5seed_rankboost_20260520/rendered")
DEFAULT_OUTPUT = DEFAULT_RENDER_DIR / "user_churn_ablation_barplots_broken_axis.png"
METRICS = ["AUROC", "AUPRC", "P@K"]
LANL_MODELS = [
    "RAD LLM no BCE",
    "RAD LLM + BCE",
    "GAE no rules",
    "GAE RF sampled",
    "GAE RF features",
]
LANL_VALUES = {
    "AUROC": [(0.996, 0.001), (0.983, 0.031), (0.994, 0.004), (0.985, 0.020), (0.993, 0.008)],
    "AUPRC": [(0.659, 0.053), (0.552, 0.025), (0.370, 0.069), (0.530, 0.051), (0.423, 0.123)],
    "P@K": [(0.601, 0.012), (0.572, 0.009), (0.484, 0.097), (0.560, 0.035), (0.509, 0.114)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amazon-csv",
        type=Path,
        default=DEFAULT_RENDER_DIR / "amazon_main_table_5seed_rankboost.csv",
        help="Amazon rendered summary table.",
    )
    parser.add_argument(
        "--hm-csv",
        type=Path,
        default=DEFAULT_RENDER_DIR / "hm_main_table_5seed_rankboost.csv",
        help="H&M rendered summary table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--include-lanl",
        action="store_true",
        help="Include the legacy LANL ablation panel alongside the new Amazon and H&M panels.",
    )
    return parser.parse_args()


def parse_mean_std(cell: str) -> tuple[float, float]:
    mean_str, std_str = [part.strip() for part in cell.split("+/-")]
    return float(mean_str), float(std_str)


def load_table(path: Path) -> tuple[list[str], dict[str, list[tuple[float, float]]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)
    models = df["Model"].tolist()
    values = {
        metric: [parse_mean_std(cell) for cell in df[metric].tolist()]
        for metric in METRICS
    }
    return models, values


def y_ranges(dataset_values: dict[str, list[tuple[float, float]]]) -> tuple[tuple[float, float], tuple[float, float]]:
    auroc_means = np.array([mean for mean, _ in dataset_values["AUROC"]], dtype=float)
    auroc_stds = np.array([std for _, std in dataset_values["AUROC"]], dtype=float)
    low_means = np.array(
        [mean for metric in ["AUPRC", "P@K"] for mean, _ in dataset_values[metric]],
        dtype=float,
    )
    low_stds = np.array(
        [std for metric in ["AUPRC", "P@K"] for _, std in dataset_values[metric]],
        dtype=float,
    )

    top_min = max(0.30, float(np.min(auroc_means - auroc_stds)) - 0.03)
    top_max = min(1.02, float(np.max(auroc_means + auroc_stds)) + 0.03)
    bot_min = max(0.0, float(np.min(low_means - low_stds)) - 0.005)
    bot_max = float(np.max(low_means + low_stds)) + 0.01
    return (top_min, top_max), (bot_min, bot_max)


def lanl_y_ranges() -> tuple[tuple[float, float], tuple[float, float]]:
    return (0.90, 1.02), (0.25, 0.75)


def plot_dataset(
    ax_top,
    ax_bot,
    dataset_name: str,
    models: list[str],
    values: dict[str, list[tuple[float, float]]],
    *,
    fixed_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    bar_width = min(0.10, 0.78 / max(len(models), 1))
    x = np.arange(len(METRICS))
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * bar_width

    for index, model in enumerate(models):
        means = [values[metric][index][0] for metric in METRICS]
        stds = [values[metric][index][1] for metric in METRICS]
        ax_top.bar(
            x + offsets[index],
            means,
            bar_width,
            yerr=stds,
            capsize=2,
            label=model,
            edgecolor="black",
            linewidth=0.4,
        )
        ax_bot.bar(
            x + offsets[index],
            means,
            bar_width,
            yerr=stds,
            capsize=2,
            edgecolor="black",
            linewidth=0.4,
        )

    if fixed_ranges is None:
        top_range, bot_range = y_ranges(values)
    else:
        top_range, bot_range = fixed_ranges
    ax_top.set_ylim(*top_range)
    ax_bot.set_ylim(*bot_range)
    ax_top.set_title(dataset_name, fontweight="bold")
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


def main() -> None:
    args = parse_args()
    amazon_models, amazon_values = load_table(args.amazon_csv)
    hm_models, hm_values = load_table(args.hm_csv)

    if amazon_models != hm_models:
        raise ValueError("Amazon and H&M tables do not share the same model order")

    panel_count = 3 if args.include_lanl else 2
    fig, axes = plt.subplots(
        2,
        panel_count,
        figsize=(4.2 * panel_count + 1.0, 4.8),
        sharex="col",
        gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.06, "wspace": 0.25},
    )

    col = 0
    if args.include_lanl:
        plot_dataset(
            axes[0, col],
            axes[1, col],
            "LANL",
            LANL_MODELS,
            LANL_VALUES,
            fixed_ranges=lanl_y_ranges(),
        )
        col += 1

    plot_dataset(axes[0, col], axes[1, col], "Amazon", amazon_models, amazon_values)
    plot_dataset(axes[0, col + 1], axes[1, col + 1], "H&M", hm_models, hm_values)

    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")

    handles = []
    labels = []
    for top_ax in axes[0]:
        ax_handles, ax_labels = top_ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),
    )
    title = "Unified User-Churn Ablations"
    if args.include_lanl:
        title += " + LANL"
    fig.suptitle(title, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.30, wspace=0.25, hspace=0.06)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    pdf_output = args.output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")
    print(f"Saved {pdf_output}")


if __name__ == "__main__":
    main()