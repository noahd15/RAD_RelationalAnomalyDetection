#!/usr/bin/env python3
"""Render unified 5-seed user-churn tables and P@K plots.

Inputs are the output directories produced by run_user_churn_unified_5seed.py.
Outputs are written under <out_root>/rendered/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "results" / "shared" / "user_churn_unified_5seed_rankboost_20260520"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(candidates[0])


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def mean_std_keys(metric: str, columns: pd.Index) -> tuple[str, str]:
    if f"{metric}_mean" in columns and f"{metric}_std" in columns:
        return f"{metric}_mean", f"{metric}_std"
    if metric == "p@k" and "p_at_k_mean" in columns and "p_at_k_std" in columns:
        return "p_at_k_mean", "p_at_k_std"
    raise KeyError(metric)


def load_amazon_table() -> pd.DataFrame:
    mean_std = read_csv(first_existing_path(
        ROOT / "results" / "amazon" / "full_ablation_5seed_table_rankboost" / "mean_std_metrics.csv",
        ROOT / "results" / "amazon" / "full_ablation_10seed_table" / "mean_std_metrics.csv",
    ))
    weighted = read_csv(first_existing_path(
        ROOT / "results" / "amazon" / "rdl_weighted_5seed_rankboost" / "summary_mean_std.csv",
        ROOT / "results" / "amazon" / "rdl_weighted_10seed" / "summary_mean_std.csv",
    ))
    unweighted_path = first_existing_path(
        ROOT / "results" / "amazon" / "rdl_unweighted_5seed_rankboost" / "summary_mean_std.csv",
        ROOT / "results" / "amazon" / "rdl_unweighted_10seed" / "summary_mean_std.csv",
        ROOT / "results" / "amazon" / "full_ablation_10seed_table" / "mean_std_metrics.csv",
    )
    unweighted = read_csv(unweighted_path)

    rename_map = {
        "RAD (rules, no BCE)": "RAD (rules, no BCE)",
        "RAD (rules, BCE)": "RAD (rules, BCE)",
        "AE+Rank": "AE+Rank",
        "DRL": "DRL",
        "AE": "AE",
        "Deep SVDD": "Deep SVDD",
        "Isolation Forest": "Isolation Forest",
    }
    rows = []
    for model_name, out_name in rename_map.items():
        row = mean_std.loc[mean_std["model"] == model_name].iloc[0]
        rows.append(
            {
                "Model": out_name,
                "AUROC": fmt(row["auroc_mean"], row["auroc_std"]),
                "AUPRC": fmt(row["auprc_mean"], row["auprc_std"]),
                "P@K": fmt(row["p_at_k_mean"], row["p_at_k_std"]),
            }
        )

    w = weighted.iloc[0]
    rows.insert(2, {"Model": "RDL weighted", "AUROC": fmt(w["auroc_mean"], w["auroc_std"]), "AUPRC": fmt(w["auprc_mean"], w["auprc_std"]), "P@K": fmt(w["p@k_mean"], w["p@k_std"])})
    if "model" in unweighted.columns:
        u = unweighted.loc[unweighted["model"] == "Supervised RDL"].iloc[0]
        rows.insert(3, {"Model": "RDL unweighted", "AUROC": fmt(u["auroc_mean"], u["auroc_std"]), "AUPRC": fmt(u["auprc_mean"], u["auprc_std"]), "P@K": fmt(u["p_at_k_mean"], u["p_at_k_std"])})
    else:
        u = unweighted.iloc[0]
        rows.insert(3, {"Model": "RDL unweighted", "AUROC": fmt(u["auroc_mean"], u["auroc_std"]), "AUPRC": fmt(u["auprc_mean"], u["auprc_std"]), "P@K": fmt(u["p@k_mean"], u["p@k_std"])})
    return pd.DataFrame(rows)


def load_hm_table() -> pd.DataFrame:
    mean_std = read_csv(first_existing_path(
        ROOT / "results" / "hm" / "full_ablation_hm_5seed_rankboost" / "mean_std_metrics.csv",
        ROOT / "results" / "hm" / "full_ablation_hm_10seed" / "mean_std_metrics.csv",
    ))
    weighted = read_csv(first_existing_path(
        ROOT / "results" / "hm" / "rdl_weighted_5seed_rankboost" / "summary_mean_std.csv",
        ROOT / "results" / "hm" / "rdl_weighted_10seed" / "summary_mean_std.csv",
    ))
    unweighted_path = first_existing_path(
        ROOT / "results" / "hm" / "rdl_unweighted_5seed_rankboost" / "summary_mean_std.csv",
        ROOT / "results" / "hm" / "rdl_unweighted_10seed" / "summary_mean_std.csv",
        ROOT / "results" / "hm" / "full_ablation_hm_10seed" / "mean_std_metrics.csv",
    )
    unweighted = read_csv(unweighted_path)

    rename_map = {
        "RAD (no BCE)": "RAD (rules, no BCE)",
        "RAD (with BCE)": "RAD (rules, BCE)",
        "AE+Rank": "AE+Rank",
        "DRL": "DRL",
        "AE": "AE",
        "Deep SVDD": "Deep SVDD",
        "Isolation Forest": "Isolation Forest",
    }
    rows = []
    for model_name, out_name in rename_map.items():
        row = mean_std.loc[mean_std["model"] == model_name].iloc[0]
        rows.append(
            {
                "Model": out_name,
                "AUROC": fmt(row["auroc_mean"], row["auroc_std"]),
                "AUPRC": fmt(row["auprc_mean"], row["auprc_std"]),
                "P@K": fmt(row["p@k_mean"], row["p@k_std"]),
            }
        )

    w = weighted.iloc[0]
    rows.insert(2, {"Model": "RDL weighted", "AUROC": fmt(w["auroc_mean"], w["auroc_std"]), "AUPRC": fmt(w["auprc_mean"], w["auprc_std"]), "P@K": fmt(w["p@k_mean"], w["p@k_std"])})
    if "model" in unweighted.columns:
        u = unweighted.loc[unweighted["model"] == "Supervised RDL"].iloc[0]
        rows.insert(3, {"Model": "RDL unweighted", "AUROC": fmt(u["auroc_mean"], u["auroc_std"]), "AUPRC": fmt(u["auprc_mean"], u["auprc_std"]), "P@K": fmt(u["p@k_mean"], u["p@k_std"])})
    else:
        u = unweighted.iloc[0]
        rows.insert(3, {"Model": "RDL unweighted", "AUROC": fmt(u["auroc_mean"], u["auroc_std"]), "AUPRC": fmt(u["auprc_mean"], u["auprc_std"]), "P@K": fmt(u["p@k_mean"], u["p@k_std"])})
    return pd.DataFrame(rows)


def save_table_png(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig_h = 0.8 + 0.42 * (len(df) + 1)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.5)
    for j in range(len(df.columns)):
        cell = tbl[(0, j)]
        cell.set_facecolor("#1f2937")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    for i in range(1, len(df) + 1):
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        for j in range(len(df.columns)):
            tbl[(i, j)].set_facecolor(bg)
    ax.set_title(title, fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_method_metric(dataset: str, method: str, metric: str) -> tuple[float, float]:
    if dataset == "amazon":
        if method == "RAD (rules, no BCE)":
            df = read_csv(first_existing_path(
                ROOT / "results" / "amazon" / "full_ablation_5seed_table_rankboost" / "mean_std_metrics.csv",
                ROOT / "results" / "amazon" / "full_ablation_10seed_table" / "mean_std_metrics.csv",
            ))
            row = df.loc[df["model"] == "RAD (rules, no BCE)"].iloc[0]
            mean_key, std_key = mean_std_keys(metric, df.columns)
            return float(row[mean_key]), float(row[std_key])
        if method == "RAD (rules, BCE)":
            df = read_csv(first_existing_path(
                ROOT / "results" / "amazon" / "full_ablation_5seed_table_rankboost" / "mean_std_metrics.csv",
                ROOT / "results" / "amazon" / "full_ablation_10seed_table" / "mean_std_metrics.csv",
            ))
            row = df.loc[df["model"] == "RAD (rules, BCE)"].iloc[0]
            mean_key, std_key = mean_std_keys(metric, df.columns)
            return float(row[mean_key]), float(row[std_key])
        if method == "RAD (no rules)":
            df = read_csv(first_existing_path(
                ROOT / "results" / "amazon" / "gae_norules_5seed_rankboost" / "summary_mean_std.csv",
                ROOT / "results" / "amazon" / "gae_norules_10seed" / "summary_mean_std.csv",
            ))
            row = df.loc[df["metric"] == metric].iloc[0]
            return float(row["mean"]), float(row["std"])
        if method == "RDL weighted":
            df = read_csv(first_existing_path(
                ROOT / "results" / "amazon" / "rdl_weighted_5seed_rankboost" / "summary_mean_std.csv",
                ROOT / "results" / "amazon" / "rdl_weighted_10seed" / "summary_mean_std.csv",
            ))
            row = df.iloc[0]
            mean_key, std_key = mean_std_keys(metric, df.columns)
            return float(row[mean_key]), float(row[std_key])
        if method == "RDL unweighted":
            path = first_existing_path(
                ROOT / "results" / "amazon" / "rdl_unweighted_5seed_rankboost" / "summary_mean_std.csv",
                ROOT / "results" / "amazon" / "rdl_unweighted_10seed" / "summary_mean_std.csv",
                ROOT / "results" / "amazon" / "full_ablation_10seed_table" / "mean_std_metrics.csv",
            )
            df = read_csv(path)
            if "model" in df.columns:
                row = df.loc[df["model"] == "Supervised RDL"].iloc[0]
                mean_key, std_key = mean_std_keys(metric, df.columns)
                return float(row[mean_key]), float(row[std_key])
            row = df.iloc[0]
            mean_key, std_key = mean_std_keys(metric, df.columns)
            return float(row[mean_key]), float(row[std_key])
    if dataset == "hm":
        if method == "RAD (rules, no BCE)":
            df = read_csv(first_existing_path(
                ROOT / "results" / "hm" / "full_ablation_hm_5seed_rankboost" / "mean_std_metrics.csv",
                ROOT / "results" / "hm" / "full_ablation_hm_10seed" / "mean_std_metrics.csv",
            ))
            row = df.loc[df["model"] == "RAD (no BCE)"].iloc[0]
            mean_key, std_key = mean_std_keys(metric, df.columns)
            return float(row[mean_key]), float(row[std_key])
        if method == "RAD (rules, BCE)":
            df = read_csv(first_existing_path(
                ROOT / "results" / "hm" / "full_ablation_hm_5seed_rankboost" / "mean_std_metrics.csv",
                ROOT / "results" / "hm" / "full_ablation_hm_10seed" / "mean_std_metrics.csv",
            ))
            row = df.loc[df["model"] == "RAD (with BCE)"].iloc[0]
            mean_key, std_key = mean_std_keys(metric, df.columns)
            return float(row[mean_key]), float(row[std_key])
        if method == "RAD (no rules)":
            df = read_csv(first_existing_path(
                ROOT / "results" / "hm" / "gae_norules_5seed_rankboost" / "summary_mean_std.csv",
                ROOT / "results" / "hm" / "gae_norules_5seed" / "summary_mean_std.csv",
                ROOT / "results" / "hm" / "gae_norules_10seed" / "summary_mean_std.csv",
            ))
            row = df.loc[df["metric"] == metric].iloc[0]
            return float(row["mean"]), float(row["std"])
        if method == "RDL weighted":
            df = read_csv(first_existing_path(
                ROOT / "results" / "hm" / "rdl_weighted_5seed_rankboost" / "summary_mean_std.csv",
                ROOT / "results" / "hm" / "rdl_weighted_10seed" / "summary_mean_std.csv",
            ))
            row = df.iloc[0]
            mean_key, std_key = mean_std_keys(metric, df.columns)
            return float(row[mean_key]), float(row[std_key])
        if method == "RDL unweighted":
            path = first_existing_path(
                ROOT / "results" / "hm" / "rdl_unweighted_5seed_rankboost" / "summary_mean_std.csv",
                ROOT / "results" / "hm" / "rdl_unweighted_10seed" / "summary_mean_std.csv",
                ROOT / "results" / "hm" / "full_ablation_hm_10seed" / "mean_std_metrics.csv",
            )
            df = read_csv(path)
            if "model" in df.columns:
                row = df.loc[df["model"] == "Supervised RDL"].iloc[0]
                mean_key, std_key = mean_std_keys(metric, df.columns)
                return float(row[mean_key]), float(row[std_key])
            row = df.iloc[0]
            mean_key, std_key = mean_std_keys(metric, df.columns)
            return float(row[mean_key]), float(row[std_key])
    raise KeyError((dataset, method, metric))


def render_p_at_k_plot(out_path: Path) -> None:
    methods = ["RAD (rules, no BCE)", "RAD (rules, BCE)", "RAD (no rules)", "RDL weighted", "RDL unweighted"]
    datasets = ["amazon", "hm"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
    width = 0.16
    for ax, dataset in zip(axes, datasets):
        xs = list(range(len(methods)))
        means = []
        stds = []
        for method in methods:
            mean, std = load_method_metric(dataset, method, "p@k")
            means.append(mean)
            stds.append(std)
        ax.bar(xs, means, yerr=stds, color=colors, capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylabel("Precision@K")
        ax.set_title(dataset.upper())
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Precision@K (K = number of positives in filtered validation set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = args.out_root
    rendered = out_root / "rendered"
    rendered.mkdir(parents=True, exist_ok=True)

    amazon_df = load_amazon_table()
    hm_df = load_hm_table()

    amazon_csv = rendered / "amazon_main_table_5seed_rankboost.csv"
    hm_csv = rendered / "hm_main_table_5seed_rankboost.csv"
    amazon_png = rendered / "amazon_main_table_5seed_rankboost.png"
    hm_png = rendered / "hm_main_table_5seed_rankboost.png"
    pak_png = rendered / "precision_at_k_barplot_5seed_rankboost.png"

    amazon_df.to_csv(amazon_csv, index=False)
    hm_df.to_csv(hm_csv, index=False)
    save_table_png(amazon_df, amazon_png, "Amazon Main Table (5-seed, rank-boosted RAD)")
    save_table_png(hm_df, hm_png, "H&M Main Table (5-seed, rank-boosted RAD)")
    render_p_at_k_plot(pak_png)

    print(f"Saved {amazon_csv}")
    print(f"Saved {hm_csv}")
    print(f"Saved {amazon_png}")
    print(f"Saved {hm_png}")
    print(f"Saved {pak_png}")


if __name__ == "__main__":
    main()
