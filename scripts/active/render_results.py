#!/usr/bin/env python3
"""
render_results.py
-----------------
Render run_all_experiments results (mean ± std) as a PNG table.

Outputs:
  - Pretty-printed table to stdout
  - results/all_results_table.png
"""

import argparse
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


os.chdir(ROOT_DIR)


RESULTS_DIR = Path("./results")
INPUT_FILE  = RESULTS_DIR / "all_results_mean_std.csv"

METRIC_PAIRS = [
    ("auroc_mean",              "auroc_std",              "AUROC"),
    ("auprc_mean",              "auprc_std",              "AUPRC"),
    ("p@127_mean",              "p@127_std",              "P@127"),
    ("precision_at_k_mean",     "precision_at_k_std",     "P@k"),
    ("auprc_lift_vs_prev_mean", "auprc_lift_vs_prev_std", "AUPRC Lift"),
]

MODEL_LABELS = {
    "vanilla_ae_rank":                       "AE-Rank",
    "vanilla_ae":                            "Vanilla AE",
    "ae_rules":                              "AE + Rules",
    "ae_rank_rules":                         "AE-Rank + Rules",
    "iforest":                               "Isolation Forest",
    "deep_svdd":                             "Deep SVDD",
    "drl":                                   "DRL",
    "gae_features_only":                     "GAE (features only)",
    "gae_features_only_no_struct_bce":       "GAE (features only, no-struct BCE)",
    "gae_rules_only":                        "GAE (rules only)",
    "gae_features_plus_rules":               "GAE (feat + rules)",
    "gae_features_plus_rules_no_struct_bce": "GAE (feat+rules, no-struct BCE)",
    "gae_features_only_plus_rules_post_lr":  "GAE (feat-only + rules post-LR)",
    "rules_only_baseline":                   "Rules-Only Baseline",
}


def fmt(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def build_display_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        model_id = row["model"]
        entry = {"Model": MODEL_LABELS.get(model_id, model_id)}
        for mean_col, std_col, label in METRIC_PAIRS:
            if mean_col in row and std_col in row:
                entry[label] = fmt(row[mean_col], row[std_col])
            else:
                entry[label] = "—"
        rows.append(entry)
    return pd.DataFrame(rows)


def save_png(table: pd.DataFrame, out_path: Path, title: str = "Experiment Results  (mean ± std)") -> None:
    n_rows, n_cols = table.shape
    col_widths = [3.2] + [2.2] * (n_cols - 1)
    fig_width  = sum(col_widths) + 0.4
    row_height = 0.38
    table_height = row_height * (n_rows + 1)
    title_height = 0.45
    fig_height   = table_height + title_height

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(top=1 - title_height / fig_height, bottom=0, left=0.01, right=0.99)
    ax.axis("off")

    tbl = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        bbox=[0, 0, 1, 1],   # fill the entire axes area
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Set relative column widths
    total_width = sum(col_widths)
    for col_idx, width in enumerate(col_widths):
        for row_idx in range(n_rows + 1):
            tbl[row_idx, col_idx].set_width(width / total_width)

    # Style header row
    header_color = "#4472C4"
    for col_idx in range(n_cols):
        cell = tbl[0, col_idx]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    # Style data rows with alternating colours
    for row_idx in range(1, n_rows + 1):
        bg = "#dce6f1" if row_idx % 2 == 0 else "white"
        for col_idx in range(n_cols):
            tbl[row_idx, col_idx].set_facecolor(bg)
            tbl[row_idx, col_idx].set_edgecolor("#cccccc")

    # Left-align model name column
    for row_idx in range(1, n_rows + 1):
        tbl[row_idx, 0].set_text_props(ha="left")
        tbl[row_idx, 0]._loc = "left"

    fig.suptitle(title, fontsize=12, fontweight="bold",
                 y=1 - (title_height / 2) / fig_height)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


GAE_MODELS = {
    "gae_features_only",
    "gae_features_only_no_struct_bce",
    "gae_rules_only",
    "gae_features_plus_rules",
    "gae_features_plus_rules_no_struct_bce",
    "gae_features_only_plus_rules_post_lr",
    "rules_only_baseline",
}

TABULAR_MODELS = {
    "vanilla_ae_rank", "vanilla_ae", "ae_rules", "ae_rank_rules",
    "iforest", "deep_svdd", "drl",
}


def merge_graph_split(gae_path: Path, tabular_graph_split_path: Path) -> pd.DataFrame:
    """Replace tabular rows in gae_path with re-run rows from tabular_graph_split_path."""
    gae_df = pd.read_csv(gae_path)
    tab_df = pd.read_csv(tabular_graph_split_path)

    # Keep only GAE models from the original full run
    gae_only = gae_df[gae_df["model"].isin(GAE_MODELS)]
    # Keep only tabular models from the new graph-split run
    tab_only = tab_df[tab_df["model"].isin(TABULAR_MODELS)]

    merged = pd.concat([gae_only, tab_only], ignore_index=True)
    if "auroc_mean" in merged.columns:
        merged = merged.sort_values("auroc_mean", ascending=False).reset_index(drop=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Render experiment results table as PNG")
    parser.add_argument(
        "--input", default=str(INPUT_FILE),
        help="Path to mean/std CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--merge-graph-split", action="store_true",
        help="Merge graph-split tabular results (all_results_mean_std_graph_split.csv) "
             "with GAE results from the main run for an apples-to-apples table",
    )
    parser.add_argument(
        "--sort-by", default="auroc_mean",
        help="Column to sort by descending (default: %(default)s)",
    )
    parser.add_argument(
        "--output", default=str(RESULTS_DIR / "all_results_table.png"),
        help="Output PNG path (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.merge_graph_split:
        gae_path = RESULTS_DIR / "all_results_mean_std.csv"
        tab_path = RESULTS_DIR / "all_results_mean_std_graph_split.csv"
        for p in (gae_path, tab_path):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")
        raw = merge_graph_split(gae_path, tab_path)
        if not args.output or args.output == str(RESULTS_DIR / "all_results_table.png"):
            args.output = str(RESULTS_DIR / "all_results_table_graph_split.png")
        title = "Experiment Results  (mean ± std, graph-aligned split)"
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Results file not found: {input_path}")
        raw = pd.read_csv(input_path)
        title = "Experiment Results  (mean ± std)"

    if args.sort_by in raw.columns:
        raw = raw.sort_values(args.sort_by, ascending=False).reset_index(drop=True)
    else:
        print(f"[warn] sort column '{args.sort_by}' not found — skipping sort")

    table = build_display_table(raw)

    # ── stdout ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(table.to_string(index=False))
    print("=" * 80 + "\n")

    # ── PNG ───────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = Path(args.output)
    save_png(table, out_path, title=title)
    print(f"Saved PNG table → {out_path}")


if __name__ == "__main__":
    main()
