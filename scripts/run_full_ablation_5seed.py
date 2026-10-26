#!/usr/bin/env python3
"""
Run full ablation over 5 seeds and report mean +/- std with raw prediction saving.

Models included:
- Isolation Forest
- DRL
- Deep SVDD
- AE
- AE+Rank
- Supervised RDL
- RAD (with BCE)
- RAD (no BCE)

Outputs:
- results/amazon/full_ablation_5seed/per_seed_metrics.csv
- results/amazon/full_ablation_5seed/mean_std_metrics.csv
- results/amazon/full_ablation_5seed/mean_std_metrics_pretty.csv
- results/amazon/full_ablation_5seed/raw_predictions/ (per-seed predictions)
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from ablation_provenance import check_same_eval_set, parse_rad_final, write_run_config


ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)

BASELINES_SCRIPT = ROOT_DIR / "scripts" / "amazon_user_churn_baselines.py"
RAD_SCRIPT = ROOT_DIR / "scripts" / "rad_vs_supervised_rdl.py"
OUT_DIR = ROOT_DIR / "results" / "amazon" / "full_ablation_5seed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_PRED_DIR = OUT_DIR / "raw_predictions"
RAW_PRED_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_AUROC_MAX_FPR = float(os.environ.get("PARTIAL_AUROC_MAX_FPR", "0.1"))


def _partial_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores).astype(float).reshape(-1)
    if np.unique(labels).size <= 1:
        return float("nan")
    if not (0.0 < PARTIAL_AUROC_MAX_FPR < 1.0):
        return float("nan")
    return float(roc_auc_score(labels, scores, max_fpr=PARTIAL_AUROC_MAX_FPR))


def _load_baseline_paurocs(raw_path: Path) -> dict[str, float]:
    raw_obj = torch.load(raw_path, map_location="cpu", weights_only=False)
    alias_map = {
        "IForest": "Isolation Forest",
        "VanillaAE": "AE",
        "AE-Rank": "AE+Rank",
        "DeepSVDD": "Deep SVDD",
        "DRL": "DRL",
    }
    return {
        alias_map.get(model_name, model_name): _partial_auroc(model_scores["labels"], model_scores["scores"])
        for model_name, model_scores in raw_obj.items()
    }


def _load_rad_pauroc(raw_path: Path) -> float:
    raw_obj = torch.load(raw_path, map_location="cpu", weights_only=False)
    return _partial_auroc(raw_obj["test_labels"], raw_obj["test_scores"])


def _run_baselines(seed: int) -> pd.DataFrame:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["SEED"] = str(seed)
    baselines_dir = OUT_DIR / "baselines" / f"seed{seed}"
    env["BASELINES_OUT_DIR"] = baselines_dir.as_posix()
    raw_path = RAW_PRED_DIR / f"baselines_seed{seed}.pt"
    env["SAVE_RAW_RESULTS_PATH"] = raw_path.as_posix()

    cmd = [sys.executable, str(BASELINES_SCRIPT)]
    print(f"[seed={seed}] Running baselines script", flush=True)
    proc = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    csv_path = baselines_dir / "baselines_metrics.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Baselines script did not produce expected file: {csv_path}")

    df = pd.read_csv(csv_path)
    df["seed"] = int(seed)
    df["pauroc"] = df["model"].map(_load_baseline_paurocs(raw_path))
    return df


def _run_rad_variant(seed: int, lambda_struct: float, tune_metric: str = "auprc") -> pd.Series:
    raw_path = RAW_PRED_DIR / f"rad_seed{seed}__lambda_{str(lambda_struct).replace('.', 'p')}.pt"
    env = os.environ.copy()
    env.update(
        {
            "SEED": str(seed),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "1"),
            "LATENT_DIM": "32",
            "LAMBDA_STRUCT": str(lambda_struct),
            "LAMBDA_RANK": "4.0",
            "ALPHA_FEAT": "0.5",
            "ALPHA_STRUCT": "2.0",
            "TUNE_METRIC": tune_metric,
            "EPOCHS": env.get("EPOCHS", "60"),
            "PATIENCE": env.get("PATIENCE", "10"),
            "WARMUP_EPOCHS": env.get("WARMUP_EPOCHS", "15"),
            "WARMUP_RANK_SCALE": env.get("WARMUP_RANK_SCALE", "0.25"),
            "WARMUP_STRUCT_SCALE": env.get("WARMUP_STRUCT_SCALE", "1.0"),
            "SAVE_RAW_RESULTS_PATH": raw_path.as_posix(),
        }
    )

    cmd = [sys.executable, str(RAD_SCRIPT)]
    print(f"[seed={seed}] Running RAD variant lambda_struct={lambda_struct}", flush=True)
    proc = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    txt = proc.stdout or ""
    parsed = parse_rad_final(txt)

    return pd.Series(
        {
            "auprc": parsed["auprc"],
            "auroc": parsed["auroc"],
            "pauroc": _load_rad_pauroc(raw_path),
            "p@k": parsed["p@k"],
            "lift": parsed["lift"],
            "n_eval": parsed["n_eval"],
            "n_pos": parsed["n_pos"],
            "seed": int(seed),
        }
    )


def _alias_models(df: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        "IForest": "Isolation Forest",
        "VanillaAE": "AE",
        "AE-Rank": "AE+Rank",
        "DeepSVDD": "Deep SVDD",
        "DRL": "DRL",
    }
    out = df.copy()
    out["model"] = out["model"].map(lambda x: alias_map.get(x, x))
    return out


def _mean_std_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["auroc", "auprc", "pauroc", "p@k", "lift"]
    agg = (
        df.groupby("model", as_index=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "model",
        "auroc_mean",
        "auroc_std",
        "auprc_mean",
        "auprc_std",
        "pauroc_mean",
        "pauroc_std",
        "p@k_mean",
        "p@k_std",
        "lift_mean",
        "lift_std",
    ]

    pretty = agg.copy()
    for m in metrics:
        pretty[m] = pretty.apply(
            lambda r: f"{r[f'{m}_mean']:.4f} +/- {r[f'{m}_std']:.4f}", axis=1
        )
    pretty = pretty[["model", "auroc", "auprc", "pauroc", "p@k", "lift"]]

    agg = agg.sort_values(["auprc_mean", "auroc_mean"], ascending=[False, False]).reset_index(drop=True)
    pretty = pretty.set_index("model").loc[agg["model"]].reset_index()
    return agg, pretty


def main() -> None:
    seeds = [0, 1, 2, 3, 4]  # 5-seed configuration

    print("=" * 90)
    print(f"FULL ABLATION AMAZON (5 SEEDS) | seeds={seeds}")
    print("=" * 90)

    write_run_config(OUT_DIR, seeds)
    all_rows: list[dict] = []
    for seed in seeds:
        baselines = _run_baselines(seed)
        baselines = _alias_models(baselines)
        baselines["source"] = "baselines_script"

        rad_bce = _run_rad_variant(seed=seed, lambda_struct=0.1)
        rad_no_bce = _run_rad_variant(seed=seed, lambda_struct=0.0)

        rad_rows = pd.DataFrame(
            [
                {
                    "model": "RAD (with BCE)",
                    "auroc": float(rad_bce["auroc"]),
                    "auprc": float(rad_bce["auprc"]),
                    "pauroc": float(rad_bce["pauroc"]),
                    "p@k": float(rad_bce["p@k"]),
                    "lift": float(rad_bce["lift"]),
                    "n_eval": int(rad_bce["n_eval"]),
                    "n_pos": int(rad_bce["n_pos"]),
                    "seed": int(seed),
                    "source": "rad_vs_script",
                },
                {
                    "model": "RAD (no BCE)",
                    "auroc": float(rad_no_bce["auroc"]),
                    "auprc": float(rad_no_bce["auprc"]),
                    "pauroc": float(rad_no_bce["pauroc"]),
                    "p@k": float(rad_no_bce["p@k"]),
                    "lift": float(rad_no_bce["lift"]),
                    "n_eval": int(rad_no_bce["n_eval"]),
                    "n_pos": int(rad_no_bce["n_pos"]),
                    "seed": int(seed),
                    "source": "rad_vs_script",
                },
            ]
        )

        combined = pd.concat(
            [
                baselines[["model", "auroc", "auprc", "pauroc", "p@k", "lift", "n_eval", "n_pos", "seed", "source"]],
                rad_rows,
            ],
            ignore_index=True,
        )
        check_same_eval_set(combined, seed)
        all_rows.extend(combined.to_dict(orient="records"))

    per_seed_df = pd.DataFrame(all_rows)
    per_seed_path = OUT_DIR / "per_seed_metrics.csv"
    per_seed_df.to_csv(per_seed_path, index=False)
    print(f"\nSaved per-seed metrics: {per_seed_path}")

    mean_std_df, pretty_df = _mean_std_report(per_seed_df)
    mean_std_path = OUT_DIR / "mean_std_metrics.csv"
    pretty_path = OUT_DIR / "mean_std_metrics_pretty.csv"
    mean_std_df.to_csv(mean_std_path, index=False)
    pretty_df.to_csv(pretty_path, index=False)
    print(f"Saved aggregates: {mean_std_path}")
    print(f"Saved pretty table: {pretty_path}")
    print(f"Raw predictions saved to: {RAW_PRED_DIR}")

    print("\nFinal Summary (Mean ± Std):")
    print(pretty_df.to_string(index=False))


if __name__ == "__main__":
    main()
