#!/usr/bin/env python3
"""
Run full H&M ablation over 5 seeds and report mean +/- std with raw prediction saving.

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
- results/hm/full_ablation_hm_5seed/per_seed_metrics.csv
- results/hm/full_ablation_hm_5seed/mean_std_metrics.csv
- results/hm/full_ablation_hm_5seed/mean_std_metrics_pretty.csv
- results/hm/full_ablation_hm_5seed/raw_predictions/ (per-seed predictions)
"""

from __future__ import annotations

import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)

BASELINES_SCRIPT = ROOT_DIR / "scripts" / "active" / "hm_user_churn_baselines.py"
RAD_SCRIPT = ROOT_DIR / "scripts" / "active" / "hm_rad_vs_supervised_rdl.py"
OUT_DIR = ROOT_DIR / "results" / os.environ.get("OUT_SUBDIR", "hm/full_ablation_hm_5seed")
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
    raw_path = RAW_PRED_DIR / f"baselines_seed{seed}.pt"
    env["SAVE_RAW_RESULTS_PATH"] = raw_path.as_posix()

    cmd = [sys.executable, str(BASELINES_SCRIPT)]
    print(f"[seed={seed}] Running H&M baselines script", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Baselines script exited {proc.returncode} (seed={seed})")

    csv_path = ROOT_DIR / "results" / "full_ablation_hm" / "baselines_metrics.csv"
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
            "LATENT_DIM": env.get("LATENT_DIM", "48"),
            "LAMBDA_STRUCT": str(lambda_struct),
            "LAMBDA_RANK": "1.0",
            "ALPHA_FEAT": "1.0",
            "ALPHA_STRUCT": "1.0",
            "TUNE_METRIC": tune_metric,
            "EPOCHS": env.get("EPOCHS", "30"),
            "PATIENCE": env.get("PATIENCE", "5"),
            "WARMUP_EPOCHS": env.get("WARMUP_EPOCHS", "5"),
            "WARMUP_RANK_SCALE": env.get("WARMUP_RANK_SCALE", "0.25"),
            "WARMUP_STRUCT_SCALE": env.get("WARMUP_STRUCT_SCALE", "1.0"),
            "USE_CACHED_LLM_RULES": "1",
            "SAVE_RAW_RESULTS_PATH": raw_path.as_posix(),
        }
    )

    cmd = [sys.executable, str(RAD_SCRIPT)]
    print(f"[seed={seed}] Running H&M RAD variant lambda_struct={lambda_struct}", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"RAD script exited {proc.returncode} (seed={seed}, lambda_struct={lambda_struct})")

    txt = proc.stdout or ""
    m = re.search(
        r"FINAL COMPARISON \(Val Set.*?At-Risk Filtered Labels\).*?"
        r"RAD \(rule-augmented\)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)",
        txt,
        flags=re.S,
    )
    if not m:
        raise RuntimeError("Could not parse RAD at-risk filtered metrics from output")

    return pd.Series(
        {
            "auprc": float(m.group(1)),
            "auroc": float(m.group(2)),
            "pauroc": _load_rad_pauroc(raw_path),
            "p@k": float(m.group(3)),
            "lift": float(m.group(4)),
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
    grp = df.groupby("model")[metrics]
    agg = pd.concat([grp.mean().add_suffix("_mean"), grp.std().add_suffix("_std")], axis=1).reset_index()
    cols = ["model"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    agg = agg[cols]

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
    print(f"FULL H&M ABLATION (5 SEEDS) | seeds={seeds}")
    print("=" * 90)

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
                    "seed": int(seed),
                    "source": "rad_vs_script",
                },
            ]
        )

        combined = pd.concat(
            [
                baselines[["model", "auroc", "auprc", "pauroc", "p@k", "lift", "seed", "source"]],
                rad_rows,
            ],
            ignore_index=True,
        )
        all_rows.extend(combined.to_dict(orient="records"))

    per_seed = pd.DataFrame(all_rows)
    mean_std, pretty = _mean_std_report(per_seed)

    per_seed_out = OUT_DIR / "per_seed_metrics.csv"
    mean_std_out = OUT_DIR / "mean_std_metrics.csv"
    pretty_out = OUT_DIR / "mean_std_metrics_pretty.csv"

    per_seed.to_csv(per_seed_out, index=False)
    mean_std.to_csv(mean_std_out, index=False)
    pretty.to_csv(pretty_out, index=False)

    print(f"[done] Wrote per-seed metrics: {per_seed_out}")
    print(f"[done] Wrote mean/std metrics: {mean_std_out}")
    print(f"[done] Wrote pretty report:    {pretty_out}")
    print(f"[done] Raw predictions saved to: {RAW_PRED_DIR}")
    print("\nMean +/- std summary:")
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
