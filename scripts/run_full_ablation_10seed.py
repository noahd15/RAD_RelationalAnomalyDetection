#!/usr/bin/env python3
"""
Run full ablation over multiple seeds and report mean +/- std.

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
- results/amazon/full_ablation_10seed/per_seed_metrics.csv
- results/amazon/full_ablation_10seed/mean_std_metrics.csv
- results/amazon/full_ablation_10seed/mean_std_metrics_pretty.csv
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from path_helpers import RESULTS_JSONL_PATH


ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)

BASELINES_SCRIPT = ROOT_DIR / "scripts" / "amazon_user_churn_baselines.py"
RAD_SCRIPT = ROOT_DIR / "scripts" / "rad_vs_supervised_rdl.py"
OUT_DIR = ROOT_DIR / "results" / "amazon" / "full_ablation_10seed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _read_last_jsonl_record(run_tag: str, method: str) -> dict:
    if not RESULTS_JSONL_PATH.exists():
        raise RuntimeError(f"results.jsonl not found: {RESULTS_JSONL_PATH}")
    record = None
    with open(RESULTS_JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("run_tag") == run_tag and rec.get("method") == method:
                record = rec
    if record is None:
        raise RuntimeError(f"No results.jsonl record for run_tag={run_tag!r} method={method!r}")
    return record


def _run_baselines(seed: int) -> pd.DataFrame:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["SEED"] = str(seed)

    cmd = [sys.executable, str(BASELINES_SCRIPT)]
    print(f"[seed={seed}] Running baselines script", flush=True)
    proc = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    # Seed-specific path: the baselines script also writes an un-seeded path for
    # backward compatibility, but that one is overwritten every seed.
    csv_path = ROOT_DIR / "results" / "amazon" / "full_ablation" / f"baselines_metrics_seed{seed}.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Baselines script did not produce expected file: {csv_path}")

    df = pd.read_csv(csv_path)
    df["seed"] = int(seed)
    return df


def _run_rad_variant(seed: int, lambda_struct: float, tune_metric: str = "auprc") -> pd.Series:
    run_tag = f"amazon10seed_seed{seed}_ls{lambda_struct}_{uuid.uuid4().hex[:8]}"
    env = os.environ.copy()
    env.update(
        {
            "SEED": str(seed),
            "RUN_TAG": run_tag,
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "1"),
            "LATENT_DIM": "32",
            "LAMBDA_STRUCT": str(lambda_struct),
            # Retuned via scripts/optuna_rad32_tune.py against the post-C1-C9-fix rule
            # table and val split (best trial: val AUPRC 0.0289). Old frozen values were
            # tuned against a since-replaced rule set (pre C4/C7 rule-mining fixes).
            "LAMBDA_RANK": "2.0",
            "ALPHA_FEAT": "1.0",
            "ALPHA_STRUCT": "1.0",
            "TUNE_METRIC": tune_metric,
            "EPOCHS": env.get("EPOCHS", "60"),
            "PATIENCE": env.get("PATIENCE", "10"),
            "WARMUP_EPOCHS": env.get("WARMUP_EPOCHS", "15"),
            "WARMUP_RANK_SCALE": env.get("WARMUP_RANK_SCALE", "0.25"),
            "WARMUP_STRUCT_SCALE": env.get("WARMUP_STRUCT_SCALE", "1.0"),
        }
    )

    cmd = [sys.executable, str(RAD_SCRIPT)]
    print(f"[seed={seed}] Running RAD variant lambda_struct={lambda_struct}", flush=True)
    proc = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    rec = _read_last_jsonl_record(run_tag, "RAD (rule-augmented)")
    return pd.Series(
        {
            "auprc": float(rec["auprc"]),
            "auroc": float(rec["auroc"]),
            "p@k": float(rec["p@k"]),
            "lift": float(rec["lift"]),
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
    metrics = ["auroc", "auprc", "p@k", "lift"]
    agg = (
        df.groupby("model", as_index=False)[metrics]
        .agg(["mean", "std"])
    )
    agg.columns = [
        "model",
        "auroc_mean",
        "auroc_std",
        "auprc_mean",
        "auprc_std",
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
    pretty = pretty[["model", "auroc", "auprc", "p@k", "lift"]]

    agg = agg.sort_values(["auprc_mean", "auroc_mean"], ascending=[False, False]).reset_index(drop=True)
    pretty = pretty.set_index("model").loc[agg["model"]].reset_index()
    return agg, pretty


def main() -> None:
    n_seeds = int(os.environ.get("N_SEEDS", "10"))
    start_seed = int(os.environ.get("START_SEED", "0"))
    seeds = list(range(start_seed, start_seed + n_seeds))

    print("=" * 90)
    print(f"FULL ABLATION OVER SEEDS | n_seeds={n_seeds} | seeds={seeds}")
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
                    "p@k": float(rad_bce["p@k"]),
                    "lift": float(rad_bce["lift"]),
                    "seed": int(seed),
                    "source": "rad_vs_script",
                },
                {
                    "model": "RAD (no BCE)",
                    "auroc": float(rad_no_bce["auroc"]),
                    "auprc": float(rad_no_bce["auprc"]),
                    "p@k": float(rad_no_bce["p@k"]),
                    "lift": float(rad_no_bce["lift"]),
                    "seed": int(seed),
                    "source": "rad_vs_script",
                },
            ]
        )

        combined = pd.concat(
            [
                baselines[["model", "auroc", "auprc", "p@k", "lift", "seed", "source"]],
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
    print("\nMean +/- std summary:")
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
