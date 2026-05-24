#!/usr/bin/env python3
"""
10-seed run: RAD (no rules, no supervised GNN) on H&M user-churn.

Calls hm_rad_vs_supervised_rdl.py as a subprocess with:
  RAD_DISABLE_RULES=1  SUP_EPOCHS=0  USE_CACHED_LLM_RULES=1

Outputs:
  results/hm/gae_norules_10seed/per_seed.csv
  results/hm/gae_norules_10seed/summary_mean_std.csv
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
os.chdir(ROOT_DIR)

RAD_SCRIPT = ROOT_DIR / "scripts" / "active" / "hm_rad_vs_supervised_rdl.py"
OUT_DIR = ROOT_DIR / "results" / os.environ.get("OUT_SUBDIR", "hm/gae_norules_10seed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_and_stream(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    chunks: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        chunks.append(line)

    return proc.wait(), "".join(chunks)


def _run_seed(seed: int) -> pd.Series:
    env = os.environ.copy()
    env.update(
        {
            "SEED": str(seed),
            "RAD_DISABLE_RULES": "1",
            "SUP_EPOCHS": "0",
            "USE_CACHED_LLM_RULES": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "1"),
            "LATENT_DIM": env.get("LATENT_DIM", "48"),
            "EPOCHS": env.get("EPOCHS", "30"),
            "PATIENCE": env.get("PATIENCE", "5"),
            "WARMUP_EPOCHS": env.get("WARMUP_EPOCHS", "5"),
            "WARMUP_RANK_SCALE": env.get("WARMUP_RANK_SCALE", "0.25"),
            "WARMUP_STRUCT_SCALE": env.get("WARMUP_STRUCT_SCALE", "1.0"),
            "LAMBDA_STRUCT": env.get("LAMBDA_STRUCT", "0.5"),
            "LAMBDA_RANK": env.get("LAMBDA_RANK", "1.0"),
            "ALPHA_FEAT": env.get("ALPHA_FEAT", "1.0"),
            "ALPHA_STRUCT": env.get("ALPHA_STRUCT", "1.0"),
            # Safer defaults for shared-GPU runs; can still be overridden by caller env.
            "LOADER_BATCH_SIZE": env.get("LOADER_BATCH_SIZE", "256"),
            "ORIG_VAL_BATCH_SIZE": env.get("ORIG_VAL_BATCH_SIZE", env.get("LOADER_BATCH_SIZE", "256")),
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        }
    )

    cmd = [sys.executable, "-u", str(RAD_SCRIPT)]
    print(f"\n[seed={seed}] Starting H&M GAE no-rules run", flush=True)
    returncode, txt = _run_and_stream(cmd, env)
    if returncode != 0:
        raise RuntimeError(f"RAD script exited {returncode} for seed={seed}")
    m = re.search(
        r"FINAL COMPARISON \(Val Set.*?At-Risk Filtered Labels\).*?"
        r"RAD \(rule-augmented\)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)",
        txt,
        flags=re.S,
    )
    if not m:
        raise RuntimeError(f"Could not parse RAD metrics from output for seed={seed}")

    return pd.Series(
        {
            "seed": int(seed),
            "auprc": float(m.group(1)),
            "auroc": float(m.group(2)),
            "p@k": float(m.group(3)),
            "lift": float(m.group(4)),
        }
    )


def main() -> None:
    n_seeds = int(os.environ.get("N_SEEDS", "10"))
    start_seed = int(os.environ.get("START_SEED", "0"))
    seeds = list(range(start_seed, start_seed + n_seeds))

    print("=" * 80, flush=True)
    print(f"10-SEED H&M GAE no-rules | seeds={seeds}", flush=True)
    print("=" * 80, flush=True)

    per_seed_csv = OUT_DIR / "per_seed.csv"
    summary_csv = OUT_DIR / "summary_mean_std.csv"

    rows = []
    for seed in seeds:
        row = _run_seed(seed)
        rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(per_seed_csv, index=False)
        print(f"  [seed={seed}] auprc={row['auprc']:.4f}  auroc={row['auroc']:.4f}  p@k={row['p@k']:.4f}  lift={row['lift']:.4f}", flush=True)
        print(f"  Progress saved -> {per_seed_csv}", flush=True)

    df = pd.DataFrame(rows)
    metrics = ["auprc", "auroc", "p@k", "lift"]
    summary = df[metrics].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80, flush=True)
    print("10-SEED SUMMARY — H&M GAE no-rules", flush=True)
    print("=" * 80, flush=True)
    for _, r in summary.iterrows():
        print(f"  {r['metric']:<8}  {r['mean']:.4f} +/- {r['std']:.4f}", flush=True)
    print(f"\nPer-seed:  {per_seed_csv}", flush=True)
    print(f"Summary:   {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
