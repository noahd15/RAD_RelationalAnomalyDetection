#!/usr/bin/env python3
"""Sweep LATENT_DIM for RAD on H&M to find optimal bottleneck for anomaly detection."""

from __future__ import annotations
import os, re, subprocess, sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)
RAD_SCRIPT = ROOT_DIR / "scripts" / "hm_rad_vs_supervised_rdl.py"
OUT_DIR = ROOT_DIR / "results" / "hm_rad_bottleneck_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LATENT_DIMS = [4, 8, 16, 32, 48]
N_SEEDS = int(os.environ.get("N_SEEDS", "3"))
START_SEED = int(os.environ.get("START_SEED", "0"))

def run_rad(seed: int, latent_dim: int) -> dict:
    env = os.environ.copy()
    env.update({
        "SEED": str(seed),
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "1"),
        "LATENT_DIM": str(latent_dim),
        "LAMBDA_STRUCT": "0.1",
        "LAMBDA_RANK": "1.0",
        "ALPHA_FEAT": "1.0",
        "ALPHA_STRUCT": "1.0",
        "TUNE_METRIC": "auprc",
        "EPOCHS": env.get("EPOCHS", "30"),
        "PATIENCE": env.get("PATIENCE", "5"),
        "WARMUP_EPOCHS": env.get("WARMUP_EPOCHS", "5"),
        "WARMUP_RANK_SCALE": env.get("WARMUP_RANK_SCALE", "0.25"),
        "WARMUP_STRUCT_SCALE": env.get("WARMUP_STRUCT_SCALE", "1.0"),
        "USE_CACHED_LLM_RULES": "1",
    })
    print(f"[seed={seed}] LATENT_DIM={latent_dim} — running RAD", flush=True)
    proc = subprocess.run([sys.executable, str(RAD_SCRIPT)], env=env, capture_output=True, text=True)
    if proc.stdout: print(proc.stdout, end="")
    if proc.stderr: print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"RAD script exited {proc.returncode} (seed={seed}, latent_dim={latent_dim})")

    txt = proc.stdout or ""
    m = re.search(
        r"FINAL COMPARISON \(Val Set.*?At-Risk Filtered Labels\).*?"
        r"RAD \(rule-augmented\)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)",
        txt, flags=re.S,
    )
    if not m:
        raise RuntimeError(f"Could not parse RAD metrics (seed={seed}, latent_dim={latent_dim})")
    return {"seed": seed, "latent_dim": latent_dim,
            "auprc": float(m.group(1)), "auroc": float(m.group(2)),
            "p@k": float(m.group(3)), "lift": float(m.group(4))}

def main():
    import pandas as pd, numpy as np
    seeds = list(range(START_SEED, START_SEED + N_SEEDS))
    print("=" * 80)
    print(f"H&M RAD BOTTLENECK SWEEP | latent_dims={LATENT_DIMS} | seeds={seeds}")
    print("=" * 80)

    rows = []
    for latent_dim in LATENT_DIMS:
        for seed in seeds:
            rows.append(run_rad(seed, latent_dim))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_seed.csv", index=False)

    grp = df.groupby("latent_dim")[["auprc", "auroc", "p@k", "lift"]]
    summary = pd.concat([grp.mean().add_suffix("_mean"), grp.std().add_suffix("_std")], axis=1)
    summary.to_csv(OUT_DIR / "summary.csv")

    print("\n=== BOTTLENECK SWEEP SUMMARY (mean +/- std) ===")
    for dim, row in summary.iterrows():
        print(f"  latent_dim={dim:3d}  auprc={row['auprc_mean']:.4f}+/-{row['auprc_std']:.4f}"
              f"  auroc={row['auroc_mean']:.4f}+/-{row['auroc_std']:.4f}"
              f"  p@k={row['p@k_mean']:.4f}+/-{row['p@k_std']:.4f}")

if __name__ == "__main__":
    main()
