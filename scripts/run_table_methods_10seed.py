#!/usr/bin/env python3
"""
Run 10-seed evaluation for methods shown in comparison_table_0507.

Methods:
- Isolation Forest
- DRL
- Deep SVDD
- AE
- AE+Rank
- RAD (rules, BCE)
- RAD (rules, no BCE)
- RAD (no rules, BCE)
- RAD (no rules, no BCE)

Outputs:
- results/amazon/full_ablation_10seed_table/per_seed_metrics.csv
- results/amazon/full_ablation_10seed_table/mean_std_metrics.csv
- results/amazon/full_ablation_10seed_table/mean_std_metrics_pretty.csv
- results/amazon/full_ablation_10seed_table/mean_std_metrics.png
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

BASELINES_SCRIPT    = ROOT / "scripts" / "amazon_user_churn_baselines.py"
BASELINES_MS_SCRIPT = ROOT / "scripts" / "run_baselines_multiseed.py"
RAD_SCRIPT          = ROOT / "scripts" / "rad_vs_supervised_rdl.py"
OUT_DIR = ROOT / "results" / os.environ.get("OUT_SUBDIR", "amazon/full_ablation_10seed_table")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_SEED_OUT = OUT_DIR / "per_seed_metrics.csv"
MEAN_STD_OUT = OUT_DIR / "mean_std_metrics.csv"
PRETTY_OUT = OUT_DIR / "mean_std_metrics_pretty.csv"
PNG_OUT = OUT_DIR / "mean_std_metrics.png"
PARTIAL_AUROC_MAX_FPR = float(os.environ.get("PARTIAL_AUROC_MAX_FPR", "0.1"))


def _partial_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    scores = np.asarray(scores).astype(float).reshape(-1)
    if np.unique(labels).size <= 1 or not (0.0 < PARTIAL_AUROC_MAX_FPR < 1.0):
        return float("nan")
    return float(roc_auc_score(labels, scores, max_fpr=PARTIAL_AUROC_MAX_FPR))


def _load_rad_pauroc(raw_path: Path) -> float:
    raw_obj = torch.load(raw_path, map_location="cpu", weights_only=False)
    return _partial_auroc(raw_obj["test_labels"], raw_obj["test_scores"])


def _run_and_stream(cmd: list[str], env: dict[str, str]) -> str:
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    rc = proc.wait()
    out = "".join(chunks)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output=out)
    return out


def _build_baseline_env(seed: int | None = None) -> dict[str, str]:
    """Return the env dict used for any baseline subprocess call."""
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    include_supervised = os.environ.get("INCLUDE_SUPERVISED_RDL", "1").strip() != "0"
    exit_after_ae_default = "0" if include_supervised else "1"
    update: dict[str, str] = {
        "PYTHONUNBUFFERED":   "1",
        "BASELINES_DEVICE":   env.get("BASELINES_DEVICE", "cuda"),
        "TARGET_ANOM_RATE":   env.get("TARGET_ANOM_RATE", "0.015"),
        "TRAIN_EQ_RATE":      env.get("TRAIN_EQ_RATE", "1"),
        "AE_RANK_POOL_MODE":  env.get("AE_RANK_POOL_MODE", "labeled"),
        "CAUSAL_GUARD":       env.get("CAUSAL_GUARD", "1"),
        "CAUSAL_GUARD_SAMPLES": env.get("CAUSAL_GUARD_SAMPLES", "128"),
        "EXIT_AFTER_AE":      env.get("EXIT_AFTER_AE", exit_after_ae_default),
        "DRL_EPOCHS":         env.get("DRL_EPOCHS", "0"),
        "SVDD_EPOCHS":        env.get("SVDD_EPOCHS", "0"),
        "INCLUDE_SUPERVISED_RDL": "1" if include_supervised else "0",
    }
    if seed is not None:
        update["SEED"] = str(seed)
    env.update(update)
    return env


def _parse_baseline_csv(csv_path: Path, seed: int) -> pd.DataFrame:
    include_supervised = os.environ.get("INCLUDE_SUPERVISED_RDL", "1").strip() != "0"
    if not csv_path.exists():
        raise RuntimeError(f"Missing baselines output: {csv_path}")
    df   = pd.read_csv(csv_path)
    keep = {"IForest", "DRL", "DeepSVDD", "VanillaAE", "AE-Rank"}
    if include_supervised:
        keep.add("Supervised RDL")
    df = df[df["model"].isin(keep)].copy()
    alias = {
        "IForest":        "Isolation Forest",
        "DRL":            "DRL",
        "DeepSVDD":       "Deep SVDD",
        "VanillaAE":      "AE",
        "AE-Rank":        "AE+Rank",
        "Supervised RDL": "Supervised RDL",
    }
    df["model"] = df["model"].map(alias)
    df["seed"]  = seed
    if "pauroc" not in df.columns:
        df["pauroc"] = float("nan")
    return df[["model", "auroc", "auprc", "pauroc", "p@k", "lift", "seed"]]


def _run_baselines_multiseed(seeds: list[int]) -> dict[int, pd.DataFrame]:
    """Run the multi-seed baselines script ONCE for all requested seeds.

    Loads the RelBench DB only once, then loops over seeds internally,
    avoiding redundant database loads.
    """
    env = _build_baseline_env()
    env["MULTI_SEED"] = ",".join(str(s) for s in seeds)
    print(f"[baselines] Running multi-seed: seeds={seeds}", flush=True)
    _run_and_stream([sys.executable, str(BASELINES_MS_SCRIPT)], env)

    ms_csv = ROOT / "results" / "full_ablation" / "baselines_metrics_multiseed.csv"
    if not ms_csv.exists():
        raise RuntimeError(f"Missing multi-seed baselines output: {ms_csv}")

    ms_df = pd.read_csv(ms_csv)
    include_supervised = os.environ.get("INCLUDE_SUPERVISED_RDL", "1").strip() != "0"
    keep  = {"IForest", "DRL", "DeepSVDD", "VanillaAE", "AE-Rank"}
    if include_supervised:
        keep.add("Supervised RDL")
    ms_df = ms_df[ms_df["model"].isin(keep)].copy()
    alias = {
        "IForest":        "Isolation Forest",
        "DRL":            "DRL",
        "DeepSVDD":       "Deep SVDD",
        "VanillaAE":      "AE",
        "AE-Rank":        "AE+Rank",
        "Supervised RDL": "Supervised RDL",
    }
    ms_df["model"] = ms_df["model"].map(alias)
    if "pauroc" not in ms_df.columns:
        ms_df["pauroc"] = float("nan")

    result: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        df_s = ms_df[ms_df["seed"] == seed][["model", "auroc", "auprc", "pauroc", "p@k", "lift"]].copy()
        df_s["seed"] = seed
        result[seed] = df_s
    return result


def _run_rad_variant(seed: int, lambda_struct: float, disable_rules: bool) -> dict[str, float]:
    env = os.environ.copy()
    raw_path = None
    raw_dir = os.environ.get("SAVE_RAW_RESULTS_DIR", "").strip()
    if raw_dir:
        raw_name = f"seed_{seed}__disable_rules_{int(disable_rules)}__lambda_{str(lambda_struct).replace('.', 'p')}.pt"
        raw_path = Path(raw_dir) / raw_name
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        env["SAVE_RAW_RESULTS_PATH"] = str(raw_path)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "SEED": str(seed),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "1"),
            "USE_CACHED_LLM_RULES": env.get("USE_CACHED_LLM_RULES", "1"),
            "RAD_DISABLE_RULES": "1" if disable_rules else "0",
            "LAMBDA_STRUCT": str(lambda_struct),
            "EPOCHS": env.get("EPOCHS", "20"),
            "PATIENCE": env.get("PATIENCE", "5"),
            "WARMUP_EPOCHS": env.get("WARMUP_EPOCHS", "5"),
            # Supervised block in rad_vs_supervised_rdl.py is not used for RAD metric parsing
            # in this script, so keep it at 0 by default for speed.
            "SUP_EPOCHS": env.get("SUP_EPOCHS", "0"),
            "SUP_PATIENCE": env.get("SUP_PATIENCE", "3"),
        }
    )

    print(
        f"[seed={seed}] Running RAD: lambda_struct={lambda_struct}, disable_rules={int(disable_rules)}",
        flush=True,
    )
    out = _run_and_stream([sys.executable, str(RAD_SCRIPT)], env)

    m = re.search(
        r"FINAL COMPARISON \((?:Val|Test) Set.*?At-Risk Filtered Labels\).*?"
        r"RAD \(rule-augmented\)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)",
        out,
        flags=re.S,
    )
    if not m:
        raise RuntimeError("Could not parse RAD metrics from output")

    return {
        "auprc": float(m.group(1)),
        "auroc": float(m.group(2)),
        "pauroc": _load_rad_pauroc(raw_path) if raw_path is not None else float("nan"),
        "p@k": float(m.group(3)),
        "lift": float(m.group(4)),
    }


def _render_png(pretty: pd.DataFrame, out_png: Path) -> None:
    fig_h = 0.5 + 0.44 * (len(pretty) + 1)
    fig, ax = plt.subplots(figsize=(12.8, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=pretty.values,
        colLabels=pretty.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.5)

    for j in range(len(pretty.columns)):
        cell = tbl[(0, j)]
        cell.set_facecolor("#1f2937")
        cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")

    for i in range(1, len(pretty) + 1):
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        for j in range(len(pretty.columns)):
            tbl[(i, j)].set_facecolor(bg)

    n_seeds = int(os.environ.get("N_SEEDS", "10"))
    ax.set_title(f"{n_seeds}-Seed Summary (mean +/- std) — Causal Baselines + RAD Ablations", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_outputs(per_seed: pd.DataFrame) -> None:
    per_seed = per_seed.copy()
    per_seed = per_seed.sort_values(["seed", "model"]).reset_index(drop=True)
    per_seed.to_csv(PER_SEED_OUT, index=False)

    agg = per_seed.groupby("model", as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"),
        auprc_std=("auprc", "std"),
        pauroc_mean=("pauroc", "mean"),
        pauroc_std=("pauroc", "std"),
        p_at_k_mean=("p@k", "mean"),
        p_at_k_std=("p@k", "std"),
        lift_mean=("lift", "mean"),
        lift_std=("lift", "std"),
    )
    agg = agg.sort_values(["auprc_mean", "auroc_mean"], ascending=[False, False]).reset_index(drop=True)
    agg.to_csv(MEAN_STD_OUT, index=False)

    pretty = pd.DataFrame(
        {
            "model": agg["model"],
            "auroc": agg.apply(lambda r: f"{r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f}", axis=1),
            "auprc": agg.apply(lambda r: f"{r['auprc_mean']:.4f} +/- {r['auprc_std']:.4f}", axis=1),
            "pauroc": agg.apply(lambda r: f"{r['pauroc_mean']:.4f} +/- {r['pauroc_std']:.4f}", axis=1),
            "p@k": agg.apply(lambda r: f"{r['p_at_k_mean']:.4f} +/- {r['p_at_k_std']:.4f}", axis=1),
            "lift": agg.apply(lambda r: f"{r['lift_mean']:.4f} +/- {r['lift_std']:.4f}", axis=1),
        }
    )
    pretty.to_csv(PRETTY_OUT, index=False)
    _render_png(pretty, PNG_OUT)


def _append_rows(rows_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    add_df = pd.DataFrame(new_rows)
    if rows_df.empty:
        out = add_df
    else:
        out = pd.concat([rows_df, add_df], ignore_index=True)
    out = out.drop_duplicates(subset=["seed", "model"], keep="last")
    return out


def _completed_models(rows_df: pd.DataFrame, seed: int) -> set[str]:
    if rows_df.empty:
        return set()
    return set(rows_df.loc[rows_df["seed"] == seed, "model"].astype(str).tolist())


def main() -> None:
    n_seeds = int(os.environ.get("N_SEEDS", "10"))
    start_seed = int(os.environ.get("START_SEED", "0"))
    seeds = list(range(start_seed, start_seed + n_seeds))

    print("=" * 90)
    print(f"RUNNING {n_seeds}-SEED TABLE-METHOD EVAL | seeds={seeds}")
    print("=" * 90)

    if PER_SEED_OUT.exists():
        per_seed = pd.read_csv(PER_SEED_OUT)
        if "pauroc" not in per_seed.columns:
            per_seed["pauroc"] = float("nan")
        print(f"[resume] Loaded existing checkpoint: {PER_SEED_OUT} ({len(per_seed)} rows)")
    else:
        per_seed = pd.DataFrame(columns=["model", "auroc", "auprc", "pauroc", "p@k", "lift", "seed"])

    include_supervised = os.environ.get("INCLUDE_SUPERVISED_RDL", "1").strip() != "0"
    baseline_models = {
        "Isolation Forest",
        "DRL",
        "Deep SVDD",
        "AE",
        "AE+Rank",
    }
    if include_supervised:
        baseline_models.add("Supervised RDL")
    rad_variants = [
        ("RAD (rules, BCE)", 0.1, False),
        ("RAD (rules, no BCE)", 0.0, False),
        ("RAD (no rules, BCE)", 0.1, True),
        ("RAD (no rules, no BCE)", 0.0, True),
    ]

    # ── Run all pending baseline seeds in a SINGLE subprocess (DB loaded once) ─
    pending_baseline_seeds = [
        s for s in seeds
        if not baseline_models.issubset(_completed_models(per_seed, s))
    ]
    if pending_baseline_seeds:
        print(f"[baselines] Pending seeds: {pending_baseline_seeds} — loading DB once for all")
        baseline_results = _run_baselines_multiseed(pending_baseline_seeds)
        for seed in pending_baseline_seeds:
            if seed in baseline_results:
                per_seed = _append_rows(per_seed, baseline_results[seed].to_dict(orient="records"))
        _save_outputs(per_seed)
        print(f"[checkpoint] saved baselines for seeds {pending_baseline_seeds}")
    else:
        print("[baselines] All seeds already checkpointed — skipping baseline run")

    # ── Per-seed RAD variants ─────────────────────────────────────────────────
    for seed in seeds:
        done = _completed_models(per_seed, seed)

        for model, lambda_struct, disable_rules in rad_variants:
            if model in done:
                print(f"[seed={seed}] Skipping {model} (already checkpointed)")
                continue
            m = _run_rad_variant(seed=seed, lambda_struct=lambda_struct, disable_rules=disable_rules)
            per_seed = _append_rows(
                per_seed,
                [{
                    "model": model,
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "pauroc": m["pauroc"],
                    "p@k": m["p@k"],
                    "lift": m["lift"],
                    "seed": seed,
                }],
            )
            _save_outputs(per_seed)
            print(f"[checkpoint] saved {model} for seed={seed}")

    print(f"[done] wrote {PER_SEED_OUT}")
    print(f"[done] wrote {MEAN_STD_OUT}")
    print(f"[done] wrote {PRETTY_OUT}")
    print(f"[done] wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
