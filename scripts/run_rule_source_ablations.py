#!/usr/bin/env python3
"""
Run RAD rule-source ablations for Amazon and H&M across multiple seeds.

Ablations:
- LLM-derived rules (current behavior)
- RF sampled rules only (no LLM generation)
- RF decision features (tree predictions as rule features)

By default this runs both structural-BCE settings:
- with BCE  (lambda_struct=0.1)
- no BCE    (lambda_struct=0.0)

Outputs:
- results/shared/rule_source_ablation/per_seed_metrics.csv
- results/shared/rule_source_ablation/mean_std_metrics.csv
- results/shared/rule_source_ablation/mean_std_metrics_pretty.csv
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd

from path_helpers import RESULTS_JSONL_PATH

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

DATASET_TO_SCRIPT = {
    "amazon": ROOT / "scripts" / "rad_vs_supervised_rdl.py",
    "hm": ROOT / "scripts" / "hm_rad_vs_supervised_rdl.py",
}

OUT_DIR = ROOT / "results" / os.environ.get("OUT_SUBDIR", "shared/rule_source_ablation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_SEED_OUT = OUT_DIR / "per_seed_metrics.csv"
MEAN_STD_OUT = OUT_DIR / "mean_std_metrics.csv"
PRETTY_OUT = OUT_DIR / "mean_std_metrics_pretty.csv"


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


def _append_rows(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    add_df = pd.DataFrame(rows)
    if df.empty:
        out = add_df
    else:
        out = pd.concat([df, add_df], ignore_index=True)
    key_cols = ["dataset", "seed", "model"]
    return out.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)


def _save_outputs(per_seed: pd.DataFrame) -> None:
    per_seed = per_seed.sort_values(["dataset", "seed", "model"]).reset_index(drop=True)
    per_seed.to_csv(PER_SEED_OUT, index=False)

    agg = per_seed.groupby(["dataset", "model"], as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"),
        auprc_std=("auprc", "std"),
        p_at_k_mean=("p@k", "mean"),
        p_at_k_std=("p@k", "std"),
        lift_mean=("lift", "mean"),
        lift_std=("lift", "std"),
    )
    agg = agg.sort_values(["dataset", "auprc_mean", "auroc_mean"], ascending=[True, False, False]).reset_index(drop=True)
    agg.to_csv(MEAN_STD_OUT, index=False)

    pretty = pd.DataFrame(
        {
            "dataset": agg["dataset"],
            "model": agg["model"],
            "auroc": agg.apply(lambda r: f"{r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f}", axis=1),
            "auprc": agg.apply(lambda r: f"{r['auprc_mean']:.4f} +/- {r['auprc_std']:.4f}", axis=1),
            "p@k": agg.apply(lambda r: f"{r['p_at_k_mean']:.4f} +/- {r['p_at_k_std']:.4f}", axis=1),
            "lift": agg.apply(lambda r: f"{r['lift_mean']:.4f} +/- {r['lift_std']:.4f}", axis=1),
        }
    )
    pretty.to_csv(PRETTY_OUT, index=False)


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


def _run_one(
    dataset: str,
    seed: int,
    lambda_struct: float,
    rf_rule_mode: str,
    rf_decision_trees: int,
    use_cached_llm_rules: bool,
) -> dict[str, float]:
    script = DATASET_TO_SCRIPT[dataset]
    run_tag = f"rulesource_{dataset}_seed{seed}_{rf_rule_mode}_ls{lambda_struct}_{uuid.uuid4().hex[:8]}"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "SEED": str(seed),
            "RUN_TAG": run_tag,
            "LAMBDA_STRUCT": str(lambda_struct),
            "RF_RULE_MODE": rf_rule_mode,
            "RF_DECISION_TREES": str(rf_decision_trees),
            "USE_CACHED_LLM_RULES": "1" if use_cached_llm_rules else "0",
            "SUP_EPOCHS": env.get("SUP_EPOCHS", "0"),
            "SUP_PATIENCE": env.get("SUP_PATIENCE", "3"),
            "EPOCHS": env.get("EPOCHS", "20"),
            "PATIENCE": env.get("PATIENCE", "5"),
            "WARMUP_EPOCHS": env.get("WARMUP_EPOCHS", "5"),
        }
    )

    print(
        f"[dataset={dataset} seed={seed}] running rf_rule_mode={rf_rule_mode} "
        f"lambda_struct={lambda_struct} rf_decision_trees={rf_decision_trees}",
        flush=True,
    )
    _run_and_stream([sys.executable, str(script)], env)
    rec = _read_last_jsonl_record(run_tag, "RAD (rule-augmented)")
    return {
        "auprc": float(rec["auprc"]),
        "auroc": float(rec["auroc"]),
        "p@k": float(rec["p@k"]),
        "lift": float(rec["lift"]),
    }


def main() -> None:
    datasets_env = os.environ.get("DATASETS", "amazon,hm")
    datasets = [d.strip().lower() for d in datasets_env.split(",") if d.strip()]
    bad = [d for d in datasets if d not in DATASET_TO_SCRIPT]
    if bad:
        raise ValueError(f"Unsupported DATASETS entries: {bad}; choose from {sorted(DATASET_TO_SCRIPT)}")

    n_seeds = int(os.environ.get("N_SEEDS", "10"))
    start_seed = int(os.environ.get("START_SEED", "0"))
    seeds = list(range(start_seed, start_seed + n_seeds))

    include_no_bce = os.environ.get("INCLUDE_NO_BCE", "1").strip() != "0"
    lambda_variants = [("with BCE", 0.1)]
    if include_no_bce:
        lambda_variants.append(("no BCE", 0.0))

    rf_decision_trees = int(os.environ.get("RF_DECISION_TREES", "15"))
    use_cached_llm_rules = os.environ.get("USE_CACHED_LLM_RULES", "1").strip() == "1"

    mode_variants = [
        ("LLM rules", "llm"),
        ("RF sampled rules", "rf_seed"),
        (f"RF decision features ({rf_decision_trees} trees)", "rf_decision"),
    ]

    print("=" * 96)
    print(
        f"RUNNING RULE-SOURCE ABLATIONS | datasets={datasets} seeds={seeds} "
        f"lambdas={[x[1] for x in lambda_variants]}"
    )
    print("=" * 96)

    if PER_SEED_OUT.exists():
        per_seed = pd.read_csv(PER_SEED_OUT)
        print(f"[resume] Loaded checkpoint: {PER_SEED_OUT} rows={len(per_seed)}")
    else:
        per_seed = pd.DataFrame(columns=["dataset", "model", "auroc", "auprc", "p@k", "lift", "seed"])

    for dataset in datasets:
        for seed in seeds:
            done_models = set(
                per_seed.loc[
                    (per_seed["dataset"] == dataset) & (per_seed["seed"] == seed), "model"
                ]
                .astype(str)
                .tolist()
            )

            for lambda_label, lambda_struct in lambda_variants:
                for mode_label, mode in mode_variants:
                    model_name = f"RAD ({mode_label}, {lambda_label})"
                    if model_name in done_models:
                        print(f"[dataset={dataset} seed={seed}] skipping {model_name} (checkpointed)")
                        continue

                    m = _run_one(
                        dataset=dataset,
                        seed=seed,
                        lambda_struct=lambda_struct,
                        rf_rule_mode=mode,
                        rf_decision_trees=rf_decision_trees,
                        use_cached_llm_rules=use_cached_llm_rules,
                    )
                    per_seed = _append_rows(
                        per_seed,
                        [
                            {
                                "dataset": dataset,
                                "model": model_name,
                                "auroc": m["auroc"],
                                "auprc": m["auprc"],
                                "p@k": m["p@k"],
                                "lift": m["lift"],
                                "seed": seed,
                            }
                        ],
                    )
                    _save_outputs(per_seed)
                    print(f"[checkpoint] saved {dataset} seed={seed} {model_name}")

    print(f"[done] wrote {PER_SEED_OUT}")
    print(f"[done] wrote {MEAN_STD_OUT}")
    print(f"[done] wrote {PRETTY_OUT}")


if __name__ == "__main__":
    main()
