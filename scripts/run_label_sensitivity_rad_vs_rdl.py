#!/usr/bin/env python3
"""Label-sensitivity ablation for RAD vs supervised RDL on Amazon and H&M.

Runs existing single-seed scripts with a sweep over:
- TARGET_ANOM_RATE
- ATRISK_TAU_MULTIPLIER
- seed

Outputs:
- results/shared/label_sensitivity/per_run_metrics.csv
- results/shared/label_sensitivity/summary_mean_std.csv
- results/shared/label_sensitivity/ratio_card.csv
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT_DIR / "results" / "shared" / "label_sensitivity"

SCRIPT_BY_DATASET = {
    "amazon": ROOT_DIR / "scripts" / "rad_vs_supervised_rdl.py",
    "hm": ROOT_DIR / "scripts" / "hm_rad_vs_supervised_rdl.py",
}

ACTUAL_EXPERIMENT_RAD_ENV = {
    "LAMBDA_RANK": "4.0",
    "ALPHA_FEAT": "1.0",
    "ALPHA_STRUCT": "1.0",
    "WARMUP_RANK_SCALE": "1.0",
    "TUNE_METRIC": "auprc",
    "USE_CACHED_LLM_RULES": "1",
}

DATASET_ACTUAL_EXPERIMENT_ENV = {
    "amazon": {
        "EPOCHS": "20",
        "PATIENCE": "5",
        "WARMUP_EPOCHS": "5",
        "SUP_EPOCHS": "0",
        "SUP_PATIENCE": "3",
    },
    "hm": {
        "LATENT_DIM": "48",
        "EPOCHS": "30",
        "PATIENCE": "5",
        "WARMUP_EPOCHS": "5",
        "WARMUP_STRUCT_SCALE": "1.0",
    },
}


@dataclass
class SweepConfig:
    datasets: List[str]
    target_rates: List[float]
    tau_multipliers: List[float]
    seeds: List[int]
    out_dir: Path
    fail_fast: bool


def _parse_list_floats(env_key: str, default_csv: str) -> List[float]:
    raw = os.environ.get(env_key, default_csv)
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{env_key} produced an empty list")
    return vals


def _parse_list_ints(env_key: str, default_csv: str) -> List[int]:
    raw = os.environ.get(env_key, default_csv)
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{env_key} produced an empty list")
    return vals


def _parse_list_str(env_key: str, default_csv: str) -> List[str]:
    raw = os.environ.get(env_key, default_csv)
    vals = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{env_key} produced an empty list")
    bad = [x for x in vals if x not in SCRIPT_BY_DATASET]
    if bad:
        raise ValueError(f"Unsupported datasets in {env_key}: {bad}")
    return vals


def read_config() -> SweepConfig:
    out_dir = Path(os.environ.get("OUT_DIR", str(DEFAULT_OUT_DIR)))
    return SweepConfig(
        datasets=_parse_list_str("DATASETS", "amazon,hm"),
        target_rates=_parse_list_floats("TARGET_ANOM_RATES", "0.005,0.01,0.015,0.02,0.03"),
        tau_multipliers=_parse_list_floats("ATRISK_TAU_MULTIPLIERS", "0.6,0.8,1.0"),
        seeds=_parse_list_ints("SEEDS", "0,1,2"),
        out_dir=out_dir,
        fail_fast=os.environ.get("FAIL_FAST", "0") == "1",
    )


def _slug_rate(x: float) -> str:
    return f"{x:.4f}".replace(".", "p")


def _extract_section(text: str, start_token: str, end_token: str) -> str:
    i = text.find(start_token)
    if i < 0:
        return ""
    j = text.find(end_token, i + len(start_token))
    if j < 0:
        return text[i:]
    return text[i:j]


def _parse_metrics(stdout: str) -> Dict[str, float]:
    # Normalize Unicode dashes used by notebook/script logging.
    text = stdout.replace("\u2014", "-").replace("\u2013", "-")

    filtered_sec = ""
    orig_sec = ""
    for split in ("Test", "Val"):
        filtered_sec = _extract_section(
            text,
            f"FINAL COMPARISON ({split} Set - At-Risk Filtered Labels)",
            f"FINAL COMPARISON ({split} Set - Original Unfiltered Labels)",
        )
        orig_sec = _extract_section(
            text,
            f"FINAL COMPARISON ({split} Set - Original Unfiltered Labels)",
            "Sweep prevalence target:",
        )
        if filtered_sec and orig_sec:
            break

    if not filtered_sec:
        filtered_sec = _extract_section(
            text,
            "FINAL COMPARISON (",  # fallback for historical logs
            "FINAL COMPARISON (",
        )

    re_val = re.compile(
        r"(?:Val|Test):\s*([\d,]+)\s*users\s*\|\s*([\d,]+)\s*anomalies\s*\(([0-9.]+)\s*rate\)",
        flags=re.IGNORECASE,
    )
    re_sup = re.compile(r"Supervised RDL GNN\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")
    re_rad = re.compile(r"RAD \(rule-augmented\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")
    re_sup_orig = re.compile(r"Supervised RDL GNN\s+([0-9.]+)\s+([0-9.]+)")
    re_rad_orig = re.compile(r"RAD \(rule-augmented\)\s+([0-9.]+)\s+([0-9.]+)")

    f_val = re_val.search(filtered_sec)
    o_val = re_val.search(orig_sec)
    f_sup = re_sup.search(filtered_sec)
    f_rad = re_rad.search(filtered_sec)
    o_sup = re_sup_orig.search(orig_sec)
    o_rad = re_rad_orig.search(orig_sec)

    if not (f_val and o_val and f_sup and f_rad and o_sup and o_rad):
        raise ValueError("Could not parse required metrics from run output")

    return {
        "filtered_users": int(f_val.group(1).replace(",", "")),
        "filtered_anomalies": int(f_val.group(2).replace(",", "")),
        "filtered_rate": float(f_val.group(3)),
        "orig_users": int(o_val.group(1).replace(",", "")),
        "orig_anomalies": int(o_val.group(2).replace(",", "")),
        "orig_rate": float(o_val.group(3)),
        "sup_auprc": float(f_sup.group(1)),
        "sup_auroc": float(f_sup.group(2)),
        "sup_p_at_k": float(f_sup.group(3)),
        "sup_lift": float(f_sup.group(4)),
        "rad_auprc": float(f_rad.group(1)),
        "rad_auroc": float(f_rad.group(2)),
        "rad_p_at_k": float(f_rad.group(3)),
        "rad_lift": float(f_rad.group(4)),
        "sup_orig_auprc": float(o_sup.group(1)),
        "sup_orig_auroc": float(o_sup.group(2)),
        "rad_orig_auprc": float(o_rad.group(1)),
        "rad_orig_auroc": float(o_rad.group(2)),
    }


def _build_env(dataset: str, target_rate: float, tau_multiplier: float, seed: int) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SEED": str(seed),
            "SUP_LOSS_MODE": "unweighted",
            "TARGET_ANOM_RATE": str(target_rate),
            "ATRISK_TAU_MULTIPLIER": str(tau_multiplier),
            "POS_DOWNSAMPLE_MODE": os.environ.get("POS_DOWNSAMPLE_MODE", "subtle"),
            "ATRISK_TAU_A": os.environ.get("ATRISK_TAU_A", "0.08"),
            "ATRISK_TAU_B": os.environ.get("ATRISK_TAU_B", "6"),
            "ATRISK_TAU_ALPHA": os.environ.get("ATRISK_TAU_ALPHA", "0.05"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update({k: os.environ.get(k, v) for k, v in ACTUAL_EXPERIMENT_RAD_ENV.items()})
    env.update(
        {
            k: os.environ.get(k, v)
            for k, v in DATASET_ACTUAL_EXPERIMENT_ENV[dataset].items()
        }
    )
    cuda_device = os.environ.get("CUDA_DEVICE_INDEX")
    if cuda_device is not None and cuda_device != "":
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
    return env


def _run_key(dataset: str, target_rate: float, tau_multiplier: float, seed: int) -> str:
    return f"{dataset}|{target_rate:.4f}|{tau_multiplier:.4f}|{int(seed)}"


def _write_checkpoint(rows: List[Dict[str, object]], per_run_path: Path) -> None:
    pd.DataFrame(rows).to_csv(per_run_path, index=False)


def _load_existing_rows(per_run_path: Path) -> List[Dict[str, object]]:
    if not per_run_path.exists():
        return []
    df = pd.read_csv(per_run_path)
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _parse_log_filename(name: str):
    m = re.match(
        r"^(amazon|hm)__target_([0-9p]+)__tau_([0-9p]+)__seed_([0-9]+)\.log$",
        name,
    )
    if not m:
        return None
    dataset = m.group(1)
    target_rate = float(m.group(2).replace("p", "."))
    tau_mult = float(m.group(3).replace("p", "."))
    seed = int(m.group(4))
    return dataset, target_rate, tau_mult, seed


def _backfill_rows_from_logs(
    logs_dir: Path,
    existing_keys: set[str],
) -> List[Dict[str, object]]:
    recovered: List[Dict[str, object]] = []
    for log_path in sorted(logs_dir.glob("*.log")):
        parsed = _parse_log_filename(log_path.name)
        if parsed is None:
            continue
        dataset, target_rate, tau_mult, seed = parsed
        key = _run_key(dataset, target_rate, tau_mult, seed)
        if key in existing_keys:
            continue

        text = log_path.read_text(encoding="utf-8", errors="ignore")
        try:
            m = _parse_metrics(text)
            recovered.append(
                {
                    "dataset": dataset,
                    "target_anom_rate": target_rate,
                    "tau_multiplier": tau_mult,
                    "seed": seed,
                    "status": "ok",
                    "log_path": str(log_path),
                    **m,
                    "auprc_gap_rad_minus_rdl": m["rad_auprc"] - m["sup_auprc"],
                    "auroc_gap_rad_minus_rdl": m["rad_auroc"] - m["sup_auroc"],
                    "p_at_k_gap_rad_minus_rdl": m["rad_p_at_k"] - m["sup_p_at_k"],
                }
            )
        except Exception:
            recovered.append(
                {
                    "dataset": dataset,
                    "target_anom_rate": target_rate,
                    "tau_multiplier": tau_mult,
                    "seed": seed,
                    "status": "failed_existing",
                    "log_path": str(log_path),
                }
            )
    return recovered


def run_sweep(cfg: SweepConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = cfg.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cfg.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = cfg.out_dir / "per_run_metrics.csv"

    rows: List[Dict[str, object]] = _load_existing_rows(per_run_path)
    existing_keys = {
        _run_key(
            str(r.get("dataset", "")).strip().lower(),
            float(r.get("target_anom_rate", 0.0)),
            float(r.get("tau_multiplier", 0.0)),
            int(r.get("seed", -1)),
        )
        for r in rows
        if r.get("dataset") is not None and r.get("seed") is not None
    }

    recovered = _backfill_rows_from_logs(logs_dir, existing_keys)
    if recovered:
        rows.extend(recovered)
        _write_checkpoint(rows, per_run_path)
        existing_keys.update(
            _run_key(r["dataset"], float(r["target_anom_rate"]), float(r["tau_multiplier"]), int(r["seed"]))
            for r in recovered
        )
        print(f"[resume] recovered {len(recovered)} rows from existing logs", flush=True)

    total = len(cfg.datasets) * len(cfg.target_rates) * len(cfg.tau_multipliers) * len(cfg.seeds)
    run_idx = 0

    print("=" * 88, flush=True)
    print("Label-sensitivity ablation: RAD vs supervised RDL", flush=True)
    print(f"datasets={cfg.datasets}", flush=True)
    print(f"target_rates={cfg.target_rates}", flush=True)
    print(f"tau_multipliers={cfg.tau_multipliers}", flush=True)
    print(f"seeds={cfg.seeds}", flush=True)
    print(f"out_dir={cfg.out_dir}", flush=True)
    print("=" * 88, flush=True)

    for dataset in cfg.datasets:
        script = SCRIPT_BY_DATASET[dataset]
        for target_rate in cfg.target_rates:
            for tau_mult in cfg.tau_multipliers:
                for seed in cfg.seeds:
                    run_idx += 1
                    tag = (
                        f"dataset={dataset} target={target_rate:.4f} "
                        f"tau_mult={tau_mult:.3f} seed={seed}"
                    )

                    key = _run_key(dataset, target_rate, tau_mult, seed)
                    if key in existing_keys:
                        print(f"[{run_idx}/{total}] {tag} [skip: already recorded]", flush=True)
                        continue

                    print(f"[{run_idx}/{total}] {tag}", flush=True)

                    log_name = (
                        f"{dataset}__target_{_slug_rate(target_rate)}"
                        f"__tau_{_slug_rate(tau_mult)}__seed_{seed}.log"
                    )
                    log_path = logs_dir / log_name

                    env = _build_env(dataset, target_rate, tau_mult, seed)
                    env["SAVE_RAW_RESULTS_PATH"] = str(raw_dir / f"{log_name[:-4]}.pt")
                    proc = subprocess.run(
                        [sys.executable, str(script)],
                        cwd=str(ROOT_DIR),
                        env=env,
                        capture_output=True,
                        text=True,
                    )

                    with log_path.open("w", encoding="utf-8") as f:
                        f.write(proc.stdout)
                        if proc.stderr:
                            f.write("\n\n[stderr]\n")
                            f.write(proc.stderr)

                    if proc.returncode != 0:
                        msg = f"Run failed ({proc.returncode}) for {tag}. Log: {log_path}"
                        print(msg, flush=True)
                        if cfg.fail_fast:
                            raise RuntimeError(msg)
                        rows.append(
                            {
                                "dataset": dataset,
                                "target_anom_rate": target_rate,
                                "tau_multiplier": tau_mult,
                                "seed": seed,
                                "status": "failed",
                                "log_path": str(log_path),
                            }
                        )
                        existing_keys.add(key)
                        _write_checkpoint(rows, per_run_path)
                        continue

                    try:
                        m = _parse_metrics(proc.stdout)
                    except Exception as ex:
                        msg = f"Parse failed for {tag}: {ex}. Log: {log_path}"
                        print(msg, flush=True)
                        if cfg.fail_fast:
                            raise
                        rows.append(
                            {
                                "dataset": dataset,
                                "target_anom_rate": target_rate,
                                "tau_multiplier": tau_mult,
                                "seed": seed,
                                "status": "parse_failed",
                                "log_path": str(log_path),
                            }
                        )
                        existing_keys.add(key)
                        _write_checkpoint(rows, per_run_path)
                        continue

                    rows.append(
                        {
                            "dataset": dataset,
                            "target_anom_rate": target_rate,
                            "tau_multiplier": tau_mult,
                            "seed": seed,
                            "status": "ok",
                            "log_path": str(log_path),
                            **m,
                            "auprc_gap_rad_minus_rdl": m["rad_auprc"] - m["sup_auprc"],
                            "auroc_gap_rad_minus_rdl": m["rad_auroc"] - m["sup_auroc"],
                            "p_at_k_gap_rad_minus_rdl": m["rad_p_at_k"] - m["sup_p_at_k"],
                        }
                    )
                    existing_keys.add(key)
                    _write_checkpoint(rows, per_run_path)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(per_run_path, index=False)
    print(f"[done] wrote {per_run_path}", flush=True)

    ok = per_run[per_run["status"] == "ok"].copy()
    if ok.empty:
        print("No successful runs to summarize.", flush=True)
        return

    metric_cols = [
        "orig_rate",
        "filtered_rate",
        "sup_auprc",
        "sup_auroc",
        "sup_p_at_k",
        "sup_lift",
        "rad_auprc",
        "rad_auroc",
        "rad_p_at_k",
        "rad_lift",
        "auprc_gap_rad_minus_rdl",
        "auroc_gap_rad_minus_rdl",
        "p_at_k_gap_rad_minus_rdl",
    ]

    summary = (
        ok.groupby(["dataset", "target_anom_rate", "tau_multiplier"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        c if isinstance(c, str) else (c[0] if c[1] == "" else f"{c[0]}_{c[1]}")
        for c in summary.columns
    ]
    summary_path = cfg.out_dir / "summary_mean_std.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[done] wrote {summary_path}", flush=True)

    ratio_card = (
        ok.groupby(["dataset", "target_anom_rate", "tau_multiplier"], as_index=False)
        .agg(
            raw_positive_rate=("orig_rate", "mean"),
            unexpected_positive_rate=("filtered_rate", "mean"),
            raw_users=("orig_users", "mean"),
            raw_anomalies=("orig_anomalies", "mean"),
            unexpected_users=("filtered_users", "mean"),
            unexpected_anomalies=("filtered_anomalies", "mean"),
            n_success=("status", "count"),
        )
        .sort_values(["dataset", "target_anom_rate", "tau_multiplier"])
    )
    ratio_path = cfg.out_dir / "ratio_card.csv"
    ratio_card.to_csv(ratio_path, index=False)
    print(f"[done] wrote {ratio_path}", flush=True)


def main() -> None:
    cfg = read_config()
    run_sweep(cfg)


if __name__ == "__main__":
    main()
