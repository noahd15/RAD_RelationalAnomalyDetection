#!/usr/bin/env python3
"""Unified 5-seed user-churn run for the table/bar-chart method bundle.

This orchestrator runs the required Amazon and H&M jobs in GPU-balanced phases,
collects logs in one place, and records a manifest of the produced artifacts.

Included jobs:
- Amazon table-methods 5-seed
- H&M full ablation 5-seed
- Amazon supervised RDL weighted 5-seed
- H&M supervised RDL weighted 5-seed
- Amazon supervised RDL unweighted 5-seed
- H&M supervised RDL unweighted 5-seed
- Amazon no-rules 5-seed
- H&M no-rules 5-seed
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DEFAULT_OUT_ROOT = ROOT / "results" / "shared" / "user_churn_unified_5seed_rankboost_20260520"
OUT_ROOT = Path(os.environ.get("OUT_ROOT", str(DEFAULT_OUT_ROOT)))
LOG_DIR = OUT_ROOT / "logs"
RAW_ROOT = OUT_ROOT / "raw_scores"
MANIFEST_CSV = OUT_ROOT / "manifest.csv"

RAD_COMMON_ENV = {
    "LAMBDA_RANK": "4.0",
    "ALPHA_FEAT": "1.0",
    "ALPHA_STRUCT": "1.0",
    "WARMUP_RANK_SCALE": "1.0",
    "TUNE_METRIC": "auprc",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    gpu: int
    out_subdir: str
    log_name: str
    command: list[str]
    env_updates: dict[str, str]


def build_tasks() -> list[list[TaskSpec]]:
    return [
        [
            TaskSpec(
                name="amazon_table_methods_5seed",
                gpu=0,
                out_subdir="amazon/full_ablation_5seed_table",
                log_name="amazon_table_methods_5seed.log",
                command=[PYTHON, str(ROOT / "scripts" / "run_table_methods_10seed.py")],
                env_updates={
                    "N_SEEDS": "5",
                    "OUT_SUBDIR": "amazon/full_ablation_5seed_table_rankboost",
                    "SAVE_RAW_RESULTS_DIR": str(RAW_ROOT / "amazon_table_methods"),
                    **RAD_COMMON_ENV,
                },
            ),
            TaskSpec(
                name="hm_full_ablation_5seed",
                gpu=1,
                out_subdir="hm/full_ablation_hm_5seed_rankboost",
                log_name="hm_full_ablation_5seed.log",
                command=[PYTHON, str(ROOT / "scripts" / "run_full_ablation_hm_5seed.py")],
                env_updates={
                    "OUT_SUBDIR": "hm/full_ablation_hm_5seed_rankboost",
                    **RAD_COMMON_ENV,
                },
            ),
        ],
        [
            TaskSpec(
                name="amazon_rdl_weighted_5seed",
                gpu=0,
                out_subdir="amazon/rdl_weighted_5seed",
                log_name="amazon_rdl_weighted_5seed.log",
                command=[PYTHON, str(ROOT / "scripts" / "run_10seed_supervised_rdl_weighted.py")],
                env_updates={
                    "DATASET": "amazon",
                    "N_SEEDS": "5",
                    "WEIGHTED_LOSS": "1",
                    "OUT_SUBDIR": "amazon/rdl_weighted_5seed_rankboost",
                    "SAVE_RAW_RESULTS_DIR": str(RAW_ROOT / "amazon_rdl_weighted_5seed"),
                },
            ),
            TaskSpec(
                name="hm_rdl_weighted_5seed",
                gpu=1,
                out_subdir="hm/rdl_weighted_5seed",
                log_name="hm_rdl_weighted_5seed.log",
                command=[PYTHON, str(ROOT / "scripts" / "run_10seed_supervised_rdl_weighted.py")],
                env_updates={
                    "DATASET": "hm",
                    "N_SEEDS": "5",
                    "WEIGHTED_LOSS": "1",
                    "OUT_SUBDIR": "hm/rdl_weighted_5seed_rankboost",
                    "SAVE_RAW_RESULTS_DIR": str(RAW_ROOT / "hm_rdl_weighted_5seed"),
                },
            ),
        ],
        [
            TaskSpec(
                name="amazon_rdl_unweighted_5seed",
                gpu=0,
                out_subdir="amazon/rdl_unweighted_5seed",
                log_name="amazon_rdl_unweighted_5seed.log",
                command=[PYTHON, str(ROOT / "scripts" / "run_10seed_supervised_rdl_weighted.py")],
                env_updates={
                    "DATASET": "amazon",
                    "N_SEEDS": "5",
                    "WEIGHTED_LOSS": "0",
                    "OUT_SUBDIR": "amazon/rdl_unweighted_5seed_rankboost",
                    "SAVE_RAW_RESULTS_DIR": str(RAW_ROOT / "amazon_rdl_unweighted_5seed"),
                },
            ),
            TaskSpec(
                name="hm_rdl_unweighted_5seed",
                gpu=1,
                out_subdir="hm/rdl_unweighted_5seed",
                log_name="hm_rdl_unweighted_5seed.log",
                command=[PYTHON, str(ROOT / "scripts" / "run_10seed_supervised_rdl_weighted.py")],
                env_updates={
                    "DATASET": "hm",
                    "N_SEEDS": "5",
                    "WEIGHTED_LOSS": "0",
                    "OUT_SUBDIR": "hm/rdl_unweighted_5seed_rankboost",
                    "SAVE_RAW_RESULTS_DIR": str(RAW_ROOT / "hm_rdl_unweighted_5seed"),
                },
            ),
        ],
        [
            TaskSpec(
                name="amazon_norules_5seed",
                gpu=0,
                out_subdir="amazon/gae_norules_5seed",
                log_name="amazon_norules_5seed.log",
                command=[PYTHON, str(ROOT / "scripts" / "run_10seed_gae_norules_amazon.py")],
                env_updates={
                    "N_SEEDS": "5",
                    "OUT_SUBDIR": "amazon/gae_norules_5seed_rankboost",
                    "SAVE_RAW_RESULTS_DIR": str(RAW_ROOT / "amazon_norules_5seed"),
                    **RAD_COMMON_ENV,
                },
            ),
            TaskSpec(
                name="hm_norules_5seed",
                gpu=1,
                out_subdir="hm/gae_norules_5seed",
                log_name="hm_norules_5seed.log",
                command=[PYTHON, str(ROOT / "scripts" / "run_10seed_gae_norules_hm.py")],
                env_updates={
                    "N_SEEDS": "5",
                    "OUT_SUBDIR": "hm/gae_norules_5seed_rankboost",
                    "SAVE_RAW_RESULTS_DIR": str(RAW_ROOT / "hm_norules_5seed"),
                    **RAD_COMMON_ENV,
                },
            ),
        ],
    ]


def write_manifest(phases: Iterable[list[TaskSpec]]) -> None:
    with MANIFEST_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "gpu", "log_path", "result_subdir", "raw_subdir"])
        writer.writeheader()
        for phase in phases:
            for task in phase:
                raw_subdir = task.env_updates.get("SAVE_RAW_RESULTS_DIR", str((ROOT / "results" / task.out_subdir / "raw_predictions")))
                writer.writerow(
                    {
                        "task": task.name,
                        "gpu": task.gpu,
                        "log_path": str(LOG_DIR / task.log_name),
                        "result_subdir": task.out_subdir,
                        "raw_subdir": raw_subdir,
                    }
                )


def start_task(task: TaskSpec) -> tuple[subprocess.Popen[str], object]:
    env = os.environ.copy()
    env.update(task.env_updates)
    env["CUDA_VISIBLE_DEVICES"] = str(task.gpu)
    env["PYTHONUNBUFFERED"] = "1"

    log_path = LOG_DIR / task.log_name
    log_handle = log_path.open("w")
    proc = subprocess.Popen(
        task.command,
        cwd=ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[start] {task.name} gpu={task.gpu} log={log_path}", flush=True)
    return proc, log_handle


def run_phase(phase_index: int, phase: list[TaskSpec]) -> None:
    print("=" * 88, flush=True)
    print(f"PHASE {phase_index} | tasks={[task.name for task in phase]}", flush=True)
    print("=" * 88, flush=True)
    running: list[tuple[TaskSpec, subprocess.Popen[str], object]] = []
    for task in phase:
        proc, log_handle = start_task(task)
        running.append((task, proc, log_handle))

    failures: list[str] = []
    for task, proc, log_handle in running:
        code = proc.wait()
        log_handle.close()
        print(f"[done] {task.name} exit={code}", flush=True)
        if code != 0:
            failures.append(task.name)

    if failures:
        raise RuntimeError(f"Phase {phase_index} failed for tasks: {failures}")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    phases = build_tasks()
    write_manifest(phases)

    print(f"Unified output root: {OUT_ROOT}", flush=True)
    print(f"Manifest: {MANIFEST_CSV}", flush=True)

    for idx, phase in enumerate(phases, start=1):
        run_phase(idx, phase)

    render_cmd = [PYTHON, str(ROOT / "scripts" / "render_user_churn_unified_5seed.py"), "--out-root", str(OUT_ROOT)]
    print(f"[render] {' '.join(render_cmd)}", flush=True)
    subprocess.run(render_cmd, cwd=ROOT, check=True)

    print("All unified 5-seed user-churn tasks completed.", flush=True)


if __name__ == "__main__":
    main()
