#!/usr/bin/env python3
"""Run one RAD Optuna study per seed on that seed's fit partition (H&M or Amazon).

Each study runs the RAD script itself in its tuning mode (RAD_TUNE_TRIALS > 0): the same data
prep, identity-proxy masking, labels, fit/report split, loaders and training code (`rad_core`)
as the reported runs, with the reported runs' settings (`rad_run_settings.run_env`). It trains on
the seed's train rows with that seed's LLM rule table from RULES_FROM (read-only) and scores the
seed's fit partition. The report partition is never loaded.

Env:
  DATASET          hm | amazon
  RULES_FROM       earlier run folder whose rules/ holds each seed's rule tables (final_20261028)
  OUT_SUBDIR       output folder under results/; must not exist
  N_SEEDS, START_SEED, RAD_TUNE_TRIALS (default 30), CUDA_VISIBLE_DEVICES (required)
  RAD_DETERMINISTIC 1 runs each study through scripts/deterministic_run.py (default 0)

Writes results/<OUT_SUBDIR>/seed<s>/{run.log,study.db,trials.csv,best.json} and
results/<OUT_SUBDIR>/rad_hparams_per_seed.csv.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import rad_core
import rad_per_seed
from ablation_provenance import write_run_config
from rad_run_settings import (AUGUST_BCE_LAMBDA_STRUCT, RAD_SCRIPT, RUN_PREFIX, check_mode_line,
                              deterministic_mode, rad_command, run_env)
from rad_tune import PRUNER_KWARGS, SEARCH_SPACE, TPE_SEED

ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)


def _study_rule_table(rules_root: Path, dataset: str, seed: int) -> Path:
    """The seed's LLM rule table; both LLM arms of the earlier run must have built the same one."""
    no_bce = rad_per_seed.find_rule_table(rules_root, RUN_PREFIX[dataset], seed, 0.0, "llm")
    bce = rad_per_seed.find_rule_table(rules_root, RUN_PREFIX[dataset], seed,
                                       AUGUST_BCE_LAMBDA_STRUCT[dataset], "llm")
    rad_core.check_same_rule_table(bce, no_bce)
    return no_bce


def _run_study(dataset: str, seed: int, rule_table: Path, out_dir: Path, n_trials: int,
               deterministic: bool) -> None:
    out_dir.mkdir(parents=True)
    env = os.environ.copy()
    env.update({
        "SEED": str(seed),
        **run_env(dataset, env),
        "TUNE_METRIC": "auprc",
        "RAD_TUNE_TRIALS": str(n_trials),
        "RAD_TUNE_OUT": out_dir.as_posix(),
        "RULE_TABLE_IN": rule_table.as_posix(),
    })
    print(f"[seed={seed}] RAD study, {n_trials} trials, rules {rule_table}", flush=True)
    lines = []
    with open(out_dir / "run.log", "w") as log:
        proc = subprocess.Popen(rad_command(sys.executable, RAD_SCRIPT[dataset], deterministic), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            lines.append(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"RAD study exited {rc} (dataset={dataset}, seed={seed}); log: {out_dir / 'run.log'}")
    check_mode_line("".join(lines), deterministic, f"study seed={seed}")


def main() -> None:
    dataset = os.environ["DATASET"]
    if dataset not in RAD_SCRIPT:
        raise ValueError(f"DATASET must be one of {sorted(RAD_SCRIPT)}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES.")
    rules_root = Path(os.environ["RULES_FROM"]) / "rules"
    out_root = Path("results") / os.environ["OUT_SUBDIR"]  # repo-relative; recorded in the hp file
    n_trials = int(os.environ.get("RAD_TUNE_TRIALS", "30"))
    start, n = int(os.environ.get("START_SEED", "0")), int(os.environ.get("N_SEEDS", "10"))
    seeds = list(range(start, start + n))
    deterministic = deterministic_mode(os.environ)

    # Find and check every seed's rule table before the first study starts.
    rule_tables = {s: _study_rule_table(rules_root, dataset, s) for s in seeds}
    for s in seeds:
        if (out_root / f"seed{s}").exists():
            raise RuntimeError(f"{out_root / f'seed{s}'} exists; refusing to overwrite it.")
    out_root.mkdir(parents=True, exist_ok=True)
    write_run_config(out_root, seeds, extra={
        "dataset": dataset, "rad_script": RAD_SCRIPT[dataset], "rules_from": rules_root.as_posix(),
        "rule_tables": {str(s): {"path": p.as_posix(), "md5": rad_per_seed.file_md5(p)}
                        for s, p in rule_tables.items()},
        "n_trials": n_trials, "search_space": SEARCH_SPACE, "tpe_seed": TPE_SEED, "pruner": PRUNER_KWARGS,
        "tune_metric": "auprc", "run_env": run_env(dataset, os.environ), "deterministic_mode": deterministic,
    })

    for s in seeds:
        _run_study(dataset, s, rule_tables[s], out_root / f"seed{s}", n_trials, deterministic)

    rad_per_seed.combine(out_root / "rad_hparams_per_seed.csv", [out_root])


if __name__ == "__main__":
    main()
