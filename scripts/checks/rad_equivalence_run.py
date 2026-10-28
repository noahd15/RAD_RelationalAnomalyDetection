#!/usr/bin/env python3
"""Run RAD arms of one seed the way the 10-seed runner does, for the refactor equivalence check.

    CUDA_VISIBLE_DEVICES=0 python scripts/checks/rad_equivalence_run.py --code-root DIR --dataset hm \\
        --seed 0 --out OUT [--deterministic] [--arms bce,nobce,rf_seed] [--epochs N] \\
        [--expect-rules-from results/hm/final_20261028]

--code-root is the checkout whose RAD script runs (the old tag's worktree or the refactored one);
the script runs with that checkout as its working directory, as the runner does. The arm
hyperparameters (RAD_HP, RAD_BCE_LAMBDA_STRUCT) are read from that checkout's 10-seed runner, and
the other settings are the runner's defaults plus the reported-run settings of the launch script
(docs/table4_reporting_plan.md). --deterministic runs the script through this checkout's
scripts/deterministic_run.py, whichever --code-root is. --epochs overrides EPOCHS (timing runs only).

Per arm, OUT/<arm>/ gets stdout.log (verbatim), stderr.log, epoch_times.tsv (wall-clock arrival of
each SUP/RAD epoch line), raw_results.pt (SAVE_RAW_RESULTS_PATH) and rules/ (RULES_OUT_DIR).
With --expect-rules-from, each arm's rule table must equal that run's table for the same seed and
arm. OUT/run_config.json records the settings. Exits 1 if any arm fails. Compare two runs with
scripts/checks/rad_equivalence_compare.py.
"""

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import rad_per_seed  # noqa: E402
from rad_run_settings import (AUGUST_BCE_LAMBDA_STRUCT, DETERMINISTIC_WRAPPER, RAD_SCRIPT, RUN_PREFIX, rad_command,  # noqa: E402
                              run_env)

RUNNER = {"hm": "scripts/run_full_ablation_hm_10seed.py", "amazon": "scripts/run_full_ablation_10seed.py"}
ARMS = {"bce": "llm", "nobce": "llm", "rf_seed": "rf_seed"}  # arm -> RF_RULE_MODE
# Reported-run settings (launch script run_userchurn.sh and the reporting plan).
LABEL_ENV = {"ATRISK_TAU_A": "0.05", "ATRISK_TAU_B": "10.0", "ATRISK_TAU_ALPHA": "0.02",
             "ATRISK_TAU_MULTIPLIER": "1.0", "TARGET_ANOM_RATE": "0.015"}
FIXED_ENV = {"USE_CACHED_LLM_RULES": "0", "RF_SEED_VARIANT": "1", "PYTHONUNBUFFERED": "1", **LABEL_ENV}
# Variables some code path reads that must not leak in from the caller.
CLEARED_ENV = ("RULE_TABLE_EXPECT", "RULE_TABLE_IN", "RAD_TUNE_TRIALS", "RAD_TUNE_OUT", "RAD_DETERMINISTIC",
               "CUBLAS_WORKSPACE_CONFIG", "SAVE_RAW_RESULTS_PATH", "RULES_OUT_DIR", "LAMBDA_STRUCT", "LAMBDA_RANK",
               "ALPHA_FEAT", "ALPHA_STRUCT", "EPOCHS", "PATIENCE", "WARMUP_EPOCHS", "WARMUP_RANK_SCALE",
               "WARMUP_STRUCT_SCALE", "LATENT_DIM", "TUNE_METRIC", "RF_RULE_MODE", "SEED", "VAL_FIT_FRAC",
               "HM_INCLUDE_BROAD_RULES")
EPOCH_RE = re.compile(rb"^\s*(SUP|RAD) epoch (\d+)")


def runner_constants(code_root: Path, dataset: str) -> tuple[dict, float]:
    """RAD_HP and RAD_BCE_LAMBDA_STRUCT as assigned in that checkout's runner."""
    tree = ast.parse((code_root / RUNNER[dataset]).read_text())
    found = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             for t in n.targets if isinstance(t, ast.Name) and t.id in ("RAD_HP", "RAD_BCE_LAMBDA_STRUCT")}
    if set(found) != {"RAD_HP", "RAD_BCE_LAMBDA_STRUCT"}:
        raise RuntimeError(f"{code_root / RUNNER[dataset]}: found only {sorted(found)}")
    if found["RAD_BCE_LAMBDA_STRUCT"] != AUGUST_BCE_LAMBDA_STRUCT[dataset]:
        raise RuntimeError(f"RAD_BCE_LAMBDA_STRUCT {found['RAD_BCE_LAMBDA_STRUCT']} != {AUGUST_BCE_LAMBDA_STRUCT[dataset]}")
    return found["RAD_HP"], float(found["RAD_BCE_LAMBDA_STRUCT"])


def arm_env(base: dict, dataset: str, seed: int, arm: str, rad_hp: dict, bce_ls: float, arm_dir: Path,
            epochs: int | None) -> dict:
    env = {k: v for k, v in base.items() if k not in CLEARED_ENV}
    env.update(FIXED_ENV)
    if dataset == "hm":
        env["HM_INCLUDE_BROAD_RULES"] = "0"
    env.update(run_env(dataset, {}))
    ls = bce_ls if arm == "bce" else 0.0
    env.update({
        "SEED": str(seed), "RUN_TAG": f"equiv_{dataset}_seed{seed}_{arm}", "LAMBDA_STRUCT": str(ls), **rad_hp,
        "TUNE_METRIC": "auprc", "RF_RULE_MODE": ARMS[arm], "RULES_OUT_DIR": (arm_dir / "rules").as_posix(),
        "SAVE_RAW_RESULTS_PATH": (arm_dir / "raw_results.pt").as_posix(),
    })
    if epochs is not None:
        env["EPOCHS"] = str(epochs)
    return env


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_arm(cmd: list[str], env: dict, cwd: Path, arm_dir: Path) -> dict:
    arm_dir.mkdir(parents=True)
    t0 = time.time()
    with open(arm_dir / "stdout.log", "wb") as out, open(arm_dir / "stderr.log", "wb") as err, \
            open(arm_dir / "epoch_times.tsv", "w") as times:
        times.write("kind\tepoch\tunix_time\n")
        times.write(f"start\t0\t{t0:.3f}\n")
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=err)
        for line in proc.stdout:
            out.write(line)
            out.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            m = EPOCH_RE.match(line)
            if m:
                times.write(f"{m.group(1).decode()}\t{int(m.group(2))}\t{time.time():.3f}\n")
                times.flush()
        rc = proc.wait()
        times.write(f"end\t0\t{time.time():.3f}\n")
    return {"rc": rc, "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-root", required=True, type=Path)
    ap.add_argument("--dataset", required=True, choices=sorted(RAD_SCRIPT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--arms", default="bce,nobce,rf_seed")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--expect-rules-from", type=Path)
    a = ap.parse_args(argv)

    code_root, out = a.code_root.resolve(), a.out.resolve()
    arms = a.arms.split(",")
    if set(arms) - set(ARMS):
        ap.error(f"--arms must be from {sorted(ARMS)}")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"\d+", gpu):
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES to one GPU index (got {gpu!r}).")
    if out.exists():
        raise RuntimeError(f"{out} exists; refusing to overwrite it.")
    rad_hp, bce_ls = runner_constants(code_root, a.dataset)
    # The wrapper is always this checkout's (the old tag has none), so old and new code run under the same one.
    cmd = rad_command(sys.executable, code_root / RAD_SCRIPT[a.dataset], a.deterministic)

    expect = {}
    if a.expect_rules_from:
        for arm in arms:
            ls = bce_ls if arm == "bce" else 0.0
            expect[arm] = rad_per_seed.find_rule_table(a.expect_rules_from / "rules", RUN_PREFIX[a.dataset],
                                                       a.seed, ls, ARMS[arm])

    out.mkdir(parents=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=code_root,
                           capture_output=True, text=True).stdout.strip()
    envs = {arm: arm_env(dict(os.environ), a.dataset, a.seed, arm, rad_hp, bce_ls, out / arm, a.epochs) for arm in arms}
    config = {
        "code_root": code_root.as_posix(), "code_git_head": git, "code_tracked_changes": dirty.splitlines(),
        "driver": Path(__file__).resolve().as_posix(), "dataset": a.dataset, "seed": a.seed, "arms": arms,
        "deterministic_mode": a.deterministic, "wrapper": str(DETERMINISTIC_WRAPPER) if a.deterministic else None,
        "cmd": cmd, "epochs_override": a.epochs, "rad_hp": rad_hp, "bce_lambda_struct": bce_ls,
        "cuda_visible_devices": gpu, "host": platform.node(), "python": sys.executable,
        "expect_rules": {k: v.as_posix() for k, v in expect.items()},
        "arm_env": {arm: {k: e[k] for k in sorted(e) if k not in os.environ or e[k] != os.environ[k]
                          or k in FIXED_ENV} for arm, e in envs.items()},
    }
    (out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    failed = []
    summary = {}
    for arm in arms:
        print(f"[equiv] {a.dataset} seed={a.seed} arm={arm} deterministic={a.deterministic} code={code_root}", flush=True)
        res = run_arm(cmd, envs[arm], code_root, out / arm)
        if res["rc"] != 0:
            failed.append(f"{arm}: exit {res['rc']} (see {out / arm / 'stderr.log'})")
        elif arm in expect:
            got = out / arm / "rules" / "customer_rule_table.pt"
            res["rule_table_md5"] = md5(got) if got.exists() else None
            res["expected_rule_table_md5"] = md5(expect[arm])
            if res["rule_table_md5"] != res["expected_rule_table_md5"]:
                failed.append(f"{arm}: rule table {res['rule_table_md5']} != {expect[arm]} {res['expected_rule_table_md5']}")
        summary[arm] = res
        print(f"[equiv] arm={arm} {res}", flush=True)
        if res["rc"] != 0:
            break
    (out / "summary.json").write_text(json.dumps({"arms": summary, "failed": failed}, indent=2) + "\n")
    print(f"[equiv] {'FAILED: ' + '; '.join(failed) if failed else 'all arms ok'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
