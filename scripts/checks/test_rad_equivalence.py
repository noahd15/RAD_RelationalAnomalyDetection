"""Tests for the equivalence driver and comparer: python scripts/checks/test_rad_equivalence.py"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

CHECKS = Path(__file__).resolve().parent
REPO = CHECKS.parents[1]
sys.path.insert(0, str(CHECKS))
sys.path.insert(0, str(REPO / "scripts"))

import rad_equivalence_compare as cmp  # noqa: E402
import rad_equivalence_run as drv  # noqa: E402

FAILURES = []
OLD_TAG = "table4-final-20261026b"


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


def raises(name, fn, match=""):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, match in str(e))
        if match not in str(e):
            print("     got:", repr(e))
        return
    check(name + " (did not raise)", False)


def test_driver(tmp: Path):
    old = tmp / "old"
    (old / "scripts").mkdir(parents=True)
    for ds, runner in drv.RUNNER.items():
        src = subprocess.run(["git", "show", f"{OLD_TAG}:{runner}"], cwd=REPO, capture_output=True, text=True, check=True)
        (old / runner).write_text(src.stdout)
        hp_old, ls_old = drv.runner_constants(old, ds)
        hp_new, ls_new = drv.runner_constants(REPO, ds)
        check(f"{ds}: old and new runners have the same August RAD_HP", hp_old == hp_new and ls_old == ls_new)
    hp, ls = drv.runner_constants(REPO, "hm")
    check("hm: RAD_HP", hp == {"LAMBDA_RANK": "2.0", "ALPHA_FEAT": "0.5", "ALPHA_STRUCT": "2.0"} and ls == 1.0)

    base = {"PATH": "/bin", "LAMBDA_STRUCT": "9", "RULE_TABLE_EXPECT": "x", "EPOCHS": "3", "HF_TOKEN": "t",
            "CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    e = drv.arm_env(base, "hm", 0, "bce", hp, ls, tmp / "arm", None)
    check("bce arm: LAMBDA_STRUCT is the runner's", e["LAMBDA_STRUCT"] == "1.0")
    check("caller's EPOCHS does not leak in", e["EPOCHS"] == "30")
    check("caller's RULE_TABLE_EXPECT and CUBLAS preset are cleared",
          "RULE_TABLE_EXPECT" not in e and "CUBLAS_WORKSPACE_CONFIG" not in e)
    check("label env and reported-run settings", e["ATRISK_TAU_B"] == "10.0" and e["USE_CACHED_LLM_RULES"] == "0"
          and e["HM_INCLUDE_BROAD_RULES"] == "0" and e["TARGET_ANOM_RATE"] == "0.015")
    check("other caller env kept", e["HF_TOKEN"] == "t" and e["PATH"] == "/bin")
    check("per-arm output paths", e["RULES_OUT_DIR"].endswith("arm/rules")
          and e["SAVE_RAW_RESULTS_PATH"].endswith("arm/raw_results.pt"))
    n = drv.arm_env(base, "hm", 3, "nobce", hp, ls, tmp / "arm", 2)
    check("nobce arm: LAMBDA_STRUCT 0, seed, epochs override",
          n["LAMBDA_STRUCT"] == "0.0" and n["SEED"] == "3" and n["EPOCHS"] == "2" and n["RF_RULE_MODE"] == "llm")
    r = drv.arm_env(base, "amazon", 0, "rf_seed", *drv.runner_constants(REPO, "amazon"), tmp / "arm", None)
    check("amazon rf_seed arm", r["RF_RULE_MODE"] == "rf_seed" and r["LATENT_DIM"] == "32" and r["EPOCHS"] == "60"
          and "HM_INCLUDE_BROAD_RULES" not in r and r["LAMBDA_STRUCT"] == "0.0")

    out = tmp / "exists"; out.mkdir()
    raises("driver refuses an existing --out",
           lambda: drv.main(["--code-root", str(REPO), "--dataset", "hm", "--out", str(out)]), "")


def fake_run(d: Path, code_root: str, arms=("bce",), stdout=None, raw=None, rules=b"R", mode_line=False):
    d.mkdir(parents=True)
    (d / "run_config.json").write_text(json.dumps({"dataset": "hm", "seed": 0, "arms": list(arms),
                                                   "code_root": code_root, "deterministic_mode": mode_line}))
    for arm in arms:
        a = d / arm
        (a / "rules").mkdir(parents=True)
        text = stdout if stdout is not None else (
            "Loading Database object from /x...\nDone in 0.82 seconds.\n"
            f"Rule table saved → {d.resolve()}/{arm}/rules/customer_rule_table.pt\n"
            "  RAD epoch 01/30 [warmup] | loss=1.4362\n  RAD best val auprc: 0.0308\n")
        if mode_line:
            text = "deterministic_mode: true | x\nrad_hp: {}\nrad_best_fit_metric: auprc=0.0308\n" + text
        (a / "stdout.log").write_text(text)
        torch.save(raw if raw is not None else {
            "test_scores": np.array([0.1, np.nan, 0.3]), "best_val_metric": 0.0308,
            "model_state_dict": {"w": torch.ones(2, 2)}, "hyperparams": {"path": f"{code_root}/weights"}},
            a / "raw_results.pt")
        (a / "rules" / "customer_rule_table.pt").write_bytes(rules)
        (a / "epoch_times.tsv").write_text("kind\tepoch\tunix_time\nstart\t0\t0\nSUP\t1\t10\nSUP\t2\t13\nSUP\t3\t17\n"
                                           "RAD\t1\t30\nRAD\t2\t35\nRAD\t3\t40\nend\t0\t50\n")
    return d


def test_compare(tmp: Path):
    a = fake_run(tmp / "a", "/old")
    b = fake_run(tmp / "b", "/new", mode_line=True)
    rep = cmp.compare_runs(a, b)
    check("identical after normalization (paths, Done in, refactor-only lines)", rep["identical"])
    check("NaNs in the same place compare equal", not rep["arms"]["bce"]["raw_diffs"])

    c = fake_run(tmp / "c", "/new", stdout="Done in 0.82 seconds.\n  RAD epoch 01/30 [warmup] | loss=1.4363\n")
    rep = cmp.compare_runs(a, c)
    check("one changed digit in an epoch line is a difference", not rep["identical"] and any(
        "loss=1.4363" in d for d in rep["arms"]["bce"]["stdout_diffs"]))

    raw = {"test_scores": np.array([0.1, np.nan, np.nextafter(0.3, 1)]), "best_val_metric": 0.0308,
           "model_state_dict": {"w": torch.ones(2, 2)}, "hyperparams": {"path": "/new/weights"}}
    rep = cmp.compare_runs(a, fake_run(tmp / "d", "/new", raw=raw))
    check("one ulp in one score is a difference", rep["arms"]["bce"]["raw_diffs"] == ["raw.test_scores: 1 of 3 values differ"])
    raw = dict(raw, test_scores=np.array([0.1, np.nan, 0.3]), model_state_dict={"w": torch.ones(2, 2, dtype=torch.float64)})
    rep = cmp.compare_runs(a, fake_run(tmp / "e", "/new", raw=raw))
    check("dtype change is a difference", any("float64" in d for d in rep["arms"]["bce"]["raw_diffs"]))
    raw = dict(raw, model_state_dict={"w": torch.ones(2, 2)}, best_val_metric=0.0309)
    rep = cmp.compare_runs(a, fake_run(tmp / "f", "/new", raw=raw))
    check("best_val_metric change is a difference", rep["arms"]["bce"]["raw_diffs"] == ["raw.best_val_metric: 0.0308 != 0.0309"])
    rep = cmp.compare_runs(a, fake_run(tmp / "g", "/new", rules=b"S"))
    check("rule table md5 change is a difference", not rep["identical"])
    raises("different arms refused", lambda: cmp.compare_runs(a, fake_run(tmp / "h", "/new", arms=("nobce",))),
           "differ in")

    check("same exit code 0 when identical", cmp.main(["same", str(a), str(b)]) == 0)
    check("same exit code 1 when different", cmp.main(["same", str(a), str(c)]) == 1)
    secs = cmp.epoch_seconds(a / "bce" / "epoch_times.tsv")
    check("epoch seconds leave out the first epoch", secs == {"SUP": [3.0, 4.0], "RAD": [5.0, 5.0]})


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        test_driver(Path(t))
        test_compare(Path(t))
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall passed")
    sys.exit(1 if FAILURES else 0)
