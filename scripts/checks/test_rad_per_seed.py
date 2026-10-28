"""Synthetic tests (CPU, no data) for the per-seed RAD tuning and tuned-rerun code:
python scripts/checks/test_rad_per_seed.py

Covers rad_core.inputs_fingerprint / check_same_rule_table, rad_tune.run_study (training stubbed),
rad_per_seed, rad_run_settings, and both 10-seed runners in tuned-rerun mode with the RAD script
replaced by canned output.
"""
import importlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import rad_core  # noqa: E402
import rad_per_seed  # noqa: E402
import rad_tune  # noqa: E402
from rad_run_settings import RUN_PREFIX, run_env  # noqa: E402

FAILURES = []


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


def frame(rows):
    return pd.DataFrame(rows, columns=["customer_id", "timestamp", "churn"]).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"]))


# ── rad_core helpers ────────────────────────────────────────────────────────
def test_rad_core(tmp):
    a = frame([(1, "2020-01-01", 0), (2, "2020-01-02", 1), (3, "2020-01-01", 0)])
    check("fingerprint ignores row order", rad_core.inputs_fingerprint(a) == rad_core.inputs_fingerprint(a.iloc[::-1]))
    b = a.copy(); b.loc[1, "churn"] = 0
    check("fingerprint sees a label change", rad_core.inputs_fingerprint(a) != rad_core.inputs_fingerprint(b))
    check("fingerprint starts with the row count", rad_core.inputs_fingerprint(a).startswith("3:"))

    t = torch.zeros(5, 2); t[1, 0] = 1
    torch.save({"rule_table": t, "rule_cols": ["r0", "r1"], "n_rules": 2}, tmp / "a.pt")
    torch.save({"rule_table": t.clone(), "rule_cols": ["r0", "r1"], "n_rules": 2}, tmp / "b.pt")
    t2 = t.clone(); t2[3, 1] = 1
    torch.save({"rule_table": t2, "rule_cols": ["r0", "r1"], "n_rules": 2}, tmp / "c.pt")
    torch.save({"rule_table": t, "rule_cols": ["r0", "rX"], "n_rules": 2}, tmp / "d.pt")
    rad_core.check_same_rule_table(tmp / "a.pt", tmp / "b.pt")
    check("same rule table passes", True)
    raises("rule table values differ", lambda: rad_core.check_same_rule_table(tmp / "a.pt", tmp / "c.pt"), "values differ on 1 rows")
    raises("rule columns differ", lambda: rad_core.check_same_rule_table(tmp / "a.pt", tmp / "d.pt"), "rule_cols")


# ── rad_tune.run_study with training stubbed ───────────────────────────────
def test_run_study(tmp):
    calls = []

    def fake_fit(model, opt, hp, *a, on_epoch=None, **k):
        # Value depends on the trial's hp, so TPE has something to rank; reports 4 epochs.
        v = 0.01 * hp["lambda_rank"] + 0.001 * hp["alpha_feat"]
        calls.append(dict(hp))
        for ep in range(1, 5):
            on_epoch(ep, v * ep / 4)
        return v, {"w": torch.zeros(1)}, {}

    orig = (rad_tune.rad_core.build_model, rad_tune.rad_core.make_optimizer, rad_tune.rad_core.fit)
    rad_tune.rad_core.build_model = lambda *a, **k: torch.nn.Linear(1, 1)
    rad_tune.rad_core.make_optimizer = lambda *a, **k: None
    rad_tune.rad_core.fit = fake_fit
    try:
        data = {"customer": types.SimpleNamespace(num_nodes=5)}
        torch.save({"rule_table": torch.ones(5, 3), "rule_cols": ["a", "b", "c"], "n_rules": 3}, tmp / "rt.pt")
        train_df = frame([(1, "2020-01-01", 0), (2, "2020-01-01", 1)])
        fit_df = frame([(3, "2020-02-01", 0), (4, "2020-02-01", 1)])
        base_hp = {"epochs": 30, "patience": 5, "lambda_struct": 0.5, "lambda_rank": 1.0,
                   "alpha_feat": 1.0, "alpha_struct": 1.0, "latent_dim": 48}
        kw = dict(dataset="hm", seed=3, n_trials=6, rule_table_path=tmp / "rt.pt", data=data,
                  col_stats_dict={}, loader_train=None, loader_val=None, val_labels=np.array([0, 1]),
                  train_df=train_df, fit_df=fit_df, entity_table="customer", base_hp=base_hp,
                  device="cpu", tune_metric="auprc")
        s = rad_tune.run_study(out_dir=tmp / "study", **kw)
        best = json.loads((tmp / "study" / "best.json").read_text())
        check("study ran every trial", best["n_trials"] == 6 and len(calls) == 6)
        check("trials keep the base settings", all(c["epochs"] == 30 and c["latent_dim"] == 48 for c in calls))
        check("trials draw from the search space",
              all(c[k] in rad_tune.SEARCH_SPACE[k] for c in calls for k in rad_tune.SEARCH_SPACE))
        check("best.json records the rule md5", best["rule_table_md5"] == rad_per_seed.file_md5(tmp / "rt.pt"))
        check("best.json records the input fps", best["fit_fp"] == rad_core.inputs_fingerprint(fit_df))
        check("best value is the best trial's", abs(best["best_value"] - max(
            0.01 * c["lambda_rank"] + 0.001 * c["alpha_feat"] for c in calls if True)) < 1e-12 or best["n_pruned"] > 0)
        check("base_hp unchanged in best.json", best["base_hp"] == base_hp)
        check("trials.csv written", len(pd.read_csv(tmp / "study" / "trials.csv")) == 6)
        check("returned summary == best.json", s == best)
        raises("refuses an existing study", lambda: rad_tune.run_study(out_dir=tmp / "study", **kw), "already exists")
        kw_bad = dict(kw, data={"customer": types.SimpleNamespace(num_nodes=4)})
        raises("rule table rows must match the graph", lambda: rad_tune.run_study(out_dir=tmp / "s2", **kw_bad), "has 5 rows")
        torch.save({"rule_table": torch.zeros(5, 1), "rule_cols": [], "n_rules": 0}, tmp / "rt0.pt")
        raises("refuses a rule table with no rules",
               lambda: rad_tune.run_study(out_dir=tmp / "s3", **dict(kw, rule_table_path=tmp / "rt0.pt")), "no rules")
    finally:
        rad_tune.rad_core.build_model, rad_tune.rad_core.make_optimizer, rad_tune.rad_core.fit = orig


# ── rad_run_settings ────────────────────────────────────────────────────────
def test_run_env():
    hm = run_env("hm", {})
    check("hm run settings", hm == {"LATENT_DIM": "48", "EPOCHS": "30", "PATIENCE": "5", "WARMUP_EPOCHS": "5",
                                    "WARMUP_RANK_SCALE": "0.25", "WARMUP_STRUCT_SCALE": "1.0"})
    amz = run_env("amazon", {"LATENT_DIM": "64", "EPOCHS": "7"})
    check("amazon latent stays 32, caller can set EPOCHS", amz["LATENT_DIM"] == "32" and amz["EPOCHS"] == "7")
    check("amazon run settings", run_env("amazon", {}) == {"EPOCHS": "60", "PATIENCE": "10", "WARMUP_EPOCHS": "15",
                                                           "WARMUP_RANK_SCALE": "0.25", "WARMUP_STRUCT_SCALE": "1.0",
                                                           "LATENT_DIM": "32"})


# ── Fixtures for the runners: a reused run folder, studies, an hp file ─────
BASE_HP = {"epochs": 30, "patience": 5, "warmup_epochs": 5, "lambda_struct": 0.5, "lambda_rank": 1.0,
           "alpha_feat": 1.0, "alpha_struct": 1.0, "latent_dim": 48}
SEEDS = (0, 1)
FP = {0: "1234567890ab", 1: "710b11097e65"}  # seed 0 is all digits on purpose
N = {0: (5694, 85), 1: (5707, 86)}
MLP = {0: (0.0302, 0.6644, 0.0353, 2.0213), 1: (0.0269, 0.6592, 0.0349, 1.7839)}  # auprc, auroc, p@k, lift
TUNED = {0: dict(lambda_struct=2.0, lambda_rank=4.0, alpha_feat=0.5, alpha_struct=2.0),
         1: dict(lambda_struct=0.1, lambda_rank=0.25, alpha_feat=2.0, alpha_struct=0.5)}
LABEL_ENV = {"TARGET_ANOM_RATE": "0.015", "ATRISK_TAU_A": "0.05", "ATRISK_TAU_B": "10.0",
             "ATRISK_TAU_ALPHA": "0.02", "ATRISK_TAU_MULTIPLIER": "1.0"}


def make_fixture(tmp, ds, bce_ls):
    reuse = tmp / f"{ds}_final"
    (reuse / "rules").mkdir(parents=True)
    rows = []
    for s in SEEDS:
        for m, v in (("Isolation Forest", 0.41), ("DRL", 0.49), ("Supervised RDL (baselines-script cross-check, not reported)", 0.69)):
            rows.append([m, v, 0.012, 0.0, 0.8, *N[s], FP[s], s, "baselines_script"])
        rows.append(["RAD (with BCE)", 0.58, 0.045, 0.035, 3.0, *N[s], FP[s], s, "rad_vs_script"])
        rows.append(["RAD (no BCE)", 0.66, 0.032, 0.047, 2.2, *N[s], FP[s], s, "rad_vs_script"])
        a, r, p, l = MLP[s]
        rows.append(["Supervised MLP", r, a, p, l, *N[s], FP[s], s, "rad_vs_script"])
        for ls, mode in ((bce_ls, "llm"), (0.0, "llm"), (0.0, "rf_seed")):
            d = reuse / "rules" / f"{RUN_PREFIX[ds]}_seed{s}_ls{ls}_{mode}_{'ab12cd34' if mode == 'llm' else 'ffff0000'}"
            d.mkdir()
            torch.save({"rule_table": torch.full((4, 2), float(s + (mode == "rf_seed"))), "rule_cols": ["x", "y"],
                        "n_rules": 2}, d / "customer_rule_table.pt")
    pd.DataFrame(rows, columns=["model", "auroc", "auprc", "p@k", "lift", "n_eval", "n_pos", "eval_fp", "seed",
                                "source"]).to_csv(reuse / "per_seed_metrics.csv", index=False)
    (reuse / "run_config.json").write_text(json.dumps({"label_env": {**{k: None for k in ("TRAIN_EQ_RATE",)}, **LABEL_ENV}}))

    studies = tmp / f"{ds}_studies"
    for s in SEEDS:
        sd = studies / f"seed{s}"; sd.mkdir(parents=True)
        rt = rad_per_seed.find_rule_table(reuse / "rules", RUN_PREFIX[ds], s, 0.0, "llm")
        (sd / "best.json").write_text(json.dumps({
            "dataset": ds, "seed": s, "tune_metric": "auprc", "best_trial": 7, "best_value": 0.04,
            "best_params": TUNED[s], "n_trials": 30, "n_complete": 20, "n_pruned": 10,
            "rule_table_md5": rad_per_seed.file_md5(rt), "n_rules": 2,
            "train_fp": f"100:train{s}", "fit_fp": f"{N[s][0]}:fit{s}", "n_fit": 900, "n_fit_pos": 14,
            "base_hp": BASE_HP}))
    hp_file = tmp / f"{ds}_hparams.csv"
    rad_per_seed.combine(hp_file, [studies])
    return reuse, hp_file


def rad_stdout(seed, env, *, fit_fp=None, hp_override=None, n_eval=None, mlp=None, eval_fp=None):
    hp = dict(BASE_HP, lambda_struct=float(env["LAMBDA_STRUCT"]), lambda_rank=float(env["LAMBDA_RANK"]),
              alpha_feat=float(env["ALPHA_FEAT"]), alpha_struct=float(env["ALPHA_STRUCT"]), **(hp_override or {}))
    a, r, p, l = mlp or MLP[seed]
    n, npos = N[seed] if n_eval is None else n_eval
    rad = 0.05 + 0.001 * float(env["LAMBDA_STRUCT"]) + (0.002 if env["RF_RULE_MODE"] == "rf_seed" else 0)
    return (f"rad_hp: {json.dumps(hp, sort_keys=True)}\n"
            f"rad_inputs_fp: train=100:train{seed} fit={fit_fp or f'{N[seed][0]}:fit{seed}'}\n"
            "======\nFINAL COMPARISON (Val Set — At-Risk Filtered Labels)\n======\n"
            f"  Val:  {n:,} users  |  {npos} anomalies  (0.0149 rate)\n"
            f"  eval_fingerprint: {eval_fp or FP[seed]}\n\n"
            "Model                              AUPRC     AUROC       P@K      Lift\n"
            "Supervised RDL GNN                0.0281    0.6599    0.0588    1.8853  (graph only)\n"
            "RF (supervised detector)          0.0320    0.6847    0.0353    2.1407  (x)\n"
            f"MLP (supervised detector)         {a:.4f}    {r:.4f}    {p:.4f}    {l:.4f}  (x)\n"
            f"RAD (rule-augmented)              {rad:.4f}    0.6537    0.0588    2.0345  (graph)\n"
            "======\nFINAL COMPARISON (Val Set — Original Unfiltered Labels)\n")


def run_runner(module, tmp, case, reuse, hp_file, fake=None, extra_env=None):
    """Import the runner fresh with the rerun env, replace subprocess.run, run main()."""
    out = tmp / f"out_{module}_{case}"
    env = {"OUT_SUBDIR": out.as_posix(), "RAD_HPARAMS_FILE": hp_file.as_posix(),
           "REUSE_BASELINES_FROM": reuse.as_posix(), "N_SEEDS": str(len(SEEDS)), "START_SEED": "0",
           "RF_SEED_VARIANT": "1", "CUDA_VISIBLE_DEVICES": "0", **LABEL_ENV, **(extra_env or {})}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    calls = []
    try:
        sys.modules.pop(module, None)
        mod = importlib.import_module(module)

        def fake_run(cmd, env=None, **kw):
            if "baselines" in str(cmd[-1]):
                raise AssertionError("baselines script must not run in rerun mode")
            calls.append(dict(env))
            seed = int(env["SEED"])
            stdout = (fake or rad_stdout)(seed, env)
            return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        mod.subprocess = types.SimpleNamespace(run=fake_run)
        mod.main()
        return out, calls
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_runner(tmp, module, ds, bce_ls):
    reuse, hp_file = make_fixture(tmp / ds, ds, bce_ls)
    tmp = tmp / ds

    hp = rad_per_seed.load_hparams(hp_file, ds, SEEDS)
    check(f"{ds}: hp file keeps all-digit hashes as text", isinstance(hp[0]["train_fp"], str))
    raises(f"{ds}: hp file missing a seed", lambda: rad_per_seed.load_hparams(hp_file, ds, (0, 1, 2)), "seed 2")

    out, calls = run_runner(module, tmp, "ok", reuse, hp_file)
    df = pd.read_csv(out / "per_seed_metrics.csv", dtype={"eval_fp": str})
    final = pd.read_csv(reuse / "per_seed_metrics.csv", dtype={"eval_fp": str})
    check(f"{ds}: three RAD arms per seed, no baselines run", len(calls) == 3 * len(SEEDS))
    for s in SEEDS:
        arms = [c for c in calls if c["SEED"] == str(s)]
        bce, nobce, rf = arms
        t = TUNED[s]
        check(f"{ds} seed {s}: with-BCE arm uses the tuned lambda_struct",
              bce["LAMBDA_STRUCT"] == str(t["lambda_struct"]) and bce["RF_RULE_MODE"] == "llm")
        check(f"{ds} seed {s}: no-BCE arms use lambda_struct 0",
              nobce["LAMBDA_STRUCT"] == "0.0" and rf["LAMBDA_STRUCT"] == "0.0" and rf["RF_RULE_MODE"] == "rf_seed")
        check(f"{ds} seed {s}: every arm uses the tuned rank/alpha values",
              all(a["LAMBDA_RANK"] == str(t["lambda_rank"]) and a["ALPHA_FEAT"] == str(t["alpha_feat"])
                  and a["ALPHA_STRUCT"] == str(t["alpha_struct"]) for a in arms))
        check(f"{ds} seed {s}: arms use the shared run settings",
              all(all(a[k] == v for k, v in run_env(ds, {}).items()) for a in arms))
        want = {f"ls{bce_ls}_llm": bce, "ls0.0_llm": nobce, "ls0.0_rf_seed": rf}
        check(f"{ds} seed {s}: each arm expects the reused run's rule table for that arm",
              all(f"_seed{s}_{k}_" in a["RULE_TABLE_EXPECT"] for k, a in want.items()))
        got = df[df.seed == s].set_index("model")
        old = final[final.seed == s].set_index("model")
        base = old[old.source == "baselines_script"].index
        check(f"{ds} seed {s}: baseline rows reused unchanged",
              got.loc[base, ["auroc", "auprc", "eval_fp"]].equals(old.loc[base, ["auroc", "auprc", "eval_fp"]]))
        check(f"{ds} seed {s}: Supervised MLP row is the reused one",
              got.loc["Supervised MLP", ["auroc", "auprc", "p@k", "lift"]].tolist()
              == old.loc["Supervised MLP", ["auroc", "auprc", "p@k", "lift"]].tolist())
        check(f"{ds} seed {s}: RAD rows come from the rerun",
              abs(got.loc["RAD (with BCE)", "auprc"] - round(0.05 + 0.001 * t["lambda_struct"], 4)) < 1e-9
              and abs(got.loc["RAD (no BCE)", "auprc"] - 0.05) < 1e-9
              and abs(got.loc["RAD (RF-sampled rules, no BCE)", "auprc"] - 0.052) < 1e-9)
        check(f"{ds} seed {s}: all rows on one eval set", got["eval_fp"].nunique() == 1 and got["eval_fp"].iloc[0] == FP[s])
    cfg = json.loads((out / "run_config.json").read_text())
    check(f"{ds}: run_config records the hp file, its md5 and per-seed values",
          cfg["rad_hyperparameters_file_md5"] == rad_per_seed.file_md5(hp_file)
          and cfg["rad_hyperparameters_per_seed"]["0"] == TUNED[0] and "rad_hyperparameters" not in cfg)

    bad_mlp = lambda s, e: rad_stdout(s, e, mlp=(0.0303, *MLP[s][1:]))
    raises(f"{ds}: rerun MLP differs", lambda: run_runner(module, tmp, "mlp", reuse, hp_file, bad_mlp), "Supervised MLP differs")
    bad_fp = lambda s, e: rad_stdout(s, e, fit_fp="1:other")
    raises(f"{ds}: RAD fit rows differ from the study", lambda: run_runner(module, tmp, "fp", reuse, hp_file, bad_fp), "differ from the study")
    bad_hp = lambda s, e: rad_stdout(s, e, hp_override={"epochs": 10})
    raises(f"{ds}: RAD settings differ from the study", lambda: run_runner(module, tmp, "hp", reuse, hp_file, bad_hp), "epochs")
    bad_eval = lambda s, e: rad_stdout(s, e, eval_fp="deadbeef0000" if e["RF_RULE_MODE"] == "rf_seed" else None)
    raises(f"{ds}: RAD arm on another eval set", lambda: run_runner(module, tmp, "eval", reuse, hp_file, bad_eval), "different sets")
    raises(f"{ds}: label settings differ from the reused run",
           lambda: run_runner(module, tmp, "label", reuse, hp_file, extra_env={"ATRISK_TAU_B": "6.0"}), "label_env")

    # Study used another rule table than the reused run's no-BCE LLM arm.
    rt = rad_per_seed.find_rule_table(reuse / "rules", RUN_PREFIX[ds], 1, 0.0, "llm")
    torch.save({"rule_table": torch.zeros(4, 2), "rule_cols": ["x", "y"], "n_rules": 2}, rt)
    raises(f"{ds}: study rule table differs", lambda: run_runner(module, tmp, "rules", reuse, hp_file), "not the rule table the study used")

    raises(f"{ds}: hp file without the reuse folder",
           lambda: run_runner(module, tmp, "half", reuse, hp_file, extra_env={"REUSE_BASELINES_FROM": ""}), "together")


def test_find_rule_table(tmp):
    r = tmp / "rules"; r.mkdir()
    for name in ("hm10seed_seed1_ls0.0_llm_0123abcd", "hm10seed_seed10_ls0.0_llm_0123abce",
                 "hm10seed_seed1_ls0.01_llm_0123abcf", "hm10seed_seed2_ls0.0_llm_aaaa0000",
                 "hm10seed_seed2_ls0.0_llm_bbbb0000"):
        (r / name).mkdir(); (r / name / "customer_rule_table.pt").write_bytes(b"x")
    p = rad_per_seed.find_rule_table(r, "hm10seed", 1, 0.0, "llm")
    check("find_rule_table: exact seed and ls only", p.parent.name == "hm10seed_seed1_ls0.0_llm_0123abcd")
    raises("find_rule_table: two folders for one arm", lambda: rad_per_seed.find_rule_table(r, "hm10seed", 2, 0.0, "llm"), "found [")
    raises("find_rule_table: none", lambda: rad_per_seed.find_rule_table(r, "hm10seed", 3, 0.0, "llm"), "found []")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for sub in ("core", "tune", "rules", "runners"):
            (d / sub).mkdir()
        test_rad_core(d / "core")
        test_run_study(d / "tune")
        test_run_env()
        test_find_rule_table(d / "rules")
        test_runner(d / "runners", "run_full_ablation_hm_10seed", "hm", 1.0)
        test_runner(d / "runners", "run_full_ablation_10seed", "amazon", 0.1)
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall passed")
    sys.exit(1 if FAILURES else 0)
