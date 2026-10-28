"""Synthetic tests for scripts/table4_paired_bootstrap.py: python scripts/checks/test_table4_paired_bootstrap.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import table4_paired_bootstrap as tb  # noqa: E402

FAILURES = []
SEEDS = range(10)


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


def fp(s):
    return f"{s:012d}" if s == 0 else f"fp{s:010x}"  # seed 0: all digits


# RAD - baseline per seed: RAD below IF, above DRL; the AE difference straddles 0.
OFFSETS = {"Isolation Forest": -0.10, "DRL": 0.10, "Deep SVDD": -0.05, "AE": 0.0, "AE+Rank": -0.02,
           "Supervised MLP": 0.03}
NOISE = np.linspace(-0.01, 0.01, 10)


def user_churn_fixture(d: Path, eval_fp_override=None, drop_seed=None):
    rows = []
    for s in SEEDS:
        n_eval, n_pos = 5694 + s, 85 + s % 2
        common = dict(n_eval=n_eval, n_pos=n_pos, eval_fp=fp(s), seed=s, lift=1.0)
        rows.append(dict(model="RAD (no BCE)", auroc=0.6, auprc=0.04, **{"p@k": 0.05}, source="rad", **common))
        rows.append(dict(model="RAD (with BCE)", auroc=0.9, auprc=0.9, **{"p@k": 0.9}, source="rad", **common))
        rows.append(dict(model="Supervised RDL (baselines-script cross-check, not reported)", auroc=0.9, auprc=0.9,
                         **{"p@k": 0.9}, source="b", **common))
        for m, off in OFFSETS.items():
            if m == "DRL" and s == drop_seed:
                continue
            noise = NOISE[s] if m == "AE" else 0.0
            c = dict(common, eval_fp=eval_fp_override if (m == "DRL" and s == 3 and eval_fp_override) else fp(s))
            rows.append(dict(model=m, auroc=0.6 - off + noise, auprc=0.04 - off / 10, **{"p@k": 0.05}, source="b", **c))
    (d / "abl").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "abl" / "per_seed_metrics.csv", index=False)
    for w, sub in ((True, "w"), (False, "u")):
        (d / sub).mkdir()
        pd.DataFrame([dict(dataset="hm", seed=s, weighted_loss=w, auroc=0.55, auprc=0.03, **{"p@k": 0.05},
                           n_eval=5694 + s, n_pos=85 + s % 2, eval_fp=fp(s)) for s in SEEDS]).to_csv(
            d / sub / "per_seed.csv", index=False)
    return d / "abl", d / "w", d / "u"


def test_user_churn(tmp: Path):
    res = tb.compare(tb.load_user_churn(*user_churn_fixture(tmp / "ok")), 10).set_index(["baseline", "metric"])
    check("all baselines x metrics", len(res) == len(tb.BASELINES) * len(tb.METRICS))
    check("reported RDL rows are the RDL runner's, not the cross-check",
          abs(res.loc[("Supervised RDL (weighted)", "auroc"), "baseline_mean"] - 0.55) < 1e-12)
    check("headline RAD row is RAD (no BCE)", abs(res.loc[("DRL", "auroc"), "rad_mean"] - 0.6) < 1e-12)
    # baseline = 0.6 - off and RAD = 0.6, so d = off.
    check("d = RAD - baseline", abs(res.loc[("Isolation Forest", "auroc"), "mean_diff"] - (-0.10)) < 1e-12)
    check("CI below 0 -> baseline better", res.loc[("Isolation Forest", "auroc"), "verdict"] == "baseline better")
    check("CI above 0 -> RAD better", res.loc[("DRL", "auroc"), "verdict"] == "RAD better")
    check("CI contains 0 -> no difference", res.loc[("AE", "auroc"), "verdict"] == "no difference")
    check("constant difference gives a zero-width CI",
          res.loc[("DRL", "auroc"), "ci_low"] == res.loc[("DRL", "auroc"), "ci_high"])
    check("p@k identical -> no difference", res.loc[("DRL", "p@k"), "verdict"] == "no difference")

    # CI matches a direct computation with the documented rng.
    d = -NOISE  # AE: baseline = 0.6 + noise, RAD = 0.6
    rng = np.random.default_rng(20261028)
    want = np.percentile(d[rng.integers(0, 10, size=(10_000, 10))].mean(axis=1), [2.5, 97.5])
    got = res.loc[("AE", "auroc"), ["ci_low", "ci_high"]].to_numpy(dtype=float)
    check("CI equals the documented percentile bootstrap", np.allclose(got, want, rtol=0, atol=1e-15))
    check("each comparison starts a fresh rng (order independent)",
          tb.bootstrap_ci(d) == tuple(want) and tb.bootstrap_ci(-d) == tb.bootstrap_ci(-d))

    raises("eval_fp differs on a seed", lambda: tb.compare(tb.load_user_churn(
        *user_churn_fixture(tmp / "fp", eval_fp_override="deadbeef0000")), 10), "eval_fp differs on seeds [3]")
    raises("baseline missing a seed", lambda: tb.compare(tb.load_user_churn(
        *user_churn_fixture(tmp / "drop", drop_seed=4)), 10), "differ from RAD's")
    raises("wrong number of seeds", lambda: tb.compare(tb.load_user_churn(*user_churn_fixture(tmp / "n")), 9), "expected 9")

    a, w, u = user_churn_fixture(tmp / "wl")
    raises("RDL folders swapped", lambda: tb.load_user_churn(a, u, w), "weighted_loss")

    out = tmp / "out.csv"
    tb.main(["--dataset", "hm", "--ablation", str(a), "--rdl-weighted", str(w), "--rdl-unweighted", str(u),
             "--out", str(out)])
    check("main writes the CSV", len(pd.read_csv(out)) == 24 and set(pd.read_csv(out)["dataset"]) == {"hm"})
    raises("main refuses to overwrite", lambda: tb.main(["--dataset", "hm", "--ablation", str(a), "--rdl-weighted",
                                                         str(w), "--rdl-unweighted", str(u), "--out", str(out)]),
           "refusing")


def test_lanl(tmp: Path):
    d = tmp / "lanl"; d.mkdir()
    ev = dict(eval_fp="abc123abc123", n_eval=50000, n_pos=127)
    six = []
    for s in SEEDS:
        for v, x in (("rad_llm_rules_no_bce", 0.8), ("llm_rules_with_bce", 0.99), ("mlp_supervised_detector", 0.7),
                     ("rf_supervised_detector", 0.99)):
            six.append(dict(variant=v, seed=s, auroc=x, auprc=x / 2, **{"p@127": x / 3, "precision_at_k": x / 3},
                            test_rows=50000, test_pos=127, **ev))
    pd.DataFrame(six).to_csv(d / "lanl_six_ablations_by_seed.csv", index=False)
    pd.DataFrame([dict(model=m, seed=s, auroc=0.5, auprc=0.1, **{"p@127": 0.1}, test_rows=50000, test_pos=127, **ev)
                  for m in ("Isolation Forest", "DRL", "Deep SVDD", "AE", "AE+Rank") for s in SEEDS]).to_csv(
        d / "lanl_tabular_baselines_by_seed.csv", index=False)
    for w, name in ((True, "weighted"), (False, "unweighted")):
        pd.DataFrame([dict(seed=s, auroc=0.6, auprc=0.2, **{"p@127": 0.2}, weighted_loss=w, **ev) for s in SEEDS]).to_csv(
            d / f"lanl_rdl_{name}_by_seed.csv", index=False)
    res = tb.compare(tb.load_lanl(d), 10).set_index(["baseline", "metric"])
    check("lanl: all baselines x metrics", len(res) == 24)
    check("lanl: headline is rad_llm_rules_no_bce", abs(res.loc[("DRL", "auroc"), "rad_mean"] - 0.8) < 1e-12)
    check("lanl: MLP is mlp_supervised_detector", abs(res.loc[("Supervised MLP", "auroc"), "baseline_mean"] - 0.7) < 1e-12)
    check("lanl: P@k is p@127", abs(res.loc[("DRL", "p@k"), "rad_mean"] - 0.8 / 3) < 1e-12)
    df = pd.read_csv(d / "lanl_tabular_baselines_by_seed.csv")
    df.loc[0, "n_pos"] = 126
    df.to_csv(d / "lanl_tabular_baselines_by_seed.csv", index=False)
    raises("lanl: n_pos != 127", lambda: tb.load_lanl(d), "n_pos != 127")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        test_user_churn(Path(t))
        test_lanl(Path(t))
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nall passed")
    sys.exit(1 if FAILURES else 0)
