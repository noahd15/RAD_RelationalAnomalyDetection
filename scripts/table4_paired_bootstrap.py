#!/usr/bin/env python3
"""Paired-by-seed bootstrap CIs for Table 4: the headline RAD row against each baseline row.

    python scripts/table4_paired_bootstrap.py --dataset hm --ablation results/hm/<run> \\
        --rdl-weighted results/hm/<run>_rdl_weighted --rdl-unweighted results/hm/<run>_rdl_unweighted --out OUT.csv
    python scripts/table4_paired_bootstrap.py --dataset lanl --lanl lanl/results/<run> --out OUT.csv

Method (docs/table4_reporting_plan.md, "Comparisons"):
- d_s = RAD_s - baseline_s for each seed s. A seed's pair is used only if both rows have the same
  eval_fp, n_eval and n_pos; otherwise this script stops. Both rows must cover the same seeds,
  and there must be --n-seeds of them.
- The seeds are resampled with replacement 10,000 times; the CI is the 2.5th-97.5th percentile
  of the resampled means. Each (baseline, metric) comparison starts a fresh
  numpy.random.default_rng(20261028), so every comparison uses the same resamples and the result
  does not depend on the order of the comparisons.
- Verdict: "RAD better" if the CI is entirely above 0, "baseline better" if entirely below 0,
  otherwise "no difference".
- Metrics: AUROC, AUPRC and P@k (k = n_pos). On LANL that is the p@127 column, and every LANL
  row must have n_pos == 127.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_RESAMPLES = 10_000
RNG_SEED = 20261028
METRICS = ("auroc", "auprc", "p@k")
EVAL_COLUMNS = ("eval_fp", "n_eval", "n_pos")
HEADLINE = "RAD"  # model name of the headline row after loading
BASELINES = ("Isolation Forest", "DRL", "Deep SVDD", "AE", "AE+Rank", "Supervised RDL (weighted)",
             "Supervised RDL (unweighted)", "Supervised MLP")
LANL_K = 127
COLUMNS = ["model", "seed", *METRICS, *EVAL_COLUMNS]


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"eval_fp": str})


def _rdl(path, weighted: bool) -> pd.DataFrame:
    df = _read(path)
    if "weighted_loss" in df.columns and set(df["weighted_loss"].astype(str)) != {str(weighted)}:
        raise RuntimeError(f"{path}: weighted_loss is {sorted(set(df['weighted_loss'].astype(str)))}, expected {weighted}")
    return df.assign(model=f"Supervised RDL ({'weighted' if weighted else 'unweighted'})")


def load_user_churn(ablation, rdl_weighted, rdl_unweighted) -> pd.DataFrame:
    """H&M or Amazon rows: the ablation runner's per_seed_metrics.csv plus the two RDL runs."""
    ab = _read(Path(ablation) / "per_seed_metrics.csv")
    ab = ab[ab["model"].isin([*BASELINES, "RAD (no BCE)"])].replace({"model": {"RAD (no BCE)": HEADLINE}})
    rows = [ab, _rdl(Path(rdl_weighted) / "per_seed.csv", True), _rdl(Path(rdl_unweighted) / "per_seed.csv", False)]
    return pd.concat([r[COLUMNS] for r in rows], ignore_index=True)


def load_lanl(run_dir) -> pd.DataFrame:
    d = Path(run_dir)
    six = _read(d / "lanl_six_ablations_by_seed.csv")
    six = six[six["variant"].isin(["rad_llm_rules_no_bce", "mlp_supervised_detector"])]
    six = six.assign(model=six["variant"].map({"rad_llm_rules_no_bce": HEADLINE,
                                               "mlp_supervised_detector": "Supervised MLP"}))
    rows = [six, _read(d / "lanl_tabular_baselines_by_seed.csv"),
            _rdl(d / "lanl_rdl_weighted_by_seed.csv", True), _rdl(d / "lanl_rdl_unweighted_by_seed.csv", False)]
    df = pd.concat([r.rename(columns={"p@127": "p@k"})[COLUMNS] for r in rows], ignore_index=True)
    if (df["n_pos"] != LANL_K).any():
        raise RuntimeError(f"LANL rows with n_pos != {LANL_K}, where p@{LANL_K} is not P@k with k = n_pos:\n"
                           f"{df[df['n_pos'] != LANL_K][['model', 'seed', 'n_pos']]}")
    return df


def _one_row_per_seed(df: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = df[df["model"] == model]
    if rows.empty:
        raise RuntimeError(f"no rows for {model!r}")
    if rows["seed"].duplicated().any():
        raise RuntimeError(f"{model!r}: more than one row for seeds {sorted(rows[rows['seed'].duplicated()]['seed'])}")
    return rows.set_index("seed").sort_index()


def bootstrap_ci(d: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    means = d[rng.integers(0, len(d), size=(N_RESAMPLES, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def verdict(lo: float, hi: float) -> str:
    return "RAD better" if lo > 0 else "baseline better" if hi < 0 else "no difference"


def compare(df: pd.DataFrame, n_seeds: int) -> pd.DataFrame:
    rad = _one_row_per_seed(df, HEADLINE)
    out = []
    for base_name in BASELINES:
        base = _one_row_per_seed(df, base_name)
        if list(base.index) != list(rad.index):
            raise RuntimeError(f"{base_name!r} seeds {list(base.index)} differ from RAD's {list(rad.index)}")
        if len(rad) != n_seeds:
            raise RuntimeError(f"{len(rad)} seeds, expected {n_seeds}")
        for c in EVAL_COLUMNS:
            as_type = str if c == "eval_fp" else int
            differs = rad[c].map(as_type) != base[c].map(as_type)
            if differs.any():
                raise RuntimeError(f"RAD vs {base_name!r}: {c} differs on seeds {rad.index[differs].tolist()}")
        for m in METRICS:
            d = (rad[m] - base[m]).to_numpy(dtype=float)
            lo, hi = bootstrap_ci(d)
            out.append({"baseline": base_name, "metric": m, "n_seeds": len(d),
                        "rad_mean": rad[m].mean(), "baseline_mean": base[m].mean(), "mean_diff": d.mean(),
                        "ci_low": lo, "ci_high": hi, "verdict": verdict(lo, hi),
                        "n_resamples": N_RESAMPLES, "rng_seed": RNG_SEED})
    return pd.DataFrame(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hm", "amazon", "lanl"])
    ap.add_argument("--ablation")
    ap.add_argument("--rdl-weighted")
    ap.add_argument("--rdl-unweighted")
    ap.add_argument("--lanl")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise RuntimeError(f"{out} exists; refusing to overwrite it.")
    if a.dataset == "lanl":
        if not a.lanl:
            ap.error("--lanl is required for lanl")
        df = load_lanl(a.lanl)
    else:
        if not (a.ablation and a.rdl_weighted and a.rdl_unweighted):
            ap.error("--ablation, --rdl-weighted and --rdl-unweighted are required for hm/amazon")
        df = load_user_churn(a.ablation, a.rdl_weighted, a.rdl_unweighted)
    res = compare(df, a.n_seeds).assign(dataset=a.dataset)
    res = res[["dataset", *[c for c in res.columns if c != "dataset"]]]
    res.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.drop(columns=["n_resamples", "rng_seed"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[bootstrap] {len(res)} comparisons, {N_RESAMPLES} resamples, rng {RNG_SEED} -> {out}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")
