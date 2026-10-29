#!/usr/bin/env python3
"""Summary tables for the post-hoc Table 4 diagnostics (exploratory).

    python scripts/diagnostics/table4_posthoc_summary.py --dir docs/diagnostics/posthoc_20261029

Reads DIR/{hm,amazon,lanl}/ (written by table4_posthoc_diagnostics.py) and the final_20261028
metrics, and writes into DIR (refusing to overwrite):
  - rad_vs_rules_only_lr.csv: per seed, each RAD row next to the rules-only logistic regression on
    the same seed's rule table, with the difference; mean rows at the end (seed = "mean").
  - lanl_stability.csv: per seed, the LANL RAD AUPRCs and the LLM rule counts from
    six_ablations.log; lanl_stability_corr.csv: Pearson and Spearman correlations across seeds.
  - activity_summary.csv: per dataset, group and feature, the mean over seeds of the per-seed
    mean / median / fraction zero (and the per-seed feature AUROC, positives vs negatives).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]
# RAD row in the final per-seed metrics -> rule source of its rule table.
RAD_ROWS = {"hm": {"RAD (no BCE)": "llm", "RAD (with BCE)": "llm", "RAD (RF-sampled rules, no BCE)": "rf_sampled"},
            "amazon": {"RAD (no BCE)": "llm", "RAD (with BCE)": "llm", "RAD (RF-sampled rules, no BCE)": "rf_sampled"},
            "lanl": {"rad_llm_rules_no_bce": "llm", "llm_rules_with_bce": "llm"}}
LANL_FINAL = REPO / "lanl" / "results" / "final_20261028"


def final_rows(ds: str) -> pd.DataFrame:
    if ds == "lanl":
        df = pd.read_csv(LANL_FINAL / "lanl_six_ablations_by_seed.csv").rename(columns={"variant": "model"})
    else:
        df = pd.read_csv(REPO / "results" / ds / "final_20261028" / "per_seed_metrics.csv")
    return df[df["model"].isin(RAD_ROWS[ds])][["model", "seed", "auroc", "auprc"]]


def rad_vs_lr(d: Path) -> pd.DataFrame:
    out = []
    for ds in RAD_ROWS:
        lr = pd.read_csv(d / ds / "rules_only_lr.csv")
        rad = final_rows(ds)
        for model, source in RAD_ROWS[ds].items():
            m = rad[rad["model"] == model].merge(lr[lr["rule_source"] == source], on="seed", suffixes=("_rad", "_lr"))
            if len(m) != 10:
                raise RuntimeError(f"{ds} {model}: {len(m)} paired seeds, expected 10")
            for metric in ("auroc", "auprc"):
                m[f"{metric}_rad_minus_lr"] = m[f"{metric}_rad"] - m[f"{metric}_lr"]
            m = m[["dataset", "model", "rule_source", "seed", "n_rules", "auroc_rad", "auroc_lr", "auroc_rad_minus_lr",
                   "auprc_rad", "auprc_lr", "auprc_rad_minus_lr", "prevalence"]]
            mean = m.drop(columns=["dataset", "model", "rule_source", "seed"]).mean().to_dict()
            out += [m, pd.DataFrame([{"dataset": ds, "model": model, "rule_source": source, "seed": "mean", **mean}])]
    return pd.concat(out, ignore_index=True)


def lanl_rule_counts() -> pd.DataFrame:
    rows, seed = [], None
    for line in (LANL_FINAL / "six_ablations.log").read_text().splitlines():
        m = re.match(r"\[seed (\d+)\] starting", line)
        if m:
            seed = int(m.group(1))
        m = re.search(r"\[llm-rules\] parsed=(\d+) rejected\(missing columns\)=(\d+) sql_errors=(\d+) "
                      r"dropped\(val_support<1\)=(\d+) kept=(\d+)", line)
        if m:
            rows.append({"seed": seed, **dict(zip(["parsed", "rejected_missing_columns", "sql_errors",
                                                   "dropped_val_support", "kept"], map(int, m.groups())))})
        if "response did not close its JSON array" in line:
            rows.append({"seed": seed, "truncated": True})
    df = pd.DataFrame(rows).groupby("seed").first().reset_index()
    df["truncated"] = df["truncated"].fillna(False).astype(bool) if "truncated" in df else False
    return df


def lanl_stability(d: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rad = final_rows("lanl").pivot(index="seed", columns="model", values="auprc")
    rad.columns = [f"auprc_{c}" for c in rad.columns]
    lr = pd.read_csv(d / "lanl" / "rules_only_lr.csv").set_index("seed")[["auprc"]].rename(
        columns={"auprc": "auprc_rules_only_lr"})
    st = lanl_rule_counts().set_index("seed").join(rad).join(lr).reset_index()
    st["max_rank_loss_spike"] = np.nan  # not in six_ablations.log: run_gae_single logs no per-epoch losses
    corr = []
    for y in [c for c in st.columns if c.startswith("auprc_rad") or c.startswith("auprc_llm")]:
        for x in ("kept", "auprc_rules_only_lr"):
            p, s = pearsonr(st[x], st[y]), spearmanr(st[x], st[y])
            loo = [pearsonr(st[x].drop(i), st[y].drop(i))[0] for i in st.index]
            corr.append({"x": x, "y": y, "n": len(st), "pearson_r": p[0], "pearson_p": p[1], "spearman_rho": s[0],
                         "spearman_p": s[1], "pearson_r_leave_one_seed_out_min": min(loo),
                         "pearson_r_leave_one_seed_out_max": max(loo)})
    return st, pd.DataFrame(corr)


def activity_summary(d: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(d / ds / "activity_by_label.csv") for ds in ("hm", "amazon")])
    agg = {"n": "mean", "mean": "mean", "median": "mean", "frac_zero": "mean", "auroc_pos_vs_neg": ["mean", "min", "max"]}
    s = df.groupby(["dataset", "group", "feature"], sort=False).agg(agg)
    s.columns = [f"{a}_over_seeds" if b == "mean" else f"{a}_{b}_over_seeds" for a, b in s.columns]
    return s.reset_index().dropna(axis=1, how="all")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    a = ap.parse_args(argv)
    st, corr = lanl_stability(a.dir)
    outputs = {"rad_vs_rules_only_lr.csv": rad_vs_lr(a.dir), "lanl_stability.csv": st,
               "lanl_stability_corr.csv": corr, "activity_summary.csv": activity_summary(a.dir)}
    for name in outputs:
        if (a.dir / name).exists():
            raise RuntimeError(f"{a.dir / name} exists; refusing to overwrite it.")
    for name, df in outputs.items():
        df.to_csv(a.dir / name, index=False)
        print(f"wrote {a.dir / name} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
