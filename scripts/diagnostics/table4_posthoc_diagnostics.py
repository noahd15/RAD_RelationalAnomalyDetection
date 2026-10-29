#!/usr/bin/env python3
"""Post-hoc diagnostics on the final_20261028 Table 4 outputs (exploratory; no training).

    CUDA_VISIBLE_DEVICES= python scripts/diagnostics/table4_posthoc_diagnostics.py {hm,amazon,lanl} --out DIR

hm / amazon: re-executes only the data-preparation part of the RAD script (imports, then SECTION 3
from the task tables through the end of SECTION 4: val fit/report split, activity features, at-risk
filter, downsampling), per seed, on CPU, and stops if a seed's report rows do not reproduce the
eval_fingerprint recorded in results/<ds>/final_20261028/per_seed_metrics.csv. With those rows it
writes
  - rules_only_lr.csv: logistic regression on the saved rule table alone (train rows -> report
    rows), for the LLM rule table and the RF-sampled rule table of each seed;
  - activity_by_label.csv: recent-activity features of the report rows by label, plus the churners
    the at-risk filter removed from the report partition (for context).
lanl: rebuilds each seed's LLM rule matrix from rules/seed_N/llm_parsed_rules_df.csv with the run's
own _rule_matrix_for_auth_ids, checks its per-rule match counts against six_ablations.log, and
writes rules_only_lr.csv (train mask -> test mask).
--heuristics-only (hm / amazon): after the same row check, writes only heuristic_scores.csv, the
AUROC/AUPRC of single activity features (negated recent activity; Amazon also days since last
review) as anomaly scores on the report rows.
Nothing under results/ is written; --out must not exist.
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
SEEDS = range(10)
LABEL_ENV = {"ATRISK_TAU_A": "0.05", "ATRISK_TAU_B": "10.0", "ATRISK_TAU_ALPHA": "0.02",
             "ATRISK_TAU_MULTIPLIER": "1.0", "TARGET_ANOM_RATE": "0.015"}
CLEARED_ENV = ("VAL_FIT_FRAC", "POS_DOWNSAMPLE_MODE", "SEED")
DATASETS = {
    "hm": {"script": "scripts/hm_rad_vs_supervised_rdl.py", "prefix": "hm10seed",
           "activity": ["recent_txns", "prev_txns_30d", "trend_w", "avg_activity"],
           "heuristics": {"neg_recent_txns": ("recent_txns", -1.0)}},
    "amazon": {"script": "scripts/rad_vs_supervised_rdl.py", "prefix": "amazon10seed",
               "activity": ["recent_reviews", "prev_reviews", "trend_w", "avg_activity", "days_since_last_review"],
               "heuristics": {"neg_recent_reviews": ("recent_reviews", -1.0),
                              "days_since_last_review": ("days_since_last_review", 1.0)}},
}
DATASET_BCE_LS = {"hm": "1.0", "amazon": "0.1"}  # LAMBDA_STRUCT of the with-BCE arm (rule-table tag)
LR_SPEC = "sklearn LogisticRegression(C=1.0, lbfgs, max_iter=5000), no class weights, binary rule columns unscaled"


def lr_scores(X_tr, y_tr, X_te):
    if X_tr.shape[1] == 0 or np.unique(y_tr).size < 2:
        return None
    clf = LogisticRegression(C=1.0, max_iter=5000)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def metrics(y, s):
    if s is None:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ── H&M / Amazon ──────────────────────────────────────────────────────────────

def script_parts(src: str):
    """(imports, dataset load, per-seed data prep) slices of the RAD script."""
    a = src.index("# SECTION 2 — SETUP")
    b = src.index("rdl_dataset = get_dataset(")
    c = src.index('rdl_train_table = rdl_task.get_table("train")')
    d = src.index("# SECTION 4b")
    for marker in ("# SECTION 2 — SETUP", "rdl_dataset = get_dataset(", 'rdl_train_table = rdl_task.get_table("train")',
                   "# SECTION 4b"):
        if src.count(marker) != 1:
            raise RuntimeError(f"marker {marker!r} occurs {src.count(marker)} times")
    return src[:a], src[b:c], src[c:d]


class FreshTables:
    """The task, with get_table returning a deep copy of the (cached) table."""

    def __init__(self, task):
        self._task = task

    def get_table(self, *args, **kwargs):
        return copy.deepcopy(self._task.get_table(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._task, name)


def find_rule_table(rules_dir: Path, prefix: str, seed: int, tag: str) -> Path:
    hits = sorted(rules_dir.glob(f"{prefix}_seed{seed}_{tag}_*/customer_rule_table.pt"))
    if len(hits) != 1:
        raise RuntimeError(f"{rules_dir}: expected one {prefix}_seed{seed}_{tag}_* rule table, found {len(hits)}")
    return hits[0]


def run_relbench(ds: str, out: Path, heuristics_only: bool = False) -> None:
    import torch

    cfg = DATASETS[ds]
    final = REPO / "results" / ds / "final_20261028"
    per_seed = pd.read_csv(final / "per_seed_metrics.csv")
    script = REPO / cfg["script"]
    imports, load, prep = script_parts(script.read_text())
    sys.path.insert(0, str(script.parent))
    for k in CLEARED_ENV:
        os.environ.pop(k, None)
    os.environ.update(LABEL_ENV)
    base = {"__file__": str(script), "__name__": "posthoc_prep"}
    exec(compile(imports, str(script), "exec"), base)  # noqa: S102
    exec(compile(load, str(script), "exec"), base)  # noqa: S102
    cust = base["rdl_db"].table_dict["customer"].df["customer_id"].astype(np.int64).tolist()
    cid_to_nidx = pd.Series(np.arange(len(cust)), index=cust)

    lr_rows, act_rows, heur_rows, checks = [], [], [], []
    for seed in SEEDS:
        # The script splits and filters the task tables in place, and relbench's get_table returns
        # the same cached Table each call, so each seed gets fresh copies.
        ns = dict(base, RUN_SEED=seed, rdl_task=FreshTables(base["rdl_task"]))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(prep, str(script), "exec"), ns)  # noqa: S102
        test_df, y_te = ns["rdl_test_table"].df, ns["_test_labels"]
        train_df = ns["rdl_train_table"].df
        y_tr = train_df["churn"].astype(np.int64).to_numpy()
        fp = ns["eval_fingerprint"](test_df, y_te)
        want = per_seed.loc[per_seed["seed"] == seed, ["eval_fp", "n_eval", "n_pos"]].drop_duplicates()
        if len(want) != 1 or (want.iloc[0]["eval_fp"], int(want.iloc[0]["n_eval"]), int(want.iloc[0]["n_pos"])) \
                != (fp, len(y_te), int(y_te.sum())):
            raise RuntimeError(f"{ds} seed {seed}: rebuilt report rows {fp}/{len(y_te)}/{int(y_te.sum())} "
                               f"!= final run {want.to_dict('records')}")
        if heuristics_only:
            # A single activity feature as the anomaly score (sign chosen so that higher = more anomalous).
            act = ns["_activity_features_for_targets"](test_df, lookback_days=30)
            for name, (col, sign) in cfg["heuristics"].items():
                heur_rows.append({"dataset": ds, "seed": seed, "heuristic": name, "feature": col, "sign": sign,
                                  **metrics(y_te, sign * act[col].to_numpy(dtype=float)),
                                  "prevalence": float(y_te.mean()), "eval_fp": fp, "n_eval": len(y_te),
                                  "n_pos": int(y_te.sum())})
            print(f"[{ds}] seed {seed}: report rows {fp}/{len(y_te)}/{int(y_te.sum())} ok; "
                  + "; ".join(f"{r['heuristic']} auprc={r['auprc']:.4f} auroc={r['auroc']:.4f}"
                              for r in heur_rows[-len(cfg["heuristics"]):]), flush=True)
            continue
        llm_nobce = find_rule_table(final / "rules", cfg["prefix"], seed, "ls0.0_llm")
        llm_bce = find_rule_table(final / "rules", cfg["prefix"], seed, f"ls{DATASET_BCE_LS[ds]}_llm")
        if md5(llm_nobce) != md5(llm_bce):
            raise RuntimeError(f"{ds} seed {seed}: the two LLM rule tables differ")
        checks.append({"seed": seed, "eval_fp": fp, "n_eval": len(y_te), "n_pos": int(y_te.sum()),
                       "n_train": len(y_tr), "n_train_pos": int(y_tr.sum()), "llm_rule_table_md5": md5(llm_nobce)})

        tr_idx = cid_to_nidx.reindex(train_df["customer_id"].astype(np.int64).to_numpy()).to_numpy()
        te_idx = cid_to_nidx.reindex(test_df["customer_id"].astype(np.int64).to_numpy()).to_numpy()
        if np.isnan(tr_idx.astype(float)).any() or np.isnan(te_idx.astype(float)).any():
            raise RuntimeError(f"{ds} seed {seed}: customer ids missing from the customer table")
        for source, path in (("llm", llm_nobce), ("rf_sampled", find_rule_table(final / "rules", cfg["prefix"], seed, "ls0.0_rf_seed"))):
            obj = torch.load(path, map_location="cpu", weights_only=False)
            mat = obj["rule_table"][:, :obj["n_rules"]].numpy()
            X_tr, X_te = mat[tr_idx.astype(int)], mat[te_idx.astype(int)]
            s = lr_scores(X_tr, y_tr, X_te)
            lr_rows.append({"dataset": ds, "seed": seed, "rule_source": source, "n_rules": int(obj["n_rules"]),
                            "report_coverage": float((X_te.sum(axis=1) > 0).mean()),
                            "report_pos_coverage": float((X_te[y_te == 1].sum(axis=1) > 0).mean()),
                            "n_distinct_report_scores": int(np.unique(s).size) if s is not None else 0,
                            **metrics(y_te, s), "prevalence": float(y_te.mean()), "rule_table_md5": md5(path)})

        act = ns["_activity_features_for_targets"](test_df, lookback_days=30)
        raw = ns["_rdl_orig_test_table"].df["churn"].astype(np.int64).to_numpy()
        removed = ns["rdl_activity_test"][(raw == 1) & ~ns["rdl_test_keep"]]
        groups = {"report_positive": act[y_te == 1], "report_negative": act[y_te == 0], "report_all": act,
                  "removed_at_risk_churner": removed}
        for g, df in groups.items():
            for col in cfg["activity"]:
                v = df[col].to_numpy(dtype=float)
                row = {"dataset": ds, "seed": seed, "group": g, "feature": col, "n": len(v),
                       "mean": float(np.mean(v)) if len(v) else float("nan"),
                       "median": float(np.median(v)) if len(v) else float("nan"),
                       "frac_zero": float((v == 0).mean()) if len(v) else float("nan")}
                if g == "report_positive":  # AUROC of the feature as a churn score: <0.5 = positives lower
                    row["auroc_pos_vs_neg"] = float(roc_auc_score(y_te, act[col].to_numpy(dtype=float)))
                act_rows.append(row)
        print(f"[{ds}] seed {seed}: report rows {fp}/{len(y_te)}/{int(y_te.sum())} ok; "
              + "; ".join(f"{r['rule_source']} LR auprc={r['auprc']:.4f} auroc={r['auroc']:.4f}"
                          for r in lr_rows[-2:]), flush=True)

    if heuristics_only:
        pd.DataFrame(heur_rows).to_csv(out / "heuristic_scores.csv", index=False)
        return
    pd.DataFrame(lr_rows).to_csv(out / "rules_only_lr.csv", index=False)
    pd.DataFrame(act_rows).to_csv(out / "activity_by_label.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "rebuilt_rows_check.csv", index=False)



# ── LANL ──────────────────────────────────────────────────────────────────────

def logged_matches(log_text: str) -> dict[int, list[int]]:
    """seed -> per-rule match counts the run logged for its LLM rule table."""
    out, seed, in_llm_run = {}, None, False
    for line in log_text.splitlines():
        m = re.match(r"\[seed (\d+)\] starting", line)
        if m:
            seed = int(m.group(1))
        if "[run] " in line:
            in_llm_run = bool(re.search(r"\[run\] (rad_llm_rules_no_bce|llm_rules_with_bce) ", line))
        m = re.search(r"\[rules\] \d+ rule columns; matches per rule = (\[.*\])", line)
        if m and in_llm_run:
            counts = json.loads(m.group(1))
            if out.setdefault(seed, counts) != counts:
                raise RuntimeError(f"LANL seed {seed}: the two LLM runs logged different match counts")
    return out


def run_lanl(out: Path) -> None:
    import torch

    lanl = REPO / "lanl"
    final = lanl / "results" / "final_20261028"
    os.chdir(lanl)
    sys.path.insert(0, str(lanl))
    import run_lanl_six_ablations as six  # noqa: E402

    base_df = pd.read_parquet(six.PARQUET_PATH)
    base_df = base_df.sort_values(["time", "auth_id"]).reset_index(drop=True)
    base_df["auth_id"] = pd.to_numeric(base_df["auth_id"], errors="coerce")
    base_df = base_df.dropna(subset=["auth_id"]).copy()
    base_df["auth_id"] = base_df["auth_id"].astype(np.int64)
    graph = torch.load(six.GRAPH_PATH, map_location="cpu", weights_only=False)
    auth = graph["auth"]
    with contextlib.redirect_stdout(io.StringIO()):
        fp = six.auth_eval_fingerprint(auth)
    by_seed = pd.read_csv(final / "lanl_six_ablations_by_seed.csv")
    if {fp["eval_fp"]} != set(by_seed["eval_fp"]):
        raise RuntimeError(f"LANL eval set {fp} != final run {set(by_seed['eval_fp'])}")
    y = auth.y.view(-1).numpy().astype(np.int64)
    tr = auth.train_mask.view(-1).numpy().astype(bool)
    te = auth.test_mask.view(-1).numpy().astype(bool)
    auth_ids = auth.auth_id.view(-1).numpy().astype(np.int64)
    logged = logged_matches((final / "six_ablations.log").read_text())

    rows = []
    for seed in SEEDS:
        csv = final / "rules" / f"seed_{seed:03d}" / "llm_parsed_rules_df.csv"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mat, cols = six._rule_matrix_for_auth_ids(base_df, auth_ids, str(csv))
        m = re.search(r"matches per rule = (\[.*\])", buf.getvalue())
        got = json.loads(m.group(1)) if m else None
        if got != logged.get(seed):
            raise RuntimeError(f"LANL seed {seed}: rebuilt match counts {got} != logged {logged.get(seed)}")
        s = lr_scores(mat[tr], y[tr], mat[te])
        rows.append({"dataset": "lanl", "seed": seed, "rule_source": "llm", "n_rules": len(cols),
                     "report_coverage": float((mat[te].sum(axis=1) > 0).mean()),
                     "report_pos_coverage": float((mat[te][y[te] == 1].sum(axis=1) > 0).mean()),
                     "n_distinct_report_scores": int(np.unique(s).size) if s is not None else 0,
                     **metrics(y[te], s), "prevalence": float(y[te].mean()), "n_train": int(tr.sum()),
                     "n_train_pos": int(y[tr].sum()), "eval_fp": fp["eval_fp"]})
        print(f"[lanl] seed {seed}: {len(cols)} rules, match counts match the log; "
              f"LR auprc={rows[-1]['auprc']:.4f} auroc={rows[-1]['auroc']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "rules_only_lr.csv", index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["hm", "amazon", "lanl"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--heuristics-only", action="store_true",
                    help="hm/amazon: only score single activity features (heuristic_scores.csv)")
    a = ap.parse_args(argv)
    if a.heuristics_only and a.dataset == "lanl":
        ap.error("--heuristics-only is for hm and amazon")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES= (CPU only).")
    out = a.out.resolve()
    if out.exists():
        raise RuntimeError(f"{out} exists; refusing to overwrite it.")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({"dataset": a.dataset, "label_env": LABEL_ENV, "lr": LR_SPEC,
                                                 "seeds": list(SEEDS), "script": __file__,
                                                 "heuristics_only": a.heuristics_only}, indent=2) + "\n")
    run_lanl(out) if a.dataset == "lanl" else run_relbench(a.dataset, out, a.heuristics_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
