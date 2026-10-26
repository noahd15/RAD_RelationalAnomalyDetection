#!/usr/bin/env python3
"""
LANL-only six ablations for RAD-style GAE experiments.

Variants:
1) rad_llm_rules_no_bce
2) llm_rules_with_bce
3) rf_sampled_rules_no_bce
4) rf_sampled_rules_with_bce
5) rf_decision_features_no_bce
6) rf_decision_features_with_bce

Outputs:
  results/lanl_six_ablations_by_seed.csv
  results/lanl_six_ablations_mean_std.csv
"""

from __future__ import annotations

import copy
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


# --------------------
# Config
# --------------------
_HERE = Path(__file__).resolve().parent
SEEDS = list(range(int(os.environ.get("START_SEED", "0")),
                   int(os.environ.get("START_SEED", "0")) + int(os.environ.get("N_SEEDS", "10"))))

PARQUET_PATH = str(_HERE / "data" / "flat_features.parquet")
GRAPH_PATH = str(_HERE / "data" / "sampled_hetero_graph.pt")
LLM_RULES_CSV = str(_HERE / "data" / "parsed_rules_df.csv")
RESULTS_DIR = os.environ.get("LANL_RESULTS_DIR", "./results")
BY_SEED_CSV = os.path.join(RESULTS_DIR, "lanl_six_ablations_by_seed.csv")
MEAN_STD_CSV = os.path.join(RESULTS_DIR, "lanl_six_ablations_mean_std.csv")
RF_SAMPLED_RULES_CSV = os.path.join(RESULTS_DIR, "lanl_rf_sampled_rules.csv")
RAW_RESULTS_DIR = os.environ.get("LANL_RAW_RESULTS_DIR", os.path.join(RESULTS_DIR, "lanl_six_ablations_raw"))
PARTIAL_AUROC_MAX_FPR = float(os.environ.get("LANL_PARTIAL_AUROC_MAX_FPR", "0.1"))
# Supervised-MLP detector config; defaults are the untuned values. tune_rf_mlp.py picks
# tuned values (train-only CV), passed as e.g. LANL_MLP_HIDDEN=128,64,32 LANL_MLP_ALPHA=1e-5.
MLP_HIDDEN = tuple(int(h) for h in os.environ.get("LANL_MLP_HIDDEN", "64,32").split(","))
MLP_ALPHA = float(os.environ.get("LANL_MLP_ALPHA", "1e-4"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GAE_HP = {
    "hidden_dim": 128,
    "latent_dim": 16,
    "activation": "relu",
    "dropout": 0.2755,
    "lr": 0.00423,
    "weight_decay": 3.4e-4,
    "lambda_struct": 0.9362,
    "lambda_rank": 0.1739,
    "rank_max_pairs": 4096,
    "score_feat_alpha": 0.5634,
    "score_struct_alpha": 1.7857,
    "epochs": 1500,
    "patience": 100,
}

RF_PARAMS = {
    "n_estimators": 1200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": 42,
    "class_weight": "balanced_subsample",
}

MIN_LEAF_TRAIN_SAMPLES = 10
MIN_LEAF_POS_PROB = 0.20
MIN_VAL_SUPPORT = 1
MIN_VAL_CONFIDENCE = 0.0
PATH_TOP_TREES = 200
PATHS_PER_TREE = 30
SAMPLED_RULE_COUNT = 30
SQL_KEYWORDS = {"AND", "OR", "NOT", "IN", "IS", "NULL", "TRUE", "FALSE", "BETWEEN", "LIKE"}

# LANL_LLM_RULES selects the rules behind the llm variants:
#   cached  -- lanl/data/parsed_rules_df.csv. rule_mining.ipynb selected and filtered these with
#              test labels and features, so rows built from them are contaminated.
#   per_run -- regenerated every seed: RF seed rules scored on val only, refined by the LLM,
#              then kept only if they match val rows (val is the rule-selection split; test is not read).
LLM_RULES_MODE = os.environ.get("LANL_LLM_RULES", "cached").strip()
if LLM_RULES_MODE not in {"cached", "per_run"}:
    raise ValueError(f"LANL_LLM_RULES must be cached|per_run, got {LLM_RULES_MODE!r}")
LLM_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LLM_GEN_KWARGS = dict(max_new_tokens=int(os.environ.get("LLM_MAX_NEW_TOKENS", "4000")),
                      do_sample=False, temperature=0.0, top_p=1.0)
LLM_N_SEED_RULES = 20


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true).astype(np.int8).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    n = int(y_true.size)
    if n == 0:
        return float("nan")
    k_eff = int(max(1, min(int(k), n)))
    top_idx = np.argsort(-y_score, kind="mergesort")[:k_eff]
    return float(y_true[top_idx].mean())


def metrics_row(variant: str, seed: int, y_true: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int8).reshape(-1)
    score = np.asarray(score).astype(np.float64).reshape(-1)
    if np.unique(y_true).size > 1:
        auroc = float(roc_auc_score(y_true, score))
        auprc = float(average_precision_score(y_true, score))
        pauroc = float(roc_auc_score(y_true, score, max_fpr=PARTIAL_AUROC_MAX_FPR)) if 0.0 < PARTIAL_AUROC_MAX_FPR < 1.0 else float("nan")
    else:
        auroc, auprc, pauroc = float("nan"), float("nan"), float("nan")
    prev = float(y_true.mean())
    return {
        "variant": variant,
        "seed": seed,
        "auroc": auroc,
        "auprc": auprc,
        "pauroc": pauroc,
        "p@127": precision_at_k(y_true, score, 127),
        "precision_at_k": precision_at_k(y_true, score, int(max(1, y_true.sum()))),
        "auprc_lift_vs_prev": float(auprc / prev) if prev > 0 else float("nan"),
        "test_rows": float(y_true.size),
        "test_pos": float(y_true.sum()),
        "test_prevalence": prev,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "auroc",
        "auprc",
        "pauroc",
        "p@127",
        "precision_at_k",
        "auprc_lift_vs_prev",
    ]
    g = df.groupby("variant", as_index=False)[metric_cols].agg(["mean", "std"])
    g.columns = ["variant"] + [f"{c}_{s}" for c, s in g.columns.tolist()[1:]]
    return g


# --------------------
# RF rule mining helpers
# --------------------
def _condition_to_sql(col: str, op: str, thr: float) -> str:
    if abs(thr - round(thr)) < 1e-9:
        v = str(int(round(thr)))
    else:
        v = f"{thr:.6f}".rstrip("0").rstrip(".")
    return f"{col} {op} {v}"


def _extract_tree_rule_candidates(
    tree,
    tree_id: int,
    feature_names: Sequence[str],
    max_rules_per_tree: Optional[int] = None,
) -> List[Dict]:
    t = tree.tree_
    left = t.children_left
    right = t.children_right
    feat = t.feature
    thr = t.threshold
    value = t.value.squeeze(axis=1)
    n_samples = t.n_node_samples

    rows: List[Dict] = []

    stack: List[Tuple[int, List[Tuple[str, str, float]]]] = [(0, [])]
    while stack:
        node_id, path = stack.pop()
        is_leaf = left[node_id] == right[node_id]
        if is_leaf:
            total = float(value[node_id].sum())
            pos = float(value[node_id][1]) if value[node_id].shape[0] > 1 else 0.0
            pos_prob = (pos / total) if total > 0 else 0.0
            support = int(n_samples[node_id])
            if support >= MIN_LEAF_TRAIN_SAMPLES and pos_prob >= MIN_LEAF_POS_PROB and len(path) >= 1:
                clauses = [_condition_to_sql(c, o, th) for c, o, th in path]
                rows.append(
                    {
                        "tree_id": int(tree_id),
                        "leaf_id": int(node_id),
                        "train_support": support,
                        "leaf_pos_prob": float(pos_prob),
                        "conds": path.copy(),
                        "rule_text": " AND ".join(clauses),
                    }
                )
            continue

        fidx = int(feat[node_id])
        if fidx < 0:
            continue
        fname = feature_names[fidx]
        threshold = float(thr[node_id])

        stack.append((right[node_id], path + [(fname, ">", threshold)]))
        stack.append((left[node_id], path + [(fname, "<=", threshold)]))

    rows = sorted(rows, key=lambda r: (r["leaf_pos_prob"], r["train_support"]), reverse=True)
    if max_rules_per_tree is None:
        return rows
    return rows[: int(max_rules_per_tree)]


def _apply_conditions(X_sub: pd.DataFrame, conds: Sequence[Tuple[str, str, float]]) -> np.ndarray:
    m = np.ones(len(X_sub), dtype=bool)
    for col, op, thr in conds:
        if col not in X_sub.columns:
            return np.zeros(len(X_sub), dtype=bool)
        vals = pd.to_numeric(X_sub[col], errors="coerce").fillna(0.0).to_numpy()
        if op == "<=":
            m &= vals <= thr
        elif op == ">":
            m &= vals > thr
        else:
            return np.zeros(len(X_sub), dtype=bool)
    return m


def _score_rules(rule_rows, X_all, y_all, val_idx, test_idx) -> pd.DataFrame:
    val_y = y_all[val_idx]
    test_y = y_all[test_idx]
    val_base = float(val_y.mean()) if val_y.size else np.nan
    test_base = float(test_y.mean()) if test_y.size else np.nan

    scored = []
    for r in rule_rows:
        conds = r["conds"]
        m_val = _apply_conditions(X_all.iloc[val_idx], conds)
        m_test = _apply_conditions(X_all.iloc[test_idx], conds)
        val_support = int(m_val.sum())
        test_support = int(m_test.sum())
        val_pos = int(val_y[m_val].sum()) if val_support else 0
        test_pos = int(test_y[m_test].sum()) if test_support else 0
        val_precision = float(val_pos / val_support) if val_support else 0.0
        test_precision = float(test_pos / test_support) if test_support else 0.0
        val_lift = (val_precision / val_base) if val_base and val_base > 0 else np.nan
        test_lift = (test_precision / test_base) if test_base and test_base > 0 else np.nan
        scored.append(
            {
                "tree_id": r["tree_id"],
                "leaf_id": r["leaf_id"],
                "train_support": r["train_support"],
                "leaf_pos_prob": r["leaf_pos_prob"],
                "val_support": val_support,
                "val_pos": val_pos,
                "val_precision": val_precision,
                "val_lift": val_lift,
                "test_support": test_support,
                "test_pos": test_pos,
                "test_precision": test_precision,
                "test_lift": test_lift,
                "rule_text": r["rule_text"],
            }
        )

    out = pd.DataFrame(scored)
    if out.empty:
        return out
    out = out[out["val_support"] >= int(MIN_VAL_SUPPORT)].copy()
    out = out[out["val_precision"] >= float(MIN_VAL_CONFIDENCE)].copy()
    # Selection is validation-only: test_precision/test_support/test_lift are kept in the
    # frame for post-hoc reporting but must never be read by ranking, filtering, or sampling.
    out = out.sort_values(
        ["val_precision", "val_support", "val_lift"],
        ascending=False,
    ).reset_index(drop=True)
    return out


SELECTION_KEYS = ["val_precision", "val_support", "val_lift"]


def _sample_rules_for_prompt(
    rule_paths_df: pd.DataFrame,
    n_rules: int = 30,
    random_state: int = 42,
    top_k_pool: int = 2000,
    weighted: bool = True,
    min_val_precision: float = 0.0,
    max_val_precision: float = 0.98,
    min_val_support: int = 1,
    diversify_bins: int = 5,
) -> List[Dict]:
    assert not any("test" in c for c in SELECTION_KEYS), "test statistics in rule selection"
    if rule_paths_df.empty:
        return []

    pool = rule_paths_df.copy()
    pool = pool[pool["val_precision"].fillna(0.0) >= float(min_val_precision)]
    if max_val_precision is not None:
        pool = pool[pool["val_precision"].fillna(0.0) <= float(max_val_precision)]
    pool = pool[pool["val_support"].fillna(0.0) >= float(min_val_support)]
    if pool.empty:
        return []

    pool = pool.head(min(int(top_k_pool), len(pool))).copy()
    rng = np.random.default_rng(int(random_state))

    if weighted:
        w = (
            np.sqrt(pool["val_precision"].fillna(0.0).clip(lower=0.0))
            * np.log1p(pool["val_support"].fillna(0.0).clip(lower=0.0))
        )
        if "val_lift" in pool.columns:
            w = w * np.sqrt(pool["val_lift"].fillna(1.0).clip(lower=1.0))
        w = w.astype(float)
        pool = pool.assign(_w=(w / float(w.sum())) if float(w.sum()) > 0 else (1.0 / len(pool)))
    else:
        pool = pool.assign(_w=1.0 / len(pool))

    if int(diversify_bins) > 1 and "val_precision" in pool.columns:
        try:
            pool = pool.assign(
                _bin=pd.qcut(
                    pool["val_precision"].rank(method="first"),
                    q=min(int(diversify_bins), max(1, pool.shape[0])),
                    labels=False,
                    duplicates="drop",
                )
            )
        except Exception:
            pool = pool.assign(_bin=0)
    else:
        pool = pool.assign(_bin=0)

    bins = sorted(pool["_bin"].dropna().unique().tolist()) or [0]
    picks: List[int] = []
    target = min(int(n_rules), len(pool))
    by_bin = {b: pool[pool["_bin"] == b].copy() for b in bins}

    while len(picks) < target:
        progressed = False
        for b in bins:
            sub = by_bin[b]
            if sub.empty or len(picks) >= target:
                continue
            p = sub["_w"].to_numpy().astype(float)
            ps = float(p.sum())
            if not np.isfinite(ps) or ps <= 0.0:
                p = np.full_like(p, 1.0 / max(1, p.size), dtype=float)
            else:
                p = p / ps
            idx = int(rng.choice(sub.index.to_numpy(), p=p))
            picks.append(idx)
            by_bin[b] = sub.drop(index=idx)
            progressed = True
        if not progressed:
            break

    sampled = pool.loc[picks].copy().drop(columns=[c for c in ["_w", "_bin"] if c in pool.columns])
    sampled = sampled.sort_values(["val_precision", "val_support"], ascending=[False, False]).reset_index(drop=True)

    rules = []
    for i, row in sampled.iterrows():
        rules.append(
            {
                "candidate_id": f"rf_rule_{i+1:03d}",
                "based_on_rule_ids": "[]",
                "mutation_type": "rf_sampled_path",
                "sql_predicate": row["rule_text"],
                "predicate_struct": "[]",
                "val_support": int(row["val_support"]),
                "val_precision": float(row["val_precision"]),
                "test_support_eval": int(row["test_support"]),
                "relaxed": False,
                "relax_factor": 1.0,
                "source": "rf_sampled",
                "generation_order": i,
            }
        )
    return rules


def _cat_codes(s: pd.Series) -> np.ndarray:
    # Integer codes over the sorted distinct values. RF rules are mined on these codes, so rule
    # evaluation must expose the same codes under the same column names.
    return pd.factorize(s, sort=True)[0].astype("int32")


def _prepare_rf_training_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Prevent target/split leakage into RF-derived features.
    leakage_cols = [
        "is_red_team",
        "auth_id",
        "g_y",
        "g_train",
        "g_val",
        "g_test",
        # Avoid identifier/time memorization that can make RF trivially separable.
        "source_user_id",
        "destination_user_id",
        "source_computer_id",
        "destination_computer_id",
        "time",
        "auth_time_int",
    ]
    x = df.drop(columns=leakage_cols, errors="ignore").copy()
    for c in x.select_dtypes(include=["object", "string", "category"]).columns:
        x[c] = _cat_codes(x[c])
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0)
    return x


def train_rf_and_build_sources(df: pd.DataFrame, graph: HeteroData, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, str, Dict]:
    auth_ids = graph["auth"].auth_id.view(-1).cpu().numpy().astype(np.int64)
    g_train = graph["auth"].train_mask.view(-1).cpu().numpy().astype(bool)
    g_val = graph["auth"].val_mask.view(-1).cpu().numpy().astype(bool)
    g_test = graph["auth"].test_mask.view(-1).cpu().numpy().astype(bool)
    g_y = graph["auth"].y.view(-1).cpu().numpy().astype(np.int64)

    split_df = pd.DataFrame(
        {
            "auth_id": auth_ids,
            "g_y": g_y,
            "g_train": g_train,
            "g_val": g_val,
            "g_test": g_test,
        }
    )

    base = df.copy()
    base["auth_id"] = pd.to_numeric(base["auth_id"], errors="coerce")
    base = base.dropna(subset=["auth_id"]).copy()
    base["auth_id"] = base["auth_id"].astype(np.int64)

    merged = base.merge(split_df, on="auth_id", how="inner")
    merged = merged.sort_values("auth_id").reset_index(drop=True)

    X_df = _prepare_rf_training_matrix(merged)
    y = merged["g_y"].to_numpy(dtype=np.int64)
    train_idx = np.where(merged["g_train"].to_numpy(dtype=bool))[0]
    val_idx = np.where(merged["g_val"].to_numpy(dtype=bool))[0]
    test_idx = np.where(merged["g_test"].to_numpy(dtype=bool))[0]
    print(f"[split-check] train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,} | "
          f"train&val={len(np.intersect1d(train_idx, val_idx))} train&test={len(np.intersect1d(train_idx, test_idx))} "
          f"val&test={len(np.intersect1d(val_idx, test_idx))}", flush=True)
    if len(np.intersect1d(train_idx, val_idx)) or len(np.intersect1d(train_idx, test_idx)) or len(np.intersect1d(val_idx, test_idx)):
        raise RuntimeError("LANL train/val/test masks overlap")
    print(f"[rule-selection] ranking/filter/sampling keys={SELECTION_KEYS} (no test statistics); "
          f"RF fit on train only; test_support_eval is written for reporting and never read back", flush=True)

    rf_params = dict(RF_PARAMS)
    rf_params["random_state"] = int(seed)
    clf = RandomForestClassifier(**rf_params)
    clf.fit(X_df.iloc[train_idx], y[train_idx])

    rf_score_all = np.zeros(len(df), dtype=np.float32)
    proba = clf.predict_proba(X_df)[:, 1].astype(np.float32)
    idx_by_auth = pd.Series(np.arange(len(df)), index=df["auth_id"].to_numpy(dtype=np.int64))
    pos = idx_by_auth.reindex(merged["auth_id"].to_numpy(dtype=np.int64)).to_numpy()
    valid = ~pd.isna(pos)
    rf_score_all[pos[valid].astype(int)] = proba[valid]

    # MLP-as-detector (E1): same supervision/features as RF, minus the tree ensemble.
    mlp_scaler = StandardScaler().fit(X_df.iloc[train_idx])
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN,
        activation="relu",
        alpha=MLP_ALPHA,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=int(seed),
    )
    mlp_clf.fit(mlp_scaler.transform(X_df.iloc[train_idx]), y[train_idx])
    mlp_proba = mlp_clf.predict_proba(mlp_scaler.transform(X_df))[:, 1].astype(np.float32)
    mlp_score_all = np.zeros(len(df), dtype=np.float32)
    mlp_score_all[pos[valid].astype(int)] = mlp_proba[valid]

    selected_tree_ids = list(range(min(PATH_TOP_TREES, len(clf.estimators_))))
    raw_rule_rows = []
    feature_names = list(X_df.columns)
    for tid in selected_tree_ids:
        raw_rule_rows.extend(
            _extract_tree_rule_candidates(
                clf.estimators_[tid],
                tree_id=tid,
                feature_names=feature_names,
                max_rules_per_tree=PATHS_PER_TREE,
            )
        )

    rule_paths_df = _score_rules(raw_rule_rows, X_df, y, val_idx, test_idx)
    sampled_rules = _sample_rules_for_prompt(
        rule_paths_df,
        n_rules=SAMPLED_RULE_COUNT,
        random_state=int(seed),
        top_k_pool=2000,
        weighted=True,
        min_val_precision=0.0,
        max_val_precision=0.98,
        min_val_support=1,
        diversify_bins=5,
    )

    rf_rules_df = pd.DataFrame(sampled_rules)
    if rf_rules_df.empty:
        rf_rules_df = pd.DataFrame(
            columns=[
                "candidate_id",
                "based_on_rule_ids",
                "mutation_type",
                "sql_predicate",
                "predicate_struct",
                "test_support_eval",
                "relaxed",
                "relax_factor",
                "source",
                "generation_order",
            ]
        )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rf_rules_df.to_csv(RF_SAMPLED_RULES_CSV, index=False)
    rule_ctx = {"X_df": X_df, "y": y, "val_idx": val_idx, "sampled_rules": sampled_rules,
                "categories": {c: sorted(merged[c].unique().tolist())
                               for c in merged.select_dtypes(include=["object", "string", "category"]).columns
                               if c in X_df.columns}}
    return rf_score_all, mlp_score_all, RF_SAMPLED_RULES_CSV, rule_ctx


# --------------------
# Per-run LLM rule generation (LANL_LLM_RULES=per_run)
# --------------------
def _parse_llm(text: str) -> List[Dict]:
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    s = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        out = json.loads(s)
        return out if isinstance(out, list) else []
    except Exception:
        pass
    start, end = s.find("["), s.rfind("]")
    if start != -1 and end > start:
        try:
            out = json.loads(s[start:end + 1])
            if isinstance(out, list):
                return out
        except Exception:
            pass
    # Truncated output (generation hit max_new_tokens mid-array): keep every complete
    # top-level object and drop the unfinished tail.
    dec, out, i = json.JSONDecoder(), [], (s.find("[") + 1 if "[" in s else 0)
    while True:
        i = s.find("{", i)
        if i == -1:
            break
        try:
            obj, i = dec.raw_decode(s, i)
        except ValueError:
            break
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _referenced_columns(pred: str) -> set:
    pred = re.sub(r"'[^']*'", "", str(pred))
    return {t for t in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()", pred) if t.upper() not in SQL_KEYWORDS}


def _llm_schema(X_df: pd.DataFrame, categories: Dict[str, List[str]]) -> str:
    lines = ["Table: features_table  (one row per LANL authentication event)", "Columns:"]
    for c in X_df.columns:
        if c in categories:
            codes = ", ".join(f"{i}={v}" for i, v in enumerate(categories[c]))
            lines.append(f"  {c:28s} INTEGER  -- category code: {codes}")
        else:
            kind = "FLOAT" if pd.api.types.is_float_dtype(X_df[c]) else "INTEGER"
            lines.append(f"  {c:28s} {kind:8s} -- range {X_df[c].min():g} to {X_df[c].max():g}")
    return "\n".join(lines)


def generate_llm_rules_per_run(rule_ctx: Dict, seed: int, out_dir: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    X_df, y, val_idx = rule_ctx["X_df"], rule_ctx["y"], rule_ctx["val_idx"]
    schema = _llm_schema(X_df, rule_ctx["categories"])
    seed_rules = [{"rule_id": r["candidate_id"], "sql_predicate": r["sql_predicate"],
                   "val_precision": round(float(r["val_precision"]), 4), "val_support": int(r["val_support"])}
                  for r in rule_ctx["sampled_rules"][:LLM_N_SEED_RULES]]
    seed_text = json.dumps(seed_rules, indent=2)

    tok = AutoTokenizer.from_pretrained(LLM_MODEL_ID, token=os.environ.get("HF_TOKEN"))
    total_gib = int(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID, token=os.environ.get("HF_TOKEN"),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
        ),
        device_map="auto", max_memory={0: f"{max(4, int(total_gib * 0.9))}GiB", "cpu": "64GiB"},
        low_cpu_mem_usage=True,
    )
    pipe = pipeline("text-generation", model=llm, tokenizer=tok)
    outputs = pipe(
        [
            {"role": "system", "content": (
                "You write SQL WHERE-clause predicates for a LANL authentication anomaly detector. "
                "The positives are red-team authentication events. "
                "You receive RF seed rules and must generate refined or novel predicates. "
                "Return only a valid JSON array. No prose. Use only the columns listed in SCHEMA; "
                "categorical columns are integer codes as listed. Keep predicates simple (1-3 conditions)."
            )},
            {"role": "user", "content": (
                f"SCHEMA:\n{schema}\n\n"
                f"SEED RULES (precision/support measured on the validation split):\n{seed_text}\n\n"
                "Generate 15 SQL WHERE predicates. "
                "Each object must have keys: "
                "\"candidate_id\" (string), \"sql_predicate\" (SQL), "
                "\"mutation_type\" (tighten|relax|combine|novel), \"rationale\" (one sentence). "
                "Return only the JSON array."
            )},
        ],
        **LLM_GEN_KWARGS,
    )
    response_text = outputs[0]["generated_text"][-1]["content"]
    del llm, pipe, tok, outputs
    torch.cuda.empty_cache()

    cands = _parse_llm(response_text)
    n_parsed = len(cands)
    truncated = not response_text.strip().strip("`").strip().endswith("]")
    if truncated:
        print(f"  [llm-rules] response did not close its JSON array (max_new_tokens={LLM_GEN_KWARGS['max_new_tokens']}); "
              f"kept the {n_parsed} complete objects")
    avail = set(X_df.columns)
    val_df = X_df.iloc[val_idx].reset_index(drop=True).assign(_row=np.arange(len(val_idx)))
    val_y = y[val_idx]
    con = duckdb.connect(":memory:")
    con.register("features_table", val_df)

    kept, rejected, sql_errors, no_val_support, seen = [], [], [], [], set()
    for i, c in enumerate(cands):
        pred = str(c.get("sql_predicate", "") if isinstance(c, dict) else "").strip()
        cid = str(c.get("candidate_id", f"llm_{i:03d}") if isinstance(c, dict) else f"llm_{i:03d}")
        if not pred or pred in seen:
            continue
        seen.add(pred)
        missing = sorted(_referenced_columns(pred) - avail)
        if missing:
            rejected.append({"candidate_id": cid, "sql_predicate": pred, "missing_columns": missing})
            continue
        try:
            rows = con.execute(f"SELECT _row FROM features_table WHERE ({pred})").fetchdf()["_row"].to_numpy()
        except Exception as e:
            sql_errors.append({"candidate_id": cid, "sql_predicate": pred, "error": str(e)[:200]})
            continue
        sup, pos = int(len(rows)), int(val_y[rows].sum()) if len(rows) else 0
        prec = float(pos / sup) if sup else 0.0
        if sup < int(MIN_VAL_SUPPORT) or prec < float(MIN_VAL_CONFIDENCE):
            no_val_support.append({"candidate_id": cid, "sql_predicate": pred, "val_support": sup})
            continue
        kept.append({"candidate_id": cid, "sql_predicate": pred,
                     "mutation_type": c.get("mutation_type", ""), "val_support": sup,
                     "val_pos": pos, "val_precision": prec})
    con.close()
    print(f"  [llm-rules] parsed={n_parsed} rejected(missing columns)={len(rejected)} "
          f"sql_errors={len(sql_errors)} dropped(val_support<{MIN_VAL_SUPPORT})={len(no_val_support)} kept={len(kept)}")
    for r in rejected:
        print(f"    rejected {r['candidate_id']}: missing {r['missing_columns']}")
    fallback = not kept
    if fallback:
        print("  [llm-rules] no candidate survived -- using the RF seed rules as fallback.")
        kept = [{"candidate_id": r["rule_id"], "sql_predicate": r["sql_predicate"], "mutation_type": "seed_fallback",
                 "val_support": r["val_support"], "val_precision": r["val_precision"]} for r in seed_rules]

    os.makedirs(out_dir, exist_ok=True)
    rules_csv = os.path.join(out_dir, "llm_parsed_rules_df.csv")
    pd.DataFrame(kept).to_csv(rules_csv, index=False)
    with open(os.path.join(out_dir, "rule_generation.json"), "w") as f:
        json.dump({
            "llm_rules_mode": LLM_RULES_MODE,
            "llm": {"model_id": LLM_MODEL_ID, "quantization": "4-bit nf4", **LLM_GEN_KWARGS},
            "seed": int(seed),
            "seed_note": ("do_sample=False (greedy decoding), so the LLM output does not depend on a sampling seed; "
                          "SEED sets the RF and the seed-rule sample"),
            "selection_split": f"val mask ({len(val_idx)} rows, {int(val_y.sum())} positives); test is not read",
            "schema": schema,
            "seed_rules_in_prompt": seed_rules,
            "response_truncated": truncated,
            "n_candidates_parsed": n_parsed,
            "n_rejected_missing_columns": len(rejected), "rejected": rejected,
            "n_sql_errors": len(sql_errors), "sql_errors": sql_errors,
            "n_dropped_no_val_support": len(no_val_support), "dropped_no_val_support": no_val_support,
            "n_kept": len(kept), "seed_fallback_used": fallback,
            "llm_raw_response": response_text,
        }, f, indent=2)
    return rules_csv


# --------------------
# GAE model + graph augmentation helpers
# --------------------
def _resolve_activation(name):
    key = str(name).strip().lower()
    if key == "relu":
        return nn.ReLU(), F.relu, "relu"
    if key == "gelu":
        return nn.GELU(), F.gelu, "gelu"
    if key in {"silu", "swish"}:
        return nn.SiLU(), F.silu, "silu"
    if key in {"leaky_relu", "leakyrelu", "lrelu"}:
        return nn.LeakyReLU(0.1), lambda x: F.leaky_relu(x, 0.1), "leaky_relu"
    raise ValueError(f"Unsupported activation: {name}")


class HeteroSAGEAE(nn.Module):
    def __init__(self, metadata, hidden_dim, latent_dim, out_dim, dropout=0.2, activation="relu"):
        super().__init__()
        self.dropout = float(dropout)
        self.edge_types = list(metadata[1])
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden_dim) for et in self.edge_types}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv((hidden_dim, hidden_dim), hidden_dim) for et in self.edge_types}, aggr="sum")
        self.enc_norm1 = nn.LayerNorm(hidden_dim)
        self.enc_norm2 = nn.LayerNorm(hidden_dim)
        self.decoder_act, self.enc_act, _ = _resolve_activation(activation)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.dec_conv1 = HeteroConv({et: SAGEConv((latent_dim, latent_dim), hidden_dim) for et in self.edge_types}, aggr="sum")
        self.dec_conv2 = HeteroConv({et: SAGEConv((hidden_dim, hidden_dim), hidden_dim) for et in self.edge_types}, aggr="sum")
        self.dec_norm1 = nn.LayerNorm(hidden_dim)
        self.dec_norm2 = nn.LayerNorm(hidden_dim)
        self.auth_out = nn.Linear(hidden_dim, out_dim)

        self.rel_diag = nn.ParameterDict({self._et_key(et): nn.Parameter(torch.ones(latent_dim)) for et in self.edge_types})

    @staticmethod
    def _et_key(et):
        return f"{et[0]}__{et[1]}__{et[2]}"

    def encode(self, x_dict, eid):
        h = self.conv1(x_dict, eid)
        h = {k: self.enc_norm1(self.enc_act(v)) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv2(h, eid)
        h = {k: self.enc_norm2(self.enc_act(v)) for k, v in h.items()}
        return {k: self.to_latent(v) for k, v in h.items()}

    def decode_features(self, z, eid):
        h = self.dec_conv1(z, eid)
        h = {k: self.dec_norm1(self.enc_act(v)) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self.dec_conv2(h, eid)
        h = {k: self.dec_norm2(self.enc_act(v)) for k, v in h.items()}
        return self.auth_out(h["auth"])

    def score_edges(self, z, et, ei):
        src_t, _, dst_t = et
        if ei.numel() == 0:
            return torch.empty(0, device=ei.device)
        d = self.rel_diag[self._et_key(et)]
        z_src = z[src_t][ei[0]]
        z_dst = z[dst_t][ei[1]]
        return (z_src * d * z_dst).sum(dim=1)

    def forward(self, x_dict, eid, return_latent=False):
        z = self.encode(x_dict, eid)
        recon = self.decode_features(z, eid)
        if return_latent:
            return recon, z
        return recon


def structure_bce_loss(model, z, data_obj, neg_ratio=1.0, edge_index_dict=None):
    eid = edge_index_dict if edge_index_dict is not None else data_obj.edge_index_dict
    losses = []
    for et in data_obj.edge_types:
        if "red_team" in (et[0], et[2]):
            continue
        ei = eid.get(et, data_obj[et].edge_index)
        if ei.numel() == 0 or ei.size(1) == 0:
            continue
        pos_logits = model.score_edges(z, et, ei)
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        n_neg = max(1, int(neg_ratio * ei.size(1)))
        src_n = int(data_obj[et[0]].num_nodes)
        dst_n = int(data_obj[et[2]].num_nodes)
        if src_n == 0 or dst_n == 0:
            continue
        neg_ei = torch.stack(
            [
                torch.randint(0, src_n, (n_neg,), device=ei.device),
                torch.randint(0, dst_n, (n_neg,), device=ei.device),
            ]
        )
        neg_logits = model.score_edges(z, et, neg_ei)
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        losses.append(0.5 * (pos_loss + neg_loss))
    if not losses:
        ref = z["auth"]
        return torch.zeros((), dtype=ref.dtype, device=ref.device)
    return torch.stack(losses).mean()


def auth_structure_error(model, z, data_obj, edge_index_dict=None):
    eid = edge_index_dict if edge_index_dict is not None else data_obj.edge_index_dict
    n_auth = int(data_obj["auth"].num_nodes)
    dev = z["auth"].device
    err_sum = torch.zeros(n_auth, dtype=torch.float32, device=dev)
    err_cnt = torch.zeros(n_auth, dtype=torch.float32, device=dev)
    for et in data_obj.edge_types:
        if "red_team" in (et[0], et[2]):
            continue
        if "auth" not in (et[0], et[2]):
            continue
        ei = eid.get(et, data_obj[et].edge_index)
        if ei.numel() == 0 or ei.size(1) == 0:
            continue
        edge_err = F.softplus(-model.score_edges(z, et, ei))
        if et[0] == "auth":
            err_sum.index_add_(0, ei[0], edge_err)
            err_cnt.index_add_(0, ei[0], torch.ones_like(edge_err))
        if et[2] == "auth":
            err_sum.index_add_(0, ei[1], edge_err)
            err_cnt.index_add_(0, ei[1], torch.ones_like(edge_err))
    return err_sum / torch.clamp(err_cnt, min=1.0)


def sampled_pairwise_logistic_rank_loss(scores, labels, max_pairs=8192):
    scores = scores.view(-1)
    labels = labels.view(-1)
    pos_idx = torch.where(labels > 0.5)[0]
    neg_idx = torch.where(labels <= 0.5)[0]
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return scores.new_zeros(()), 0
    n_pairs = int(max(1, min(int(max_pairs), int(pos_idx.numel()) * int(neg_idx.numel()))))
    pos_samp = pos_idx[torch.randint(0, pos_idx.numel(), (n_pairs,), device=scores.device)]
    neg_samp = neg_idx[torch.randint(0, neg_idx.numel(), (n_pairs,), device=scores.device)]
    return F.softplus(-(scores[pos_samp] - scores[neg_samp])).mean(), n_pairs


_DISABLED_TEMPORAL_REV = {
    ("auth", "to_proc", "proc"),
    ("auth", "to_flows", "flows"),
    ("auth", "to_dns", "dns"),
    ("auth", "to_red_team", "red_team"),
    ("auth", "to_source_user", "user"),
    ("auth", "to_destination_user", "user"),
    ("auth", "to_source_computer", "computer"),
    ("auth", "to_destination_computer", "computer"),
    ("proc", "to_user", "user"),
    ("proc", "to_computer", "computer"),
    ("flows", "to_source_computer", "computer"),
    ("flows", "to_destination_computer", "computer"),
    ("dns", "to_source_computer", "computer"),
    ("red_team", "to_user", "user"),
    ("red_team", "to_source_computer", "computer"),
    ("red_team", "to_destination_computer", "computer"),
}
_CAUSAL_INTO_AUTH = {
    ("proc", "rev_to_proc", "auth"): ("proc", "auth"),
    ("flows", "rev_to_flows", "auth"): ("flows", "auth"),
    ("dns", "rev_to_dns", "auth"): ("dns", "auth"),
    ("red_team", "rev_to_red_team", "auth"): ("red_team", "auth"),
}
_STATIC_TO_EVENT = {
    ("user", "rev_to_source_user", "auth"),
    ("user", "rev_to_destination_user", "auth"),
    ("computer", "rev_to_source_computer", "auth"),
    ("computer", "rev_to_destination_computer", "auth"),
    ("user", "rev_to_user", "proc"),
    ("computer", "rev_to_computer", "proc"),
    ("computer", "rev_to_source_computer", "flows"),
    ("computer", "rev_to_destination_computer", "flows"),
    ("computer", "rev_to_source_computer", "dns"),
    ("user", "rev_to_user", "red_team"),
    ("computer", "rev_to_source_computer", "red_team"),
    ("computer", "rev_to_destination_computer", "red_team"),
}


def build_causal_edge_index_dict(data, target_auth_mask=None, drop_red_team=True):
    out = {}
    for et in data.edge_types:
        src_t, _, dst_t = et
        ei = data[et].edge_index
        if drop_red_team and "red_team" in (src_t, dst_t):
            out[et] = ei[:, :0]
            continue
        if et in _DISABLED_TEMPORAL_REV:
            out[et] = ei[:, :0]
            continue
        if et in _CAUSAL_INTO_AUTH:
            src_name, dst_name = _CAUSAL_INTO_AUTH[et]
            src_time = data[src_name].time.to(ei.device)
            dst_time = data[dst_name].time.to(ei.device)
            keep = (src_time[ei[0]] >= 0) & (dst_time[ei[1]] >= 0) & (src_time[ei[0]] <= dst_time[ei[1]])
            if target_auth_mask is not None and dst_t == "auth":
                keep &= target_auth_mask.to(ei.device)[ei[1]]
            out[et] = ei[:, keep]
            continue
        if et in _STATIC_TO_EVENT:
            keep = torch.ones(ei.size(1), dtype=torch.bool, device=ei.device)
            if target_auth_mask is not None and dst_t == "auth":
                keep &= target_auth_mask.to(ei.device)[ei[1]]
            out[et] = ei[:, keep]
            continue
        out[et] = ei[:, :0]
    return out


def _load_rules_df(path: str) -> pd.DataFrame:
    rules_df = pd.read_csv(path)
    if "sql_predicate" not in rules_df.columns:
        raise RuntimeError(f"{path} missing sql_predicate column")
    if "candidate_id" not in rules_df.columns:
        rules_df = rules_df.copy()
        rules_df["candidate_id"] = [f"rule_{i:03d}" for i in range(len(rules_df))]
    rules_df = (
        rules_df[rules_df["sql_predicate"].notna()]
        .assign(sql_predicate=lambda d: d["sql_predicate"].astype(str).str.strip())
        .query("sql_predicate != ''")
        .drop_duplicates(subset=["sql_predicate"])
        .reset_index(drop=True)
    )
    return rules_df


def _rule_matrix_for_auth_ids(base_df: pd.DataFrame, auth_ids_np: np.ndarray, rules_csv: str) -> Tuple[np.ndarray, List[str]]:
    rules_df = _load_rules_df(rules_csv)

    drop_cols = [c for c in ["is_red_team", "auth_id", "time"] if c in base_df.columns]
    x_raw = base_df.drop(columns=drop_cols).copy()

    bool_cols = x_raw.select_dtypes(include=["bool"]).columns.tolist()
    cat_cols = x_raw.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = [c for c in x_raw.columns if c not in cat_cols + bool_cols]

    if bool_cols:
        x_raw[bool_cols] = x_raw[bool_cols].astype(np.int8)
    if num_cols:
        x_raw[num_cols] = x_raw[num_cols].apply(pd.to_numeric, errors="coerce")
        x_raw[num_cols] = x_raw[num_cols].fillna(x_raw[num_cols].median(numeric_only=True)).fillna(0.0)
    if cat_cols:
        x_raw[cat_cols] = x_raw[cat_cols].fillna("<NA>").astype(str)

    x_eval_df = pd.get_dummies(x_raw, columns=cat_cols, dummy_na=False)
    bcols = x_eval_df.select_dtypes(include=["bool"]).columns.tolist()
    if bcols:
        x_eval_df[bcols] = x_eval_df[bcols].astype(np.int8)
    # RF rules are mined on integer category codes (_prepare_rf_training_matrix); without these
    # columns every RF rule with a categorical condition evaluated against a zero-filled column.
    for c in cat_cols:
        x_eval_df[c] = _cat_codes(base_df[c])

    # Reject (rather than zero-fill) rules that reference a column the evaluation table lacks.
    avail = set(x_eval_df.columns)
    keep = []
    for pred in rules_df["sql_predicate"].astype(str):
        missing = sorted(_referenced_columns(pred) - avail)
        keep.append(not missing)
        if missing:
            print(f"    [rules] rejected {pred[:80]!r}: missing {missing}")
    print(f"  [rules] {os.path.basename(rules_csv)}: rejected {keep.count(False)}/{len(keep)} rules "
          f"referencing columns missing from the rule-evaluation table")
    rules_df = rules_df[keep].reset_index(drop=True)

    feature_df = pd.concat(
        [
            base_df[["auth_id"]].rename(columns={"auth_id": "auth_event_id"}).reset_index(drop=True),
            x_eval_df.reset_index(drop=True),
        ],
        axis=1,
    )

    eval_df = pd.DataFrame({"auth_event_id": base_df["auth_id"].to_numpy(dtype=np.int64, copy=False)})
    con = duckdb.connect(":memory:")
    con.register("features_table", feature_df)

    used_cols = {"auth_event_id"}
    for i, row in rules_df.reset_index(drop=True).iterrows():
        raw_name = row.get("candidate_id", f"rule_{i:03d}")
        col_name = re.sub(r"[^0-9A-Za-z_]+", "_", str(raw_name)).strip("_") or f"rule_{i:03d}"
        base_n = col_name
        suffix = 1
        while col_name in used_cols:
            col_name = f"{base_n}_{suffix}"
            suffix += 1
        used_cols.add(col_name)

        eval_df[col_name] = np.int8(0)
        pred = row.get("sql_predicate", None)
        if not isinstance(pred, str) or not pred.strip():
            continue
        try:
            matched = con.execute(f"SELECT auth_event_id FROM features_table WHERE ({pred})").fetchdf()
            if not matched.empty:
                mids = pd.to_numeric(matched["auth_event_id"], errors="coerce").dropna().astype(np.int64).unique()
                mask = eval_df["auth_event_id"].isin(mids)
                eval_df.loc[mask, col_name] = np.int8(1)
        except Exception as e:
            print(f"    [rules] SQL error in {raw_name}: {str(e)[:160]}")
            continue
    con.close()

    rule_cols = [c for c in eval_df.columns if c != "auth_event_id"]
    print(f"  [rules] {len(rule_cols)} rule columns; matches per rule = "
          f"{[int(eval_df[c].sum()) for c in rule_cols]}")
    _ids = eval_df["auth_event_id"]
    if _ids.duplicated().any() or not np.isin(auth_ids_np, _ids.to_numpy()).all():
        raise RuntimeError("rule matrix: graph auth_ids are not a subset of the unique auth_ids in base_df")
    aligned = eval_df.set_index("auth_event_id").reindex(auth_ids_np).astype(np.float32)
    mat = aligned.to_numpy(dtype=np.float32)
    return mat, rule_cols


def _align_scores_to_auth_ids(auth_ids_np: np.ndarray, base_df: pd.DataFrame, score_all: np.ndarray) -> np.ndarray:
    idx_by_auth = pd.Series(np.arange(len(base_df)), index=base_df["auth_id"].to_numpy(dtype=np.int64))
    pos = idx_by_auth.reindex(auth_ids_np).to_numpy()
    if idx_by_auth.index.duplicated().any() or pd.isna(pos).any():
        raise RuntimeError("score alignment: graph auth_ids are not a subset of the unique auth_ids in base_df")
    return score_all[pos.astype(int)].astype(np.float32)


def augment_graph_for_variant(base_graph: HeteroData, base_df: pd.DataFrame, variant: str, rf_score_all: np.ndarray,
                              llm_rules_csv: str = LLM_RULES_CSV) -> HeteroData:
    g = copy.deepcopy(base_graph)

    if variant in {"llm", "rf_sampled"}:
        rules_csv = llm_rules_csv if variant == "llm" else RF_SAMPLED_RULES_CSV
        auth_ids_np = g["auth"].auth_id.view(-1).cpu().numpy().astype(np.int64)
        rule_matrix, rule_cols = _rule_matrix_for_auth_ids(base_df, auth_ids_np, rules_csv)
        x_old = g["auth"].x.float()
        g["auth"].x = torch.cat([x_old, torch.from_numpy(rule_matrix)], dim=1)
        old_fc = list(g["auth"].feature_cols) if hasattr(g["auth"], "feature_cols") else [f"f_{i}" for i in range(x_old.size(1))]
        g["auth"].feature_cols = old_fc + [f"rule_input__{c}" for c in rule_cols]
        return g

    if variant == "rf_decision":
        auth_ids_np = g["auth"].auth_id.view(-1).cpu().numpy().astype(np.int64)
        score_aligned = _align_scores_to_auth_ids(auth_ids_np, base_df, rf_score_all)
        x_old = g["auth"].x.float()
        score_t = torch.from_numpy(score_aligned).view(-1, 1)
        g["auth"].x = torch.cat([x_old, score_t], dim=1)
        old_fc = list(g["auth"].feature_cols) if hasattr(g["auth"], "feature_cols") else [f"f_{i}" for i in range(x_old.size(1))]
        g["auth"].feature_cols = old_fc + ["rf_decision_score"]
        return g

    raise ValueError(f"Unknown variant source: {variant}")


def run_gae_single(g: HeteroData, seed: int, lambda_struct_override: Optional[float]) -> np.ndarray:
    set_seed(seed)
    g = g.to(DEVICE)

    hp = GAE_HP
    y = g["auth"].y.view(-1).float()
    tr_mask = g["auth"].train_mask.view(-1).bool()
    va_mask = g["auth"].val_mask.view(-1).bool()
    te_mask = g["auth"].test_mask.view(-1).bool()
    normal_tr = tr_mask & (y < 0.5)

    x = g["auth"].x
    mu_n = x[normal_tr].mean(dim=0, keepdim=True)
    sig_n = x[normal_tr].std(dim=0, keepdim=True)
    sig_n = torch.where(sig_n < 1e-6, torch.ones_like(sig_n), sig_n)
    g["auth"].x = (x - mu_n) / sig_n

    tr_ei = build_causal_edge_index_dict(g, target_auth_mask=tr_mask.to(DEVICE), drop_red_team=True)
    va_ei = build_causal_edge_index_dict(g, target_auth_mask=va_mask.to(DEVICE), drop_red_team=True)
    te_ei = build_causal_edge_index_dict(g, target_auth_mask=te_mask.to(DEVICE), drop_red_team=True)
    tr_ei = {k: v.to(DEVICE) for k, v in tr_ei.items()}
    va_ei = {k: v.to(DEVICE) for k, v in va_ei.items()}
    te_ei = {k: v.to(DEVICE) for k, v in te_ei.items()}

    ls_ab = float(hp["lambda_struct"] if lambda_struct_override is None else lambda_struct_override)
    lr_ab = float(hp["lambda_rank"])

    model = HeteroSAGEAE(
        g.metadata(),
        hp["hidden_dim"],
        hp["latent_dim"],
        int(g["auth"].x.size(1)),
        hp["dropout"],
        hp["activation"],
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    val_y = y[va_mask].detach().cpu().numpy().astype(np.int8)
    val_has_both = (val_y.sum() > 0) and (val_y.sum() < val_y.size)
    val_normal = va_mask & (y < 0.5)
    if int(val_normal.sum()) == 0:
        val_normal = normal_tr

    best_state = None
    best_ap = -1.0
    bad = 0

    for _ep in range(1, hp["epochs"] + 1):
        model.train()
        opt.zero_grad()

        recon, z_tr = model(g.x_dict, tr_ei, return_latent=True)
        feat_loss = F.mse_loss(recon[normal_tr], g["auth"].x[normal_tr])
        struct_loss = structure_bce_loss(model, z_tr, g, neg_ratio=1.0, edge_index_dict=tr_ei)

        feat_node = ((recon - g["auth"].x) ** 2).mean(dim=1)
        struct_node = auth_structure_error(model, z_tr, g, edge_index_dict=tr_ei)
        stats_mask = normal_tr if int(normal_tr.sum()) > 0 else tr_mask
        fr = feat_node[stats_mask].detach()
        sr = struct_node[stats_mask].detach()
        fmu, fsd = fr.mean(), torch.clamp(fr.std(unbiased=False), min=1e-8)
        smu, ssd = sr.mean(), torch.clamp(sr.std(unbiased=False), min=1e-8)

        comb = (hp["score_feat_alpha"] * ((feat_node - fmu) / fsd)) + (hp["score_struct_alpha"] * ((struct_node - smu) / ssd))

        rank_loss = torch.zeros((), dtype=feat_loss.dtype, device=DEVICE)
        if lr_ab > 0.0:
            tr_idx = torch.where(tr_mask)[0]
            if tr_idx.numel() > 1:
                rank_loss, _ = sampled_pairwise_logistic_rank_loss(comb[tr_idx], y[tr_idx], max_pairs=hp["rank_max_pairs"])

        loss = feat_loss + ls_ab * struct_loss + lr_ab * rank_loss
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            recon_val, z_val = model(g.x_dict, va_ei, return_latent=True)

        if val_has_both:
            fv = ((recon_val - g["auth"].x) ** 2).mean(dim=1)
            sv = auth_structure_error(model, z_val, g, edge_index_dict=va_ei)
            vs = (hp["score_feat_alpha"] * ((fv - fmu) / fsd)) + (hp["score_struct_alpha"] * ((sv - smu) / ssd))
            val_ap = float(average_precision_score(val_y, vs[va_mask].detach().cpu().numpy()))
        else:
            val_ap = -float(F.mse_loss(recon_val[val_normal], g["auth"].x[val_normal]).item())

        if val_ap > best_ap + 1e-8:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= hp["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        recon_tr, z_tr_f = model(g.x_dict, tr_ei, return_latent=True)
        recon_te, z_te_f = model(g.x_dict, te_ei, return_latent=True)

    tr_m_np = tr_mask.detach().cpu().numpy().astype(bool)
    te_m_np = te_mask.detach().cpu().numpy().astype(bool)

    fe_tr = ((recon_tr - g["auth"].x) ** 2).mean(dim=1).detach().cpu().numpy()
    se_tr = auth_structure_error(model, z_tr_f, g, edge_index_dict=tr_ei).detach().cpu().numpy()
    nm = (normal_tr if int(normal_tr.sum()) > 0 else tr_mask).detach().cpu().numpy().astype(bool)
    fmu_ = float(np.mean(fe_tr[nm])) if nm.sum() > 0 else float(np.mean(fe_tr))
    fsd_ = max(float(np.std(fe_tr[nm])) if nm.sum() > 0 else float(np.std(fe_tr)), 1e-8)
    smu_ = float(np.mean(se_tr[nm])) if nm.sum() > 0 else float(np.mean(se_tr))
    ssd_ = max(float(np.std(se_tr[nm])) if nm.sum() > 0 else float(np.std(se_tr)), 1e-8)

    fe_te = ((recon_te - g["auth"].x) ** 2).mean(dim=1).detach().cpu().numpy()
    se_te = auth_structure_error(model, z_te_f, g, edge_index_dict=te_ei).detach().cpu().numpy()
    s_te = (hp["score_feat_alpha"] * ((fe_te - fmu_) / fsd_)) + (hp["score_struct_alpha"] * ((se_te - smu_) / ssd_))
    _ = tr_m_np
    return s_te[te_m_np]


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[init] device={DEVICE}")
    print(f"[init] partial_auroc_max_fpr={PARTIAL_AUROC_MAX_FPR}")
    print(f"[init] raw outputs={RAW_RESULTS_DIR}")
    print(f"[init] seeds={SEEDS} mlp_hidden={MLP_HIDDEN} mlp_alpha={MLP_ALPHA} llm_rules={LLM_RULES_MODE}")
    run_config = {
        "seeds": SEEDS,
        "mlp_supervised_detector": {"hidden_layer_sizes": list(MLP_HIDDEN), "alpha": MLP_ALPHA,
                                    "source": os.environ.get("LANL_MLP_SOURCE", "default (untuned)")},
        "rad_hyperparameters_source": os.environ.get("LANL_RAD_HP_SOURCE"),
        "llm_rules": (
            f"cached {LLM_RULES_CSV}: selected and filtered with test labels/features in rule_mining.ipynb "
            "(contaminated; llm_* rows must not be reported)" if LLM_RULES_MODE == "cached" else
            f"per_run: regenerated each seed with {LLM_MODEL_ID} ({LLM_GEN_KWARGS}); RF seed rules and LLM "
            f"rules selected on the val split only; saved under {os.path.join(RESULTS_DIR, 'rules')}"
        ),
    }
    with open(os.path.join(RESULTS_DIR, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    base_df = pd.read_parquet(PARQUET_PATH)
    base_df = base_df.sort_values(["time", "auth_id"]).reset_index(drop=True)
    base_df["auth_id"] = pd.to_numeric(base_df["auth_id"], errors="coerce")
    base_df = base_df.dropna(subset=["auth_id"]).copy()
    base_df["auth_id"] = base_df["auth_id"].astype(np.int64)

    graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    auth = graph["auth"]
    y_true_test = auth.y.view(-1).cpu().numpy().astype(np.int8)[auth.test_mask.view(-1).cpu().numpy().astype(bool)]

    variants = [
        ("rad_llm_rules_no_bce", "llm", 0.0),
        ("llm_rules_with_bce", "llm", None),
        ("rf_sampled_rules_no_bce", "rf_sampled", 0.0),
        ("rf_sampled_rules_with_bce", "rf_sampled", None),
        ("rf_decision_features_no_bce", "rf_decision", 0.0),
        ("rf_decision_features_with_bce", "rf_decision", None),
    ]

    variant_names = [v[0] for v in variants]
    known_names = variant_names + ["rf_supervised_detector", "mlp_supervised_detector"]
    rows: List[Dict] = []
    done_pairs = set()
    if os.path.exists(BY_SEED_CSV):
        try:
            prev = pd.read_csv(BY_SEED_CSV)
            need_cols = {"variant", "seed", "auroc", "auprc", "pauroc", "p@127"}
            if need_cols.issubset(set(prev.columns)):
                prev = prev[prev["variant"].isin(known_names)].copy()
                prev = prev.dropna(subset=["variant", "seed"]).copy()
                prev["seed"] = prev["seed"].astype(int)
                prev = prev.sort_values(["variant", "seed"]).drop_duplicates(subset=["variant", "seed"], keep="last")
                rows = prev.to_dict("records")
                done_pairs = set(zip(prev["variant"].astype(str), prev["seed"].astype(int)))
                if done_pairs:
                    max_done_seed = max(s for _, s in done_pairs)
                    print(f"[resume] loaded {len(done_pairs)} completed runs from {BY_SEED_CSV}; max completed seed={max_done_seed}")
            else:
                print(f"[resume] existing checkpoint missing expected columns; starting fresh")
        except Exception as e:
            print(f"[resume] failed to read checkpoint ({e}); starting fresh")

    t0 = time.time()

    for seed in SEEDS:
        print(f"\n[seed {seed}] starting")
        if all((vn, seed) in done_pairs for vn in variant_names + ["rf_supervised_detector", "mlp_supervised_detector"]):
            print(f"  [skip] all variants already completed for seed={seed}")
            continue

        # Rule construction (RF fit + rule mining/selection/sampling) is seed-dependent, so
        # the reported std across seeds captures rule-selection variance, not just GAE init variance.
        print(f"  [rf] training RF and building RF source artifacts for seed={seed}...")
        rf_score_all, mlp_score_all, rf_rules_csv, rule_ctx = train_rf_and_build_sources(base_df, graph, seed=seed)
        print(f"  [rf] sampled rule csv: {rf_rules_csv}")
        llm_rules_csv = LLM_RULES_CSV
        if LLM_RULES_MODE == "per_run" and any((vn, seed) not in done_pairs for vn, sk, _ in variants if sk == "llm"):
            llm_rules_csv = generate_llm_rules_per_run(
                rule_ctx, seed=seed, out_dir=os.path.join(RESULTS_DIR, "rules", f"seed_{seed:03d}"))
            print(f"  [llm-rules] per-run rule csv: {llm_rules_csv}")

        auth_ids_np = graph["auth"].auth_id.view(-1).cpu().numpy().astype(np.int64)
        test_mask_np = auth.test_mask.view(-1).cpu().numpy().astype(bool)

        rf_detector_name = "rf_supervised_detector"
        if (rf_detector_name, seed) not in done_pairs:
            rf_scores_aligned = _align_scores_to_auth_ids(auth_ids_np, base_df, rf_score_all)
            rf_scores_test = rf_scores_aligned[test_mask_np]
            rows.append(metrics_row(rf_detector_name, seed, y_true_test, rf_scores_test))
            done_pairs.add((rf_detector_name, seed))

        mlp_detector_name = "mlp_supervised_detector"
        if (mlp_detector_name, seed) not in done_pairs:
            mlp_scores_aligned = _align_scores_to_auth_ids(auth_ids_np, base_df, mlp_score_all)
            mlp_scores_test = mlp_scores_aligned[test_mask_np]
            rows.append(metrics_row(mlp_detector_name, seed, y_true_test, mlp_scores_test))
            done_pairs.add((mlp_detector_name, seed))

        for variant_name, source_kind, lambda_struct_override in variants:
            if (variant_name, seed) in done_pairs:
                print(f"  [skip] {variant_name} already completed for seed={seed}")
                continue
            print(f"  [run] {variant_name} | source={source_kind} | lambda_struct_override={lambda_struct_override}")
            g_aug = augment_graph_for_variant(graph, base_df, source_kind, rf_score_all, llm_rules_csv=llm_rules_csv)
            scores = run_gae_single(g_aug, seed=seed, lambda_struct_override=lambda_struct_override)
            rows.append(metrics_row(variant_name, seed, y_true_test, scores))
            os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
            np.savez_compressed(
                os.path.join(RAW_RESULTS_DIR, f"{variant_name}__seed_{seed:03d}.npz"),
                variant=np.array(variant_name),
                seed=np.array(seed, dtype=np.int64),
                y_true=y_true_test.astype(np.int8),
                y_score=np.asarray(scores, dtype=np.float32),
                partial_auroc_max_fpr=np.array(PARTIAL_AUROC_MAX_FPR, dtype=np.float32),
            )
            done_pairs.add((variant_name, seed))

        by_seed_df = pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)
        by_seed_df = by_seed_df.drop_duplicates(subset=["variant", "seed"], keep="last")
        by_seed_df.to_csv(BY_SEED_CSV, index=False)
        print(f"[seed {seed}] checkpoint saved -> {BY_SEED_CSV}")

    by_seed_df = pd.DataFrame(rows).sort_values(["variant", "seed"]).reset_index(drop=True)
    by_seed_df = by_seed_df.drop_duplicates(subset=["variant", "seed"], keep="last")
    mean_std_df = summarize(by_seed_df)
    by_seed_df.to_csv(BY_SEED_CSV, index=False)
    mean_std_df.to_csv(MEAN_STD_CSV, index=False)

    print("\n[done] LANL six ablations complete")
    print(f"[done] by-seed:  {BY_SEED_CSV}")
    print(f"[done] summary:  {MEAN_STD_CSV}")
    print(f"[done] elapsed:  {(time.time() - t0)/60.0:.1f} min")


if __name__ == "__main__":
    main()
