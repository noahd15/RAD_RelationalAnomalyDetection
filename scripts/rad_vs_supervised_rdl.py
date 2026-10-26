#!/usr/bin/env python3
"""
RAD vs Supervised RDL — Amazon User-Churn (At-Risk Filtering)
Trains both models on the same at-risk-filtered dataset and compares side-by-side.
"""

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
import copy, gc, json, math, os, re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP
from torch_geometric.seed import seed_everything
from torch_frame.data.stats import StatType
from torch_frame.config.text_embedder import TextEmbedderConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from path_helpers import amazon_data, append_results_jsonl
from eval_split import check_disjoint, eval_fingerprint

ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)

RUN_SEED = int(os.environ.get("SEED", "42"))
POS_DOWNSAMPLE_MODE = os.environ.get("POS_DOWNSAMPLE_MODE", "random").strip().lower()
# Exit right after the RF/MLP supervised-detector rows are scored, skipping graph
# encoding and GNN training entirely -- useful for getting just those two baselines
# without paying for the expensive (GPU, multi-hour) RAD/Supervised-RDL training.
RF_MLP_ONLY = os.environ.get("RF_MLP_ONLY", "0").strip() == "1"

IDENTITY_PROXY_FEATURES = {"customer": {"customer_name"}}

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — SETUP
# ═══════════════════════════════════════════════════════════════════════════
seed_everything(RUN_SEED)

if not torch.cuda.is_available():
    raise RuntimeError("GPU required but CUDA not available.")

torch.cuda.set_device(0)
device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD DATASET + GRAPH CACHE
# ═══════════════════════════════════════════════════════════════════════════
rdl_graph_cache_path = amazon_data("relbench_rdl_graph_cache.pt")
rdl_stats_cache_path = amazon_data("relbench_rdl_col_stats_cache.pt")


def _sanitize_graph(data):
    for ntype in data.node_types:
        node = data[ntype]
        if hasattr(node, "tf") and node.tf is not None and hasattr(node.tf, "feat_dict"):
            for key, tensor in node.tf.feat_dict.items():
                if torch.is_tensor(tensor) and tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    node.tf.feat_dict[key] = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def _mask_identity_proxy_features(data):
    masked = []
    for ntype, blocked_cols in IDENTITY_PROXY_FEATURES.items():
        if ntype not in data.node_types:
            continue
        node = data[ntype]
        if not hasattr(node, "tf") or node.tf is None or not hasattr(node.tf, "col_names_dict"):
            continue
        for col_type, col_names in list(node.tf.col_names_dict.items()):
            col_names = list(col_names)
            blocked_in_slot = [name for name in col_names if name in blocked_cols]
            if not blocked_in_slot:
                continue
            if len(blocked_in_slot) != len(col_names):
                raise RuntimeError(f"Cannot partially mask feature slot for {ntype}: {col_names}")
            feat = node.tf.feat_dict.get(col_type)
            values = getattr(feat, "values", None)
            if not torch.is_tensor(values):
                raise RuntimeError(f"Unsupported TensorFrame storage for {ntype}: {blocked_in_slot}")
            values.zero_()
            masked.extend(f"{ntype}.{name}" for name in blocked_in_slot)
    if masked:
        print(f"[rdl] masked identity-proxy features: {', '.join(masked)}")
    return data

if os.path.exists(rdl_graph_cache_path) and os.path.exists(rdl_stats_cache_path):
    rdl_data = _mask_identity_proxy_features(_sanitize_graph(torch.load(rdl_graph_cache_path, weights_only=False)))
    rdl_col_stats_dict = torch.load(rdl_stats_cache_path, weights_only=False)
    print(f"Loaded graph cache from disk.")
else:
    rdl_data = None
    rdl_col_stats_dict = None
    print("No cache found — will build graph from scratch.")

rdl_dataset = get_dataset("rel-amazon", download=True)
rdl_task    = get_task("rel-amazon", "user-churn", download=True)
rdl_db      = rdl_dataset.get_db()

rdl_train_table = rdl_task.get_table("train")
rdl_val_table   = rdl_task.get_table("val")
rdl_test_table  = rdl_task.get_table("test")

# Partition the validation table: an early portion drives early stopping and model selection,
# a disjoint portion is held out for reporting when RelBench hides test labels. Without this
# split, "held-out target entities" is not true of the code -- the same rows would both tune
# the model and score it. RelBench assigns a single static cutoff timestamp to the *entire*
# val split (all rows share one "timestamp" value here), so there is no per-row temporal signal
# to split on -- partition by row instead, via a seeded random permutation.
VAL_FIT_FRAC = float(os.environ.get("VAL_FIT_FRAC", "0.6"))
_val_df = rdl_val_table.df.reset_index(drop=True)
_val_perm = np.random.default_rng(RUN_SEED).permutation(len(_val_df))
_n_val_fit = int(round(VAL_FIT_FRAC * len(_val_df)))
_val_fit_idx = np.sort(_val_perm[:_n_val_fit])
_val_report_idx = np.sort(_val_perm[_n_val_fit:])
rdl_val_table.df = _val_df.iloc[_val_fit_idx].copy()
rdl_val_report_table = copy.deepcopy(rdl_val_table)
rdl_val_report_table.df = _val_df.iloc[_val_report_idx].copy()
assert set(rdl_val_table.df.index).isdisjoint(set(rdl_val_report_table.df.index)), (
    "val-fit and val-report partitions must not share rows"
)
check_disjoint(rdl_val_table.df, rdl_val_report_table.df, "val fit vs report")

_rdl_orig_val_table        = copy.deepcopy(rdl_val_table)
_rdl_orig_val_report_table = copy.deepcopy(rdl_val_report_table)
_rdl_orig_train_table      = copy.deepcopy(rdl_train_table)
_rdl_orig_test_table       = copy.deepcopy(rdl_test_table)

_eval_split_name = "Test"
if "churn" not in rdl_test_table.df.columns:
    # RelBench test labels are hidden for leaderboard usage; fall back to the held-out
    # validation partition (disjoint from the rows used for early stopping) for final evaluation.
    _eval_split_name = "Val"
    rdl_test_table = copy.deepcopy(rdl_val_report_table)
    _rdl_orig_test_table = copy.deepcopy(_rdl_orig_val_report_table)
    print("[warn] test split has no 'churn' labels; using held-out validation partition (disjoint from tuning rows) for final evaluation.")

rdl_entity_table = rdl_task.entity_table   # "customer"
_tune_metric_env = os.environ.get("TUNE_METRIC", "auprc").strip().lower()
if _tune_metric_env in {"auprc", "average_precision", "ap"}:
    rdl_tune_metric = "auprc"
elif _tune_metric_env in {"roc_auc", "auroc"}:
    rdl_tune_metric = "roc_auc"
else:
    raise ValueError(f"Unsupported TUNE_METRIC={_tune_metric_env}; use 'auprc' or 'roc_auc'.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — ACTIVITY FEATURES + AT-RISK FILTERING
# ═══════════════════════════════════════════════════════════════════════════
review_events = rdl_db.table_dict["review"].df[["customer_id", "review_time"]].dropna().copy()
review_events["customer_id"] = review_events["customer_id"].astype(np.int64)
review_events["review_time"] = pd.to_datetime(review_events["review_time"])
review_events = review_events.sort_values(["customer_id", "review_time"]).reset_index(drop=True)
review_events["cum_reviews"] = review_events.groupby("customer_id").cumcount() + 1


def _activity_features_for_targets(target_df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(np.int64)
    q["timestamp"] = pd.to_datetime(q["timestamp"])
    q["_row_id"] = np.arange(len(q), dtype=np.int64)
    q = q.sort_values(["timestamp", "customer_id", "_row_id"]).reset_index(drop=True)
    right = review_events[["customer_id", "review_time", "cum_reviews"]]
    now_join = pd.merge_asof(
        q[["customer_id", "timestamp", "_row_id"]].sort_values(["timestamp", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="timestamp", right_on="review_time",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    q["reviews_before_now"] = now_join["cum_reviews"].fillna(0).astype(np.int64)
    q["last_review_time"]   = now_join["review_time"]
    q["window_start"]       = q["timestamp"] - pd.Timedelta(days=lookback_days)
    start_join = pd.merge_asof(
        q[["customer_id", "window_start", "_row_id"]].sort_values(["window_start", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="window_start", right_on="review_time",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    q["reviews_before_window"] = start_join["cum_reviews"].fillna(0).astype(np.int64)
    q["recent_reviews"]        = q["reviews_before_now"] - q["reviews_before_window"]

    # Previous lookback window count for trend computation.
    q["prev_window_start"] = q["timestamp"] - pd.Timedelta(days=2 * lookback_days)
    prev_join = pd.merge_asof(
        q[["customer_id", "prev_window_start", "_row_id"]].sort_values(["prev_window_start", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="prev_window_start", right_on="review_time",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    q["reviews_before_prev_window"] = prev_join["cum_reviews"].fillna(0).astype(np.int64)
    q["prev_reviews"] = q["reviews_before_window"] - q["reviews_before_prev_window"]

    # trend_w(u): relative change between the last and previous windows.
    q["trend_w"] = (q["recent_reviews"] - q["prev_reviews"]) / (q["prev_reviews"] + 1.0)
    q["avg_activity"] = 0.5 * (q["recent_reviews"] + q["prev_reviews"])

    delta = q["timestamp"] - q["last_review_time"]
    q["days_since_last_review"] = delta.dt.days.fillna(9999).astype(np.int64)
    q = q.sort_values("_row_id").reset_index(drop=True)
    return q[["recent_reviews", "prev_reviews", "trend_w", "avg_activity", "days_since_last_review"]]


def _identify_at_risk_users(target_df, activity_df, thresholds=None):
    """
    Identify at-risk users via trend-based criterion:
      AtRisk(u) = 1[ trend_w(u) <= -tau(a_bar(u)) ]
      tau(a_bar) = a + log_b(a_bar + 1) * (1 + alpha * a_bar)
    """
    raw = target_df["churn"].astype(np.int64).to_numpy()

    if thresholds is None:
        thresholds = {
            "a": float(os.environ.get("ATRISK_TAU_A", "0.05")),
            "b": float(os.environ.get("ATRISK_TAU_B", "10.0")),
            "alpha": float(os.environ.get("ATRISK_TAU_ALPHA", "0.02")),
            "tau_multiplier": float(os.environ.get("ATRISK_TAU_MULTIPLIER", "1.0")),
        }
    a = float(thresholds["a"])
    b = float(thresholds["b"])
    alpha = float(thresholds["alpha"])
    tau_multiplier = float(thresholds.get("tau_multiplier", 1.0))
    if b <= 1.0:
        raise ValueError(f"ATRISK_TAU_B must be > 1.0, got {b}")
    if tau_multiplier <= 0.0:
        raise ValueError(f"ATRISK_TAU_MULTIPLIER must be > 0, got {tau_multiplier}")

    trend = activity_df["trend_w"].to_numpy(dtype=np.float64)
    a_bar = np.maximum(activity_df["avg_activity"].to_numpy(dtype=np.float64), 0.0)
    tau = (a + (np.log(a_bar + 1.0) / np.log(b)) * (1.0 + alpha * a_bar)) * tau_multiplier
    at_risk = trend <= (-tau)

    keep_mask      = ~at_risk
    anomaly_labels = raw * keep_mask.astype(np.int64)
    return anomaly_labels, keep_mask, thresholds


def _select_positive_indices(
    pos_index: np.ndarray,
    activity_df: pd.DataFrame,
    thresholds: Dict[str, float],
    n_keep: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_keep >= len(pos_index):
        return pos_index

    mode = POS_DOWNSAMPLE_MODE
    if mode == "subtle":
        a = float(thresholds["a"])
        b = float(thresholds["b"])
        alpha = float(thresholds["alpha"])
        tau_multiplier = float(thresholds.get("tau_multiplier", 1.0))

        sub = activity_df.loc[pos_index]
        trend = sub["trend_w"].to_numpy(dtype=np.float64)
        a_bar = np.maximum(sub["avg_activity"].to_numpy(dtype=np.float64), 0.0)
        tau = (a + (np.log(a_bar + 1.0) / np.log(b)) * (1.0 + alpha * a_bar)) * tau_multiplier
        margin_to_boundary = trend + tau

        order = np.argsort(margin_to_boundary, kind="mergesort")
        return pos_index[order[:n_keep]]

    return rng.choice(pos_index, size=n_keep, replace=False)


# Compute on FULL tables first (need indices aligned before filtering)
rdl_activity_train = _activity_features_for_targets(rdl_train_table.df, lookback_days=30)
rdl_activity_val   = _activity_features_for_targets(rdl_val_table.df, lookback_days=30)
rdl_activity_test  = _activity_features_for_targets(rdl_test_table.df, lookback_days=30)

rdl_train_anomaly_labels, rdl_train_keep, rdl_at_risk_thresholds = _identify_at_risk_users(
    rdl_train_table.df, rdl_activity_train)
rdl_val_anomaly_labels, rdl_val_keep, _ = _identify_at_risk_users(
    rdl_val_table.df, rdl_activity_val, thresholds=rdl_at_risk_thresholds)
rdl_test_anomaly_labels, rdl_test_keep, _ = _identify_at_risk_users(
    rdl_test_table.df, rdl_activity_test, thresholds=rdl_at_risk_thresholds)

# Apply filtering
rdl_train_table.df = rdl_train_table.df.loc[rdl_train_keep].copy()
rdl_train_table.df["churn"] = rdl_train_anomaly_labels[rdl_train_keep]
rdl_val_table.df = rdl_val_table.df.loc[rdl_val_keep].copy()
rdl_val_table.df["churn"] = rdl_val_anomaly_labels[rdl_val_keep]
rdl_test_table.df = rdl_test_table.df.loc[rdl_test_keep].copy()
rdl_test_table.df["churn"] = rdl_test_anomaly_labels[rdl_test_keep]

# Downsample to configurable target anomaly rate (default 1.5%)
_target_anom_rate = float(os.environ.get("TARGET_ANOM_RATE", "0.015"))
if not (0.0 < _target_anom_rate < 1.0):
    raise ValueError(f"TARGET_ANOM_RATE must be in (0,1), got {_target_anom_rate}")
_rdl_train_pos_df = rdl_train_table.df[rdl_train_table.df["churn"] == 1]
_rdl_train_neg_df = rdl_train_table.df[rdl_train_table.df["churn"] == 0]
if 0.0 < _target_anom_rate < 1.0 and len(_rdl_train_pos_df) > 0 and len(_rdl_train_neg_df) > 0:
    _n_pos_tgt_tr = int(round((_target_anom_rate / (1.0 - _target_anom_rate)) * len(_rdl_train_neg_df)))
    _n_pos_tgt_tr = min(_n_pos_tgt_tr, len(_rdl_train_pos_df))
    if _n_pos_tgt_tr < len(_rdl_train_pos_df):
        _rng_tr = np.random.default_rng(RUN_SEED)
        _kp_idx = _select_positive_indices(
            _rdl_train_pos_df.index.to_numpy(),
            rdl_activity_train,
            rdl_at_risk_thresholds,
            _n_pos_tgt_tr,
            _rng_tr,
        )
        rdl_train_table.df = rdl_train_table.df.loc[
            np.sort(np.concatenate([_rdl_train_neg_df.index.to_numpy(), _kp_idx]))].copy()

_rdl_val_pos_df = rdl_val_table.df[rdl_val_table.df["churn"] == 1]
_rdl_val_neg_df = rdl_val_table.df[rdl_val_table.df["churn"] == 0]
if 0.0 < _target_anom_rate < 1.0 and len(_rdl_val_pos_df) > 0 and len(_rdl_val_neg_df) > 0:
    _n_pos_tgt_vl = int(round((_target_anom_rate / (1.0 - _target_anom_rate)) * len(_rdl_val_neg_df)))
    _n_pos_tgt_vl = min(_n_pos_tgt_vl, len(_rdl_val_pos_df))
    if _n_pos_tgt_vl < len(_rdl_val_pos_df):
        _rng_vl = np.random.default_rng(RUN_SEED)
        _kp_idx_vl = _rng_vl.choice(_rdl_val_pos_df.index.to_numpy(), size=_n_pos_tgt_vl, replace=False)
        rdl_val_table.df = rdl_val_table.df.loc[
            np.sort(np.concatenate([_rdl_val_neg_df.index.to_numpy(), _kp_idx_vl]))].copy()

# Downsample the reporting split to the same target rate. Without this, the held-out
# report partition retains its natural post-at-risk-filter prevalence (which can be far
# higher than the rate the model is tuned on), making AUPRC there dominated by base rate
# rather than by ranking quality.
_rdl_test_pos_df = rdl_test_table.df[rdl_test_table.df["churn"] == 1]
_rdl_test_neg_df = rdl_test_table.df[rdl_test_table.df["churn"] == 0]
if 0.0 < _target_anom_rate < 1.0 and len(_rdl_test_pos_df) > 0 and len(_rdl_test_neg_df) > 0:
    _n_pos_tgt_te = int(round((_target_anom_rate / (1.0 - _target_anom_rate)) * len(_rdl_test_neg_df)))
    _n_pos_tgt_te = min(_n_pos_tgt_te, len(_rdl_test_pos_df))
    if _n_pos_tgt_te < len(_rdl_test_pos_df):
        _rng_te = np.random.default_rng(RUN_SEED)
        _kp_idx_te = _rng_te.choice(_rdl_test_pos_df.index.to_numpy(), size=_n_pos_tgt_te, replace=False)
        rdl_test_table.df = rdl_test_table.df.loc[
            np.sort(np.concatenate([_rdl_test_neg_df.index.to_numpy(), _kp_idx_te]))].copy()

_val_labels = rdl_val_table.df["churn"].astype(np.int64).to_numpy()
_test_labels = rdl_test_table.df["churn"].astype(np.int64).to_numpy()

print("=" * 70)
print(f"AT-RISK USER FILTERING (TARGET {_target_anom_rate:.4f} ANOMALY RATE)")
print("=" * 70)
print(f"  trend filter params: a={rdl_at_risk_thresholds['a']}  b={rdl_at_risk_thresholds['b']}  alpha={rdl_at_risk_thresholds['alpha']}")
print(f"  tau multiplier: {rdl_at_risk_thresholds.get('tau_multiplier', 1.0):.3f} | pos downsample mode: {POS_DOWNSAMPLE_MODE}")
print(f"  Train: {len(rdl_train_table.df):,} users | {int(rdl_train_table.df['churn'].sum()):,} anomalies "
      f"({rdl_train_table.df['churn'].mean():.4f})")
print(f"  Val:   {len(rdl_val_table.df):,} users | {int(rdl_val_table.df['churn'].sum()):,} anomalies "
      f"({rdl_val_table.df['churn'].mean():.4f})")
print(f"  Test:  {len(rdl_test_table.df):,} users | {int(rdl_test_table.df['churn'].sum()):,} anomalies "
    f"({rdl_test_table.df['churn'].mean():.4f})")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — FLAT FEATURES (for RF mining)
# ═══════════════════════════════════════════════════════════════════════════
review_df  = rdl_db.table_dict["review"].df.copy()
review_df["customer_id"]  = review_df["customer_id"].astype(np.int64)
review_df["review_time"]  = pd.to_datetime(review_df["review_time"])
_rating_col = next((c for c in ["rating", "star_rating", "stars"] if c in review_df.columns), None)


def _build_customer_flat(target_df, activity_df, lookback_90=90):
    base = target_df[["customer_id", "timestamp"]].copy()
    base["customer_id"] = base["customer_id"].astype(np.int64)
    base["timestamp"]   = pd.to_datetime(base["timestamp"])
    base = base.reset_index(drop=True)

    base["recent_reviews_30d"]     = activity_df["recent_reviews"].values
    base["days_since_last_review"] = activity_df["days_since_last_review"].values

    _right = review_events[["customer_id", "review_time", "cum_reviews"]]
    _j = pd.merge_asof(
        base[["customer_id", "timestamp"]].sort_values(["timestamp", "customer_id"]),
        _right.sort_values(["review_time", "customer_id"]),
        left_on="timestamp", right_on="review_time",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    base["total_reviews"] = _j["cum_reviews"].fillna(0).astype(np.int64).values

    _q2 = base[["customer_id", "timestamp"]].copy()
    _q2["window_90"] = _q2["timestamp"] - pd.Timedelta(days=lookback_90)
    _j2 = pd.merge_asof(
        _q2.sort_values(["window_90", "customer_id"]),
        _right.sort_values(["review_time", "customer_id"]),
        left_on="window_90", right_on="review_time",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    base["recent_reviews_90d"] = np.maximum(
        0, base["total_reviews"].values - _j2["cum_reviews"].fillna(0).astype(np.int64).values)

    if _rating_col is not None:
        _rated = review_df[["customer_id", _rating_col]].copy()
        _rated[_rating_col] = pd.to_numeric(_rated[_rating_col], errors="coerce")
        _rated = _rated.dropna(subset=[_rating_col])
        _agg = _rated.groupby("customer_id")[_rating_col].agg(
            mean_rating="mean", rating_std="std", min_rating="min", max_rating="max").reset_index()
        base = base.merge(_agg, on="customer_id", how="left")
        for c, v in [("mean_rating", 3.0), ("rating_std", 0.0), ("min_rating", 3.0), ("max_rating", 3.0)]:
            base[c] = base[c].fillna(v)
    else:
        for c in ["mean_rating", "rating_std", "min_rating", "max_rating"]:
            base[c] = 0.0

    if "product_id" in review_df.columns:
        _div = review_df.groupby("customer_id")["product_id"].nunique().rename("distinct_products")
        base = base.merge(_div.reset_index(), on="customer_id", how="left")
        base["distinct_products"] = base["distinct_products"].fillna(0).astype(np.int64)
    else:
        base["distinct_products"] = 0

    _span = review_df.groupby("customer_id")["review_time"].agg(
        first_review="min", last_review="max").reset_index()
    _span["review_span_days"] = (
        (_span["last_review"] - _span["first_review"]).dt.days.fillna(0).astype(np.int64))
    base = base.merge(_span[["customer_id", "review_span_days"]], on="customer_id", how="left")
    base["review_span_days"] = base["review_span_days"].fillna(0).astype(np.int64)
    base["low_rating_flag"] = (base["mean_rating"] < 3.0).astype(np.int8)
    return base


# Recompute activity on filtered table for flat features
_act_train_filt = _activity_features_for_targets(rdl_train_table.df, lookback_days=30)
train_flat = _build_customer_flat(rdl_train_table.df, _act_train_filt)
print(f"Train flat features: {train_flat.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — GRAPH BUILDING (use cache or build from scratch)
# ═══════════════════════════════════════════════════════════════════════════
class GloveTextEmbedding:
    def __init__(self, device="cpu", encode_batch_size=64):
        self.encode_batch_size = int(encode_batch_size)
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", device=device)
    def __call__(self, sentences):
        return torch.from_numpy(self.model.encode(
            sentences, batch_size=self.encode_batch_size,
            show_progress_bar=False, convert_to_numpy=True))


rdl_col_to_stype      = get_stype_proposal(rdl_db)
rdl_text_embedder_cfg = TextEmbedderConfig(
    text_embedder=GloveTextEmbedding(device="cpu", encode_batch_size=64), batch_size=64)

if rdl_data is not None and rdl_col_stats_dict is not None:
    print("Using preloaded cached graph artifacts.")
else:
    print("Building graph from scratch (this may take ~10 min)...")
    rdl_data, rdl_col_stats_dict = make_pkey_fkey_graph(
        rdl_db, col_to_stype_dict=rdl_col_to_stype,
        text_embedder_cfg=rdl_text_embedder_cfg, cache_dir=None)
    rdl_data = _mask_identity_proxy_features(_sanitize_graph(rdl_data))
    torch.save(rdl_data, rdl_graph_cache_path)
    torch.save(rdl_col_stats_dict, rdl_stats_cache_path)
    print("Graph built and cached.")

gc.collect()
torch.cuda.empty_cache()
print(rdl_data)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — RF RULE MINING
# ═══════════════════════════════════════════════════════════════════════════
_flat = train_flat.copy()
_flat["churn"] = rdl_train_table.df["churn"].to_numpy(dtype=np.int64)

print(f"\nRF table: {_flat.shape}  anomaly_rate={_flat['churn'].mean():.4f}")

RF_PARAMS = dict(n_estimators=200, max_depth=10, min_samples_leaf=5,
                 random_state=RUN_SEED, n_jobs=-1, class_weight="balanced_subsample")
RF_MAX_TRAIN = 200_000
DROP_COLS = {"customer_id", "timestamp", "churn"}

feature_names = [c for c in _flat.columns if c not in DROP_COLS]
X_df = _flat[feature_names].apply(pd.to_numeric, errors="coerce")
X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)
X    = X_df.to_numpy(dtype=np.float32)
y_rf = _flat["churn"].to_numpy(dtype=np.int64)

rng = np.random.default_rng(RUN_SEED)
if len(X) > RF_MAX_TRAIN:
    pos_idx = np.where(y_rf > 0)[0]; neg_idx = np.where(y_rf == 0)[0]
    pos_rate = float(y_rf.mean())
    pos_take = min(pos_idx.size, max(1, int(RF_MAX_TRAIN * pos_rate)))
    neg_take = min(neg_idx.size, RF_MAX_TRAIN - pos_take)
    sampled = rng.permutation(np.concatenate([
        pos_idx[rng.permutation(pos_idx.size)[:pos_take]],
        neg_idx[rng.permutation(neg_idx.size)[:neg_take]],
    ]))
    X_fit, y_fit = X[sampled], y_rf[sampled]
else:
    X_fit, y_fit = X, y_rf

# Carve a rule-selection slice out of the RF training rows: the forest fits on the
# remainder, and leaf purity/support are scored on the held-out slice only, so rule
# statistics are not in-sample on the same rows that grew the leaves.
RULE_SELECTION_FRAC = float(os.environ.get("RULE_SELECTION_FRAC", "0.2"))
_rs_rng = np.random.default_rng(RUN_SEED)
_rs_perm = _rs_rng.permutation(len(X_fit))
_n_rule_sel = max(1, int(round(RULE_SELECTION_FRAC * len(X_fit))))
_rule_sel_idx = _rs_perm[:_n_rule_sel]
_rf_fit_idx = _rs_perm[_n_rule_sel:]
X_rf_fit, y_rf_fit = X_fit[_rf_fit_idx], y_fit[_rf_fit_idx]
X_rule_sel, y_rule_sel = X_fit[_rule_sel_idx], y_fit[_rule_sel_idx]
_rule_sel_base_rate = float(y_rule_sel.mean()) if len(y_rule_sel) else float("nan")

print(f"RF training: {len(X_rf_fit):,} rows  pos={int(y_rf_fit.sum()):,}  features={X.shape[1]}  "
      f"| rule-selection holdout: {len(X_rule_sel):,} rows  pos={int(y_rule_sel.sum()):,}")
clf = RandomForestClassifier(**RF_PARAMS)
clf.fit(X_rf_fit, y_rf_fit)
print("RF trained.")

# RF_MLP_TUNE (only under RF_MLP_ONLY): small cross-validated grid search for the
# supervised-detector rows specifically. Does not touch `clf` (still the fixed-params
# forest whose leaves become RAD's rules) -- this only searches a *separate* detector.
_det_clf, _det_mlp = clf, None  # placeholders reassigned below if tuning is on
if RF_MLP_ONLY and os.environ.get("RF_MLP_FIXED_TUNED", "0").strip() == "1":
    # Reapply an already-found best config (skip the CV search) so a multi-seed sweep
    # doesn't re-tune on every seed.
    _det_clf = RandomForestClassifier(
        n_estimators=int(os.environ.get("RF_TUNED_N_ESTIMATORS", "400")),
        max_depth=None if os.environ.get("RF_TUNED_MAX_DEPTH", "none").lower() == "none" else int(os.environ["RF_TUNED_MAX_DEPTH"]),
        min_samples_leaf=int(os.environ.get("RF_TUNED_MIN_SAMPLES_LEAF", "1")),
        class_weight="balanced_subsample", n_jobs=-1, random_state=RUN_SEED,
    )
    _det_clf.fit(X_rf_fit, y_rf_fit)
    _mlp_tune_scaler = StandardScaler().fit(X_rf_fit)
    _hidden = tuple(int(h) for h in os.environ.get("MLP_TUNED_HIDDEN", "128,64,32").split(","))
    _mlp_fixed = MLPClassifier(
        hidden_layer_sizes=_hidden, alpha=float(os.environ.get("MLP_TUNED_ALPHA", "1e-5")),
        activation="relu", learning_rate_init=1e-3, max_iter=200,
        early_stopping=True, n_iter_no_change=10, random_state=RUN_SEED,
    )
    _mlp_fixed.fit(_mlp_tune_scaler.transform(X_rf_fit), y_rf_fit)

    class _ScaledMLP:
        def __init__(self, scaler, mlp):
            self.scaler, self.mlp = scaler, mlp
        def predict_proba(self, Xv):
            return self.mlp.predict_proba(self.scaler.transform(Xv))
    _det_mlp = _ScaledMLP(_mlp_tune_scaler, _mlp_fixed)
    print(f"[fixed-tuned] RF n_estimators={_det_clf.n_estimators} max_depth={_det_clf.max_depth} "
          f"min_samples_leaf={_det_clf.min_samples_leaf} | MLP hidden={_hidden} alpha={_mlp_fixed.alpha}")
elif RF_MLP_ONLY and os.environ.get("RF_MLP_TUNE", "0").strip() == "1":
    from sklearn.model_selection import GridSearchCV

    print("[tune] Grid-searching RF (cv=3, scoring=average_precision)...")
    _rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=RUN_SEED),
        param_grid={"n_estimators": [100, 200, 400], "max_depth": [6, 10, 15, None], "min_samples_leaf": [1, 5, 10]},
        scoring="average_precision", cv=3, n_jobs=-1,
    )
    _rf_grid.fit(X_rf_fit, y_rf_fit)
    _det_clf = _rf_grid.best_estimator_
    print(f"[tune] RF best params: {_rf_grid.best_params_}  cv_auprc={_rf_grid.best_score_:.4f}")

    print("[tune] Grid-searching MLP (cv=3, scoring=average_precision)...")
    _mlp_tune_scaler = StandardScaler().fit(X_rf_fit)
    _mlp_grid = GridSearchCV(
        MLPClassifier(activation="relu", learning_rate_init=1e-3, max_iter=200,
                      early_stopping=True, n_iter_no_change=10, random_state=RUN_SEED),
        param_grid={"hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)], "alpha": [1e-5, 1e-4, 1e-3]},
        scoring="average_precision", cv=3, n_jobs=-1,
    )
    _mlp_grid.fit(_mlp_tune_scaler.transform(X_rf_fit), y_rf_fit)
    print(f"[tune] MLP best params: {_mlp_grid.best_params_}  cv_auprc={_mlp_grid.best_score_:.4f}")

    class _ScaledMLP:
        def __init__(self, scaler, mlp):
            self.scaler, self.mlp = scaler, mlp
        def predict_proba(self, Xv):
            return self.mlp.predict_proba(self.scaler.transform(Xv))
    _det_mlp = _ScaledMLP(_mlp_tune_scaler, _mlp_grid.best_estimator_)

# Forest-as-detector row (E1): same supervision and features as the rules, minus the
# graph and minus the injection — answers "is this just the RF, not the graph?". Scores are
# computed now (clf is in scope) but metrics are reported in Section 15 alongside the others.
# When RF_MLP_TUNE is on, _det_clf/_det_mlp are the tuned models from the grid search above
# instead of the fixed-hyperparameter clf/_mlp_clf.
_report_act_rf = _activity_features_for_targets(rdl_test_table.df, lookback_days=30)
_report_flat_rf = _build_customer_flat(rdl_test_table.df, _report_act_rf)
_report_X_rf_df = _report_flat_rf.reindex(columns=feature_names, fill_value=0).apply(pd.to_numeric, errors="coerce")
_report_X_rf_df = _report_X_rf_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)
_report_X_rf = _report_X_rf_df.to_numpy(dtype=np.float32)
rf_detector_scores = _det_clf.predict_proba(_report_X_rf)[:, 1]

# MLP-as-detector row (E1): same supervision, features, and reporting split as the RF
# row, minus the tree ensemble -- a second, simple flattened-supervised baseline.
if _det_mlp is None:
    _mlp_scaler = StandardScaler().fit(X_rf_fit)
    _mlp_clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=RUN_SEED,
    )
    _mlp_clf.fit(_mlp_scaler.transform(X_rf_fit), y_rf_fit)
    mlp_detector_scores = _mlp_clf.predict_proba(_mlp_scaler.transform(_report_X_rf))[:, 1]
else:
    mlp_detector_scores = _det_mlp.predict_proba(_report_X_rf)[:, 1]

if RF_MLP_ONLY:
    def _rf_mlp_only_pak(y_true, scores):
        y = np.asarray(y_true).astype(int)
        s = np.asarray(scores, dtype=np.float64)
        k = int(max(1, y.sum()))
        return float(y[np.argsort(-s)[:k]].mean())

    def _rf_mlp_only_lift(auprc, y_true):
        prev = float(np.asarray(y_true).mean())
        return float(auprc / prev) if prev > 0 else float("nan")

    print("\n" + "=" * 70)
    print(f"RF/MLP SUPERVISED-DETECTOR ROWS ONLY ({_eval_split_name} Set)")
    print("=" * 70)
    for _name, _scores in [("RF (supervised detector)", rf_detector_scores), ("MLP (supervised detector)", mlp_detector_scores)]:
        _auprc = float(average_precision_score(_test_labels, _scores)) if np.unique(_test_labels).size > 1 else float("nan")
        _auroc = float(roc_auc_score(_test_labels, _scores)) if np.unique(_test_labels).size > 1 else float("nan")
        _pak = _rf_mlp_only_pak(_test_labels, _scores)
        _lift_v = _rf_mlp_only_lift(_auprc, _test_labels)
        print(f"{_name:<30}  auprc={_auprc:.4f}  auroc={_auroc:.4f}  p@k={_pak:.4f}  lift={_lift_v:.4f}")
        append_results_jsonl({
            "task": "amazon-user-churn", "seed": RUN_SEED, "split": _eval_split_name,
            "method": _name, "auprc": _auprc, "auroc": _auroc, "p@k": _pak, "lift": _lift_v,
        })
    print("=" * 70)
    raise SystemExit(0)


def _parent_map(tree_):
    parent = np.full(tree_.node_count, -1, np.int32)
    is_left = np.zeros(tree_.node_count, bool)
    for p in range(tree_.node_count):
        l, r = int(tree_.children_left[p]), int(tree_.children_right[p])
        if l >= 0: parent[l] = p; is_left[l] = True
        if r >= 0: parent[r] = p
    return parent, is_left


def _leaf_sql(tree_est, leaf_id, fnames):
    tree_ = tree_est.tree_
    parent, is_left = _parent_map(tree_)
    conds, node = [], int(leaf_id)
    while parent[node] != -1:
        p = int(parent[node])
        fidx = int(tree_.feature[p])
        if fidx >= 0:
            op = "<=" if is_left[node] else ">"
            conds.append(f"{fnames[fidx]} {op} {tree_.threshold[p]:.6g}")
        node = p
    conds.reverse()
    return " AND ".join(conds)


MIN_LEAF_SAMPLES = 5
MIN_POS_PROB = 0.50
POOL_TREES = 100

all_rows = []
leaf_assignments = np.stack([est.apply(X_rule_sel) for est in clf.estimators_[:POOL_TREES]], axis=1)

for tid, tree_est in enumerate(clf.estimators_[:POOL_TREES]):
    tree_ = tree_est.tree_
    row_leaf = leaf_assignments[:, tid]
    for leaf_id in np.where(tree_.children_left == -1)[0]:
        if int(tree_.n_node_samples[leaf_id]) < MIN_LEAF_SAMPLES: continue
        vals = tree_.value[leaf_id][0]
        pos_prob = float(vals[1] / vals.sum()) if (vals.sum() > 0 and vals.shape[0] > 1) else 0.0
        if pos_prob < MIN_POS_PROB: continue
        sql = _leaf_sql(tree_est, leaf_id, feature_names)
        if not sql: continue
        mask = row_leaf == leaf_id
        val_support = int(mask.sum())
        val_precision = float(y_rule_sel[mask].mean()) if mask.any() else 0.0
        val_lift = (
            (val_precision / _rule_sel_base_rate)
            if _rule_sel_base_rate and _rule_sel_base_rate > 0 else np.nan
        )
        all_rows.append({
            "rule_id": f"rule_{len(all_rows)+1:04d}", "tree_id": tid,
            "pos_prob": pos_prob, "val_support": val_support,
            "val_precision": val_precision, "val_lift": val_lift,
            "sql_predicate": sql,
        })

rule_paths_df = (
    pd.DataFrame(all_rows)
    .sort_values(["val_precision", "val_support", "val_lift"], ascending=False)
    .drop_duplicates(subset=["sql_predicate"])
    .reset_index(drop=True)
)
print(f"Extracted {len(rule_paths_df):,} unique rule candidates.")


def _sample_rules(df, n=30, seed=42):
    pool = df[df["val_support"] >= 1].head(2000).copy()
    if pool.empty: return []
    w = (
        np.sqrt(pool["val_precision"].clip(0))
        * np.log1p(pool["val_support"].clip(0))
        * np.sqrt(pool["val_lift"].fillna(1.0).clip(lower=1.0))
    ).to_numpy() + 1e-9
    w /= w.sum()
    pick = np.random.default_rng(seed).choice(len(pool), size=min(n, len(pool)), replace=False, p=w)
    return pool.iloc[sorted(pick)].to_dict(orient="records")


seed_rules = _sample_rules(rule_paths_df, n=30)
print(f"Sampled {len(seed_rules)} seed rules for LLM.")

RF_RULE_MODE = os.environ.get("RF_RULE_MODE", "llm").strip().lower()
if RF_RULE_MODE not in {"llm", "rf_seed", "rf_decision"}:
    raise ValueError(f"Unsupported RF_RULE_MODE={RF_RULE_MODE}; use llm|rf_seed|rf_decision")
RF_DECISION_TREES = int(os.environ.get("RF_DECISION_TREES", "15"))
print(f"Rule source mode: {RF_RULE_MODE}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — RULE CANDIDATE GENERATION
# ═══════════════════════════════════════════════════════════════════════════
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")

USE_CACHED_LLM_RULES = os.environ.get("USE_CACHED_LLM_RULES", "0").strip() == "1"
PARSED_RULES_CSV = amazon_data("amazon_parsed_rules_df.csv")
# RULES_OUT_DIR: write this run's parsed rules and rule table there instead of data/amazon/,
# so a run never rewrites the shared cache that USE_CACHED_LLM_RULES (or another run) reads.
RULES_OUT_DIR    = os.environ.get("RULES_OUT_DIR", "").strip()
PARSED_RULES_OUT = os.path.join(RULES_OUT_DIR, "parsed_rules_df.csv") if RULES_OUT_DIR else PARSED_RULES_CSV
response_text = "[]"
_used_cached_rules = False
LLM_GEN_KWARGS = dict(max_new_tokens=int(os.environ.get("LLM_MAX_NEW_TOKENS", "2500")),
                      do_sample=False, temperature=0.0, top_p=1.0)

if RF_RULE_MODE == "llm":
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    if USE_CACHED_LLM_RULES and os.path.exists(PARSED_RULES_CSV):
        _cached = pd.read_csv(PARSED_RULES_CSV)
        _cached = _cached.dropna(subset=["sql_predicate"]).copy()
        _cached = _cached[_cached["sql_predicate"].astype(str).str.strip() != ""]
        _cached = _cached.head(30)
        _cached_rules = []
        for _, _r in _cached.iterrows():
            _cached_rules.append({
                "candidate_id": str(_r.get("candidate_id", "cached_rule")),
                "sql_predicate": str(_r.get("sql_predicate", "")),
                "mutation_type": str(_r.get("mutation_type", "cached")),
                "rationale": "cached rule",
            })
        response_text = json.dumps(_cached_rules)
        _used_cached_rules = True
        print(f"Using cached parsed LLM rules from {PARSED_RULES_CSV} ({len(_cached_rules)} rules).")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        total_gib = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
        llm_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, token=HF_TOKEN,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True,
            ),
            device_map="auto",
            max_memory={0: f"{max(4, int(total_gib*0.9))}GiB", "cpu": "64GiB"},
            low_cpu_mem_usage=True,
        )
        pipe = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)
        print(f"Llama loaded (GPU budget={int(total_gib*0.9)}GiB).")
elif RF_RULE_MODE == "rf_seed":
    _rf_seed_candidates = [
        {
            "candidate_id": str(r.get("rule_id", f"seed_{i:03d}")),
            "sql_predicate": str(r.get("sql_predicate", "")),
            "mutation_type": "rf_seed",
            "rationale": "RF sampled rule",
        }
        for i, r in enumerate(seed_rules)
        if str(r.get("sql_predicate", "")).strip()
    ]
    response_text = json.dumps(_rf_seed_candidates)
    print(f"Using RF sampled seed rules only ({len(_rf_seed_candidates)} rules).")

SCHEMA = """
Table: features_table  (customer-level activity for Amazon user-churn detection)
Columns:
  recent_reviews_30d      INTEGER  -- reviews in last 30 days
  days_since_last_review  INTEGER  -- days since most recent review (9999=never)
  total_reviews           INTEGER  -- cumulative reviews before churn timestamp
  recent_reviews_90d      INTEGER  -- reviews in last 90 days
  mean_rating             FLOAT    -- average star rating (1-5)
  rating_std              FLOAT    -- std-dev of star ratings
  min_rating              FLOAT    -- minimum star rating given
  max_rating              FLOAT    -- maximum star rating given
  distinct_products       INTEGER  -- distinct products reviewed
  review_span_days        INTEGER  -- days between first and last review
  low_rating_flag         INTEGER  -- 1 if mean_rating < 3.0
""".strip()

seed_text = json.dumps([{
    "rule_id": r.get("rule_id", f"seed_{i:03d}"),
    "sql_predicate": r.get("sql_predicate", ""),
    "val_precision": round(float(r.get("val_precision", 0)), 4),
    "val_support": int(r.get("val_support", 0)),
} for i, r in enumerate(seed_rules[:20])], indent=2)

if RF_RULE_MODE == "llm" and not (USE_CACHED_LLM_RULES and os.path.exists(PARSED_RULES_CSV)):
    outputs = pipe(
        [
            {"role": "system", "content": (
                "You write SQL WHERE-clause predicates for an Amazon user anomaly detector. "
                "The positives are churn anomalies after at-risk users are removed. "
                "You receive RF seed rules and must generate refined or novel predicates. "
                "Return only a valid JSON array. No prose."
            )},
            {"role": "user", "content": (
                f"SCHEMA:\n{SCHEMA}\n\n"
                f"SEED RULES:\n{seed_text}\n\n"
                "Generate 15 SQL WHERE predicates. Each object must have keys: "
                "\"candidate_id\" (string), \"sql_predicate\" (SQL), "
                "\"mutation_type\" (tighten|relax|combine|novel), \"rationale\" (one sentence). "
                "Return only the JSON array."
            )},
        ],
        **LLM_GEN_KWARGS,
    )
    response_text = outputs[0]["generated_text"][-1]["content"]
    print("LLM response (first 500 chars):")
    print(response_text[:500])

    # Free LLM memory before loading GNN models
    del llm_model, pipe, tokenizer, outputs
    gc.collect()
    torch.cuda.empty_cache()
    print("\nLLM model freed from GPU memory.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9 — PARSE + EVALUATE SQL RULES via DuckDB
# ═══════════════════════════════════════════════════════════════════════════
import ast, duckdb

SQL_KW = {"AND", "OR", "NOT", "IN", "IS", "NULL", "TRUE", "FALSE", "BETWEEN", "LIKE"}


def _parse_llm(text):
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL | re.IGNORECASE)
    if m: s = m.group(1).strip()
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s); s = re.sub(r"\bFalse\b", "false", s)
    try:
        result = json.loads(s)
        if isinstance(result, list): return result
    except Exception:
        pass
    start = s.find("[")
    if start == -1: return []
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
        elif c == '"': in_str = True
        elif c == "[": depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                chunk = s[start:i+1]
                try: return json.loads(chunk)
                except Exception:
                    try: return ast.literal_eval(chunk)
                    except Exception: return []
    return []


def _seed_fallback():
    return [
        {"candidate_id": f"seed_{i:03d}", "sql_predicate": r["sql_predicate"],
         "mutation_type": "seed_fallback", "rationale": "RF seed rule"}
        for i, r in enumerate(seed_rules[:20]) if r.get("sql_predicate", "").strip()
    ]

llm_candidates = _parse_llm(response_text)
print(f"Parsed {len(llm_candidates)} rule candidates.")
_n_parsed, _fallback_used, _rejected = len(llm_candidates), False, []

if RF_RULE_MODE in {"llm", "rf_seed"} and not llm_candidates:
    print("Rule parsing failed — using seed rules as fallback.")
    llm_candidates = _seed_fallback()
    _fallback_used = True

# Flat feature table for SQL eval from filtered train table
_act_eval = _activity_features_for_targets(rdl_train_table.df, lookback_days=30)
eval_df = _build_customer_flat(rdl_train_table.df, _act_eval)
eval_df = eval_df.drop(columns=["timestamp"], errors="ignore").copy()
eval_df["customer_id"] = eval_df["customer_id"].astype(np.int64)

# The prompt's SCHEMA must list exactly the columns rules are evaluated against.
_schema_cols = set(re.findall(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:INTEGER|FLOAT)\b", SCHEMA, re.M))
_eval_cols = set(eval_df.columns) - {"customer_id"}
if _schema_cols != _eval_cols:
    raise RuntimeError(f"SCHEMA/rule-evaluation column mismatch: only in SCHEMA={sorted(_schema_cols - _eval_cols)}, "
                       f"only in eval table={sorted(_eval_cols - _schema_cols)}")

all_cids  = eval_df["customer_id"].to_numpy(dtype=np.int64)
result_df = pd.DataFrame({"customer_id": all_cids})

if RF_RULE_MODE == "rf_decision":
    _n_trees = min(max(1, RF_DECISION_TREES), len(clf.estimators_))
    _eval_X_df = eval_df.reindex(columns=feature_names, fill_value=0).apply(pd.to_numeric, errors="coerce")
    _eval_X_df = _eval_X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)
    _eval_X = _eval_X_df.to_numpy(dtype=np.float32)
    parsed_rules_rows = []
    for _tid, _tree in enumerate(clf.estimators_[:_n_trees]):
        _col = f"rf_tree_{_tid:03d}"
        result_df[_col] = _tree.predict(_eval_X).astype(np.int8)
        parsed_rules_rows.append(
            {
                "candidate_id": _col,
                "col_name": _col,
                "sql_predicate": f"tree_{_tid}_predict == 1",
                "mutation_type": "rf_decision",
            }
        )
    print(f"Built RF decision-feature matrix using {_n_trees} trees.")
else:
    # Reject (rather than zero-fill) candidates that reference a column the rule-evaluation
    # table does not have: a zero-filled column silently turns the predicate into a constant.
    _avail_cols = set(eval_df.columns)
    _kept = []
    for cand in llm_candidates:
        _refs = {
            tok for tok in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()", str(cand.get("sql_predicate", "")))
            if tok.upper() not in SQL_KW
        }
        _missing = sorted(_refs - _avail_cols)
        if _missing:
            _rejected.append({"candidate_id": str(cand.get("candidate_id", "")),
                              "sql_predicate": cand.get("sql_predicate"), "missing_columns": _missing})
        else:
            _kept.append(cand)
    print(f"[rules] rejected {len(_rejected)}/{len(llm_candidates)} candidates referencing columns "
          f"missing from the rule-evaluation table")
    for _r in _rejected:
        print(f"  rejected {_r['candidate_id']}: missing {_r['missing_columns']}")
    llm_candidates = _kept
    if not llm_candidates and RF_RULE_MODE in {"llm", "rf_seed"}:
        print("[rules] every candidate was rejected — using seed rules as fallback.")
        llm_candidates = _seed_fallback()
        _fallback_used = True

    con = duckdb.connect(":memory:")
    con.register("features_table", eval_df)

    parsed_rules_rows, used_names, n_matched = [], {"customer_id"}, 0
    for i, cand in enumerate(llm_candidates):
        raw = cand.get("candidate_id", f"llm_{i:03d}")
        col = re.sub(r"[^0-9A-Za-z_]+", "_", str(raw)).strip("_") or f"rule_{i:03d}"
        base, sfx = col, 1
        while col in used_names:
            col = f"{base}_{sfx}"
            sfx += 1
        used_names.add(col)
        result_df[col] = np.int8(0)
        pred = cand.get("sql_predicate", None)
        if isinstance(pred, str) and pred.strip():
            try:
                matched_ids = (
                    pd.to_numeric(
                        con.execute(f"SELECT customer_id FROM features_table WHERE ({pred})")
                        .fetchdf()["customer_id"],
                        errors="coerce",
                    )
                    .dropna()
                    .astype(np.int64)
                    .unique()
                )
                mask = result_df["customer_id"].isin(matched_ids)
                result_df.loc[mask, col] = np.int8(1)
                n_matched += 1
                print(f"  {raw}: {int(mask.sum())} matches")
            except Exception as e:
                print(f"  {raw} SQL error: {str(e)[:160]}")
        parsed_rules_rows.append(
            {
                "candidate_id": raw,
                "col_name": col,
                "sql_predicate": pred,
                "mutation_type": cand.get("mutation_type", ""),
            }
        )

    con.close()
    print(f"\nRules with matches: {n_matched}/{len(llm_candidates)}")
parsed_rules_df = pd.DataFrame(parsed_rules_rows)
Path(PARSED_RULES_OUT).parent.mkdir(parents=True, exist_ok=True)
parsed_rules_df.to_csv(PARSED_RULES_OUT, index=False)
rule_cols = [c for c in result_df.columns if c != "customer_id"]
print(f"Rule columns: {rule_cols}")

# Record how this run's rules were produced next to the saved rules.
PARSED_RULES_GEN_JSON = (os.path.join(RULES_OUT_DIR, "rule_generation.json") if RULES_OUT_DIR
                         else os.path.splitext(PARSED_RULES_CSV)[0] + "_generation.json")
_llm_generated = RF_RULE_MODE == "llm" and not _used_cached_rules
with open(PARSED_RULES_GEN_JSON, "w") as _f:
    json.dump({
        "rf_rule_mode": RF_RULE_MODE,
        "used_cached_rules": (PARSED_RULES_CSV if _used_cached_rules else None),
        "llm": ({"model_id": MODEL_ID, **LLM_GEN_KWARGS} if _llm_generated else None),
        "seed": RUN_SEED,
        "seed_note": ("do_sample=False (greedy decoding), so the LLM output does not depend on a sampling seed; "
                      "SEED sets the RF and the rule-selection slice; the seed-rule sample uses seed 42"),
        "rule_selection_rows": int(len(X_rule_sel)),
        "rule_selection_pos": int(y_rule_sel.sum()),
        "seed_rules_in_prompt": json.loads(seed_text),
        "n_candidates_parsed": _n_parsed,
        "seed_fallback_used": _fallback_used,
        "n_rejected_missing_columns": len(_rejected),
        "rejected": _rejected,
        "n_rule_columns": len(rule_cols),
        "llm_raw_response": (response_text if _llm_generated else None),
    }, _f, indent=2)
print(f"[rules] generation record → {PARSED_RULES_GEN_JSON}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — RULE TABLE BUILDING
# ═══════════════════════════════════════════════════════════════════════════
RULE_TABLE_CACHE = (os.path.join(RULES_OUT_DIR, "customer_rule_table.pt") if RULES_OUT_DIR
                    else amazon_data("amazon_customer_rule_table.pt"))

customer_df  = rdl_db.table_dict["customer"].df.copy()
customer_df["customer_id"] = customer_df["customer_id"].astype(np.int64)
cid_to_nidx  = {int(cid): i for i, cid in enumerate(customer_df["customer_id"].tolist())}
n_cust_nodes = len(cid_to_nidx)
n_rules      = len(rule_cols)

print(f"\nCustomer nodes: {n_cust_nodes:,}  |  rule features: {n_rules}")

rule_table = torch.zeros(n_cust_nodes, max(n_rules, 1), dtype=torch.float32)
if n_rules > 0:
    mat  = result_df[rule_cols].to_numpy(dtype=np.float32)
    cids = result_df["customer_id"].to_numpy(dtype=np.int64)
    for row_i, cid in enumerate(cids):
        nidx = cid_to_nidx.get(int(cid))
        if nidx is not None:
            rule_table[nidx] = torch.from_numpy(mat[row_i])
    coverage = int((rule_table[:, :n_rules].any(dim=1)).sum()) / n_cust_nodes
    print(f"Rule coverage: {coverage:.2%} of customer nodes")

torch.save({"rule_table": rule_table, "rule_cols": rule_cols, "n_rules": n_rules}, RULE_TABLE_CACHE)
print(f"Rule table saved → {RULE_TABLE_CACHE}  shape={tuple(rule_table.shape)}")

# Optional semi-supervised boost: oversample positive train rows so each epoch
# exposes more anomaly examples to ranking loss.
RAD_POS_OVERSAMPLE = int(os.environ.get("RAD_POS_OVERSAMPLE", "1"))
_loader_train_table = rdl_train_table
if RAD_POS_OVERSAMPLE > 1:
    _pos_df = rdl_train_table.df[rdl_train_table.df["churn"] == 1]
    if len(_pos_df) > 0:
        _loader_train_table = copy.deepcopy(rdl_train_table)
        _loader_train_table.df = pd.concat(
            [_loader_train_table.df] + [_pos_df] * (RAD_POS_OVERSAMPLE - 1),
            ignore_index=True,
        )
        _loader_train_table.df = _loader_train_table.df.sample(
            frac=1.0, random_state=RUN_SEED
        ).reset_index(drop=True)
        print(
            f"Train loader positive oversampling enabled: x{RAD_POS_OVERSAMPLE} | "
            f"rows={len(_loader_train_table.df):,} | "
            f"pos_rate={_loader_train_table.df['churn'].mean():.4f}"
        )
    else:
        print("Train loader positive oversampling requested, but no positives available.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 — BUILD SHARED LOADERS (num_workers=0 for script stability)
# ═══════════════════════════════════════════════════════════════════════════
def _make_loader(table, split):
    ti = get_node_train_table_input(table=table, task=rdl_task)
    return NeighborLoader(
        rdl_data,
        num_neighbors=[32, 32],
        time_attr="time",
        input_nodes=ti.nodes,
        input_time=ti.time,
        transform=ti.transform,
        batch_size=1024,
        temporal_strategy="uniform",
        shuffle=(split == "train"),
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
    )


loader_train = _make_loader(_loader_train_table, "train")
loader_val   = _make_loader(rdl_val_table, "val")
loader_test  = _make_loader(rdl_test_table, "test")
print(f"\nLoaders — train: {len(loader_train)} batches | val: {len(loader_val)} batches")

# Original-val loader for RAD final eval
_orig_ti = get_node_train_table_input(table=_rdl_orig_test_table, task=rdl_task)
loader_orig_test = NeighborLoader(
    rdl_data,
    num_neighbors=[32, 32], time_attr="time",
    input_nodes=_orig_ti.nodes, input_time=_orig_ti.time, transform=_orig_ti.transform,
    batch_size=1024, temporal_strategy="uniform",
    shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12 — SUPERVISED GNN (Baseline)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 1: SUPERVISED RDL GNN")
print("  (no rule features — those are exclusive to RAD)")
print("=" * 70)

SUP_LOSS_MODE = os.environ.get("SUP_LOSS_MODE", "unweighted").strip().lower()
if SUP_LOSS_MODE != "unweighted":
    raise ValueError(f"Unsupported SUP_LOSS_MODE={SUP_LOSS_MODE!r}; prevalence sweeps require unweighted supervised RDL.")
print("Supervised RDL loss: unweighted BCE")

seed_everything(RUN_SEED)

class SupervisedGNN(torch.nn.Module):
    def __init__(self, data, col_stats, channels=128, num_layers=2, aggr="sum"):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if data[nt].get("time") is not None],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types, edge_types=data.edge_types,
            channels=channels, aggr=aggr, num_layers=num_layers,
        )
        self.head = torch.nn.Linear(channels, 1)

    def forward(self, batch, entity_table):
        x_dict = self.encoder(batch.tf_dict)
        rel_t  = batch[entity_table].seed_time
        t_dict = self.temporal_encoder(rel_t, batch.time_dict, batch.batch_dict)
        x_dict = {k: (x_dict[k] + t_dict[k]) if k in t_dict else x_dict[k] for k in x_dict}
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            num_sampled_nodes_dict=batch.num_sampled_nodes_dict,
            num_sampled_edges_dict=batch.num_sampled_edges_dict,
        )
        out    = self.head(x_dict[entity_table]).squeeze(-1)
        seed_n = batch[entity_table].batch_size
        return out[:seed_n]


sup_model = SupervisedGNN(rdl_data, rdl_col_stats_dict, channels=128, num_layers=2).to(device)
sup_opt   = torch.optim.AdamW(sup_model.parameters(), lr=1e-3, weight_decay=1e-4)
print(f"SupervisedGNN | params={sum(p.numel() for p in sup_model.parameters() if p.requires_grad):,}")

SUP_EPOCHS    = int(os.environ.get("SUP_EPOCHS", "10"))
SUP_MAX_STEPS = 2000
SUP_PATIENCE  = int(os.environ.get("SUP_PATIENCE", "3"))
SUP_GRAD_CLIP = 2.0

_best_sup_auprc = -1.0
_best_sup_sd    = None
_sup_patience   = 0

for _epoch in range(1, SUP_EPOCHS + 1):
    sup_model.train()
    _steps, _run_loss = 0, 0.0
    for _batch in tqdm(loader_train, desc=f"  SUP epoch {_epoch:02d}", leave=False):
        _batch  = _batch.to(device)
        _logits = sup_model(_batch, rdl_entity_table)
        _lbl    = _batch[rdl_entity_table].y.float().view(-1)[:_logits.shape[0]]
        _loss   = F.binary_cross_entropy_with_logits(_logits, _lbl)
        sup_opt.zero_grad(set_to_none=True)
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(sup_model.parameters(), SUP_GRAD_CLIP)
        sup_opt.step()
        _run_loss += float(_loss.detach()); _steps += 1
        if _steps >= SUP_MAX_STEPS: break

    # Val pass
    sup_model.eval()
    _sup_scores = []
    with torch.no_grad():
        for _vb in loader_val:
            _vb = _vb.to(device)
            _sc = torch.sigmoid(sup_model(_vb, rdl_entity_table)).detach().cpu().numpy()
            _sup_scores.append(_sc)
    _vscores = np.concatenate(_sup_scores)
    _vap  = float(average_precision_score(_val_labels, _vscores)) if np.unique(_val_labels).size > 1 else float("nan")
    _vauc = float(roc_auc_score(_val_labels, _vscores))           if np.unique(_val_labels).size > 1 else float("nan")
    print(f"  SUP epoch {_epoch:02d} | steps={_steps} | loss={_run_loss/_steps:.4f} | "
          f"val_auprc={_vap:.4f} | val_auroc={_vauc:.4f}")

    if np.isfinite(_vap) and _vap > _best_sup_auprc + 1e-8:
        _best_sup_auprc = _vap
        _best_sup_sd    = {k: v.detach().cpu().clone() for k, v in sup_model.state_dict().items()}
        _sup_patience   = 0
    else:
        _sup_patience += 1
        if _sup_patience >= SUP_PATIENCE:
            print(f"  Early stopping at epoch {_epoch}"); break

# Best checkpoint → final holdout-test scores
if _best_sup_sd is not None:
    sup_model.load_state_dict(_best_sup_sd)
sup_model.eval()
_final_sup_scores = []
with torch.no_grad():
    for _vb in loader_test:
        _vb = _vb.to(device)
        _sc = torch.sigmoid(sup_model(_vb, rdl_entity_table)).detach().cpu().numpy()
        _final_sup_scores.append(_sc)
_final_sup_scores = np.concatenate(_final_sup_scores)
sup_auprc = float(average_precision_score(_test_labels, _final_sup_scores)) if np.unique(_test_labels).size > 1 else float("nan")
sup_auroc = float(roc_auc_score(_test_labels, _final_sup_scores))           if np.unique(_test_labels).size > 1 else float("nan")

# ── Metric helpers ────────────────────────────────────────────────────────
def _p_at_k(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    k = int(max(1, y.sum()))
    return float(y[np.argsort(-s)[:k]].mean())

def _lift(auprc, y_true):
    prev = float(np.asarray(y_true).mean())
    return float(auprc / prev) if prev > 0 else float("nan")

sup_pak  = _p_at_k(_test_labels, _final_sup_scores) if np.unique(_test_labels).size > 1 else float("nan")
sup_lift = _lift(sup_auprc, _test_labels)

# Original (unfiltered) test labels for reference
_orig_lbl = _rdl_orig_test_table.df["churn"].astype(np.int64).to_numpy()
_final_sup_orig_scores = []
with torch.no_grad():
    for _vb in loader_orig_test:
        _vb = _vb.to(device)
        _sc = torch.sigmoid(sup_model(_vb, rdl_entity_table)).detach().cpu().numpy()
        _final_sup_orig_scores.append(_sc)
_final_sup_orig_scores = np.concatenate(_final_sup_orig_scores)
sup_orig_auprc = float(average_precision_score(_orig_lbl, _final_sup_orig_scores)) if np.unique(_orig_lbl).size > 1 else float("nan")
sup_orig_auroc = float(roc_auc_score(_orig_lbl, _final_sup_orig_scores))           if np.unique(_orig_lbl).size > 1 else float("nan")

print(f"\n  Supervised GNN val (at-risk filtered)  AUPRC={sup_auprc:.4f}  AUROC={sup_auroc:.4f}")

# Free supervised model memory before RAD
del sup_model, sup_opt, _best_sup_sd
gc.collect()
torch.cuda.empty_cache()
print("  Supervised GNN freed from GPU.\n")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13 — RAD AUTOENCODER
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 2: RAD AUTOENCODER (with Rule Augmentation)")
print("=" * 70)


class RADAutoencoder(torch.nn.Module):
    def __init__(self, data, col_stats_dict, num_layers, channels, aggr, norm,
                 rule_dim=0, latent_dim=48, decoder_num_layers=None, decoder_head_layers=2):
        super().__init__()
        if decoder_num_layers is None:
            decoder_num_layers = num_layers
        if latent_dim >= channels:
            raise ValueError(f"latent_dim ({latent_dim}) must be < channels ({channels})")
        self.channels   = channels
        self.latent_dim = latent_dim
        self.rule_dim   = rule_dim
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]], channels=channels)
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types, edge_types=data.edge_types,
            channels=channels, aggr=aggr, num_layers=num_layers)
        self.to_latent = nn.Linear(channels, latent_dim)
        self.decoder_gnn = HeteroGraphSAGE(
            node_types=data.node_types, edge_types=data.edge_types,
            channels=latent_dim, aggr=aggr, num_layers=decoder_num_layers)
        self.from_latent  = nn.Linear(latent_dim, channels)
        self.feat_decoder = MLP(channels, hidden_channels=channels, out_channels=channels,
                                norm=norm, num_layers=decoder_head_layers)
        self.rel_diag = nn.ParameterDict(
            {self._ek(et): nn.Parameter(torch.ones(latent_dim)) for et in data.edge_types})
        self.rule_proj = nn.Linear(rule_dim, latent_dim, bias=False) if rule_dim > 0 else None
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.decoder_gnn.reset_parameters()
        self.feat_decoder.reset_parameters()
        nn.init.xavier_uniform_(self.to_latent.weight);  nn.init.zeros_(self.to_latent.bias)
        nn.init.xavier_uniform_(self.from_latent.weight); nn.init.zeros_(self.from_latent.bias)
        for p in self.rel_diag.values(): nn.init.ones_(p)
        if self.rule_proj is not None: nn.init.xavier_uniform_(self.rule_proj.weight)

    @staticmethod
    def _ek(et): return f"{et[0]}__{et[1]}__{et[2]}"

    def _encode(self, batch, entity_table):
        seed_time = batch[entity_table].seed_time
        x = self.encoder(batch.tf_dict)
        for nt, rt in self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict).items():
            x[nt] = x[nt] + rt
        pre     = {k: v.detach() for k, v in x.items()}
        z       = self.gnn(x, batch.edge_index_dict,
                           batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict)
        z_lat   = {k: self.to_latent(v) for k, v in z.items()}
        return pre, z_lat, seed_time.size(0)

    def forward(self, batch, entity_table, rule_table=None):
        pre, z_lat, n = self._encode(batch, entity_table)
        z_dec_in = {k: v for k, v in z_lat.items()}
        if rule_table is not None and self.rule_proj is not None:
            node_ids = batch[entity_table].n_id
            z_dec_in[entity_table] = (z_dec_in[entity_table]
                                      + self.rule_proj(rule_table[node_ids].to(z_dec_in[entity_table].device)))
        z_dec     = self.decoder_gnn(z_dec_in, batch.edge_index_dict,
                                     batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict)
        seed_dec  = self.from_latent(z_dec[entity_table][:n])
        return self.feat_decoder(seed_dec), pre[entity_table][:n], z_dec, n

    def score_edges(self, z, et, ei):
        k = self._ek(et)
        if k not in self.rel_diag or ei.numel() == 0:
            return ei.new_zeros(ei.size(1), dtype=torch.float32)
        d = self.rel_diag[k]
        return (z[et[0]][ei[0]] * d * z[et[2]][ei[1]]).sum(dim=1)

    def structure_bce_loss(self, z, batch, neg_ratio=1.0):
        losses = []
        for et in batch.edge_types:
            ei = batch[et].edge_index
            if ei.numel() == 0 or self._ek(et) not in self.rel_diag: continue
            ns, nd = int(z[et[0]].size(0)), int(z[et[2]].size(0))
            if ns == 0 or nd == 0: continue
            n_neg  = max(1, int(neg_ratio * ei.size(1)))
            neg_ei = torch.stack([
                torch.randint(0, ns, (n_neg,), device=ei.device),
                torch.randint(0, nd, (n_neg,), device=ei.device),
            ])
            losses.append(0.5 * (
                F.binary_cross_entropy_with_logits(self.score_edges(z, et, ei),
                                                   torch.ones_like(self.score_edges(z, et, ei)))
                + F.binary_cross_entropy_with_logits(self.score_edges(z, et, neg_ei),
                                                     torch.zeros_like(self.score_edges(z, et, neg_ei)))
            ))
        return torch.stack(losses).mean() if losses else next(iter(z.values())).new_zeros(())

    def seed_struct_error(self, z, batch, n_seed, entity_table):
        dev  = z[entity_table].device
        es   = torch.zeros(n_seed, device=dev)
        ec   = torch.zeros(n_seed, device=dev)
        for et in batch.edge_types:
            ei = batch[et].edge_index
            if ei.numel() == 0 or self._ek(et) not in self.rel_diag: continue
            err = F.softplus(-self.score_edges(z, et, ei))
            for side, idx_row in [(et[0], 0), (et[2], 1)]:
                if side == entity_table:
                    v = ei[idx_row] < n_seed
                    if v.any():
                        es.index_add_(0, ei[idx_row][v], err[v])
                        ec.index_add_(0, ei[idx_row][v], torch.ones(v.sum(), device=dev))
        return es / torch.clamp(ec, min=1.0)


def pairwise_rank_loss(scores, labels, max_pairs=32768):
    scores, labels = scores.view(-1), labels.view(-1)
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.new_zeros(()), 0
    n  = min(int(pos.numel()) * int(neg.numel()), max_pairs)
    pi = torch.randint(0, int(pos.numel()), (n,), device=scores.device)
    ni = torch.randint(0, int(neg.numel()), (n,), device=scores.device)
    return F.softplus(-(pos[pi] - neg[ni])).mean(), n


# ── Hyperparams + model build ─────────────────────────────────────────────
_latent_dim_env = os.environ.get("LATENT_DIM", "32")
_hp = dict(
    channels=128, latent_dim=int(_latent_dim_env), num_layers=2, decoder_num_layers=2, decoder_head_layers=2,
    aggr="sum", norm="batch_norm",
    lr=float(os.environ.get("LR", "1e-3")),
    weight_decay=float(os.environ.get("WEIGHT_DECAY", "1e-4")),
    epochs=int(os.environ.get("EPOCHS", "60")),
    patience=int(os.environ.get("PATIENCE", "10")),
    max_steps_train=int(os.environ.get("MAX_STEPS_TRAIN", "2000")),
    grad_clip=float(os.environ.get("GRAD_CLIP", "2.0")),
    lambda_struct=float(os.environ.get("LAMBDA_STRUCT", "0.1")),
    lambda_rank=float(os.environ.get("LAMBDA_RANK", "2.0")), rank_max_pairs=32768,
    neg_ratio=1.0,
    alpha_feat=float(os.environ.get("ALPHA_FEAT", "0.5")),
    alpha_struct=float(os.environ.get("ALPHA_STRUCT", "2.0")),
)

_warmup_epochs_env = os.environ.get("WARMUP_EPOCHS", "")
if _warmup_epochs_env.strip() == "":
    _hp["warmup_epochs"] = min(15, max(0, _hp["epochs"] // 3))
else:
    _hp["warmup_epochs"] = int(_warmup_epochs_env)

_hp["warmup_rank_scale"] = float(os.environ.get("WARMUP_RANK_SCALE", "1.0"))
_hp["warmup_struct_scale"] = float(os.environ.get("WARMUP_STRUCT_SCALE", "1.0"))

if os.path.exists(RULE_TABLE_CACHE):
    _rt = torch.load(RULE_TABLE_CACHE, weights_only=False)
    rule_table = _rt["rule_table"]; rule_cols = _rt["rule_cols"]; n_rules = _rt["n_rules"]
    print(f"Rule table loaded: {tuple(rule_table.shape)}")
else:
    rule_table, rule_cols, n_rules = None, [], 0

RAD_DISABLE_RULES = os.environ.get("RAD_DISABLE_RULES", "0").strip() == "1"
if RAD_DISABLE_RULES:
    rule_table, rule_cols, n_rules = None, [], 0
    print("RAD rules disabled via RAD_DISABLE_RULES=1 (rule_dim=0).")

seed_everything(RUN_SEED)
rad_model = RADAutoencoder(
    rdl_data, rdl_col_stats_dict,
    num_layers=_hp["num_layers"], channels=_hp["channels"], aggr=_hp["aggr"], norm=_hp["norm"],
    rule_dim=n_rules, latent_dim=_hp["latent_dim"],
    decoder_num_layers=_hp["decoder_num_layers"], decoder_head_layers=_hp["decoder_head_layers"],
).to(device)
rad_opt = torch.optim.AdamW(rad_model.parameters(), lr=_hp["lr"], weight_decay=_hp["weight_decay"])
print(f"RADAutoencoder | channels={_hp['channels']} | latent={_hp['latent_dim']} | "
      f"rule_dim={n_rules} | "
      f"params={sum(p.numel() for p in rad_model.parameters() if p.requires_grad):,}")

_rt_gpu = rule_table.to(device) if rule_table is not None else None


def _train_epoch(lambda_struct: float, lambda_rank: float):
    rad_model.train()
    S = dict(loss=0., feat=0., struct=0., rank=0., n=0, pairs=0, steps=0)
    feat_buf, struct_buf = [], []
    for batch in tqdm(loader_train, desc="  RAD train", leave=False):
        batch = batch.to(device)
        rad_opt.zero_grad(set_to_none=True)
        recon, target, z, n = rad_model(batch, rdl_entity_table, _rt_gpu)
        lbl  = batch[rdl_entity_table].y.float().view(-1)
        norm = lbl < 0.5
        if not norm.any(): continue
        feat_loss   = F.mse_loss(recon[norm], target[norm])
        struct_loss = rad_model.structure_bce_loss(z, batch, _hp["neg_ratio"])
        feat_err    = ((recon - target) ** 2).mean(dim=1)
        struct_err  = rad_model.seed_struct_error(z, batch, n, rdl_entity_table)
        feat_norm = feat_err[norm].detach()
        struct_norm = struct_err[norm].detach()
        feat_mu = feat_norm.mean()
        feat_sd = torch.clamp(feat_norm.std(unbiased=False), min=1e-8)
        struct_mu = struct_norm.mean()
        struct_sd = torch.clamp(struct_norm.std(unbiased=False), min=1e-8)
        combined = (
            _hp["alpha_feat"] * ((feat_err - feat_mu) / feat_sd)
            + _hp["alpha_struct"] * ((struct_err - struct_mu) / struct_sd)
        )
        rank_loss, np_ = pairwise_rank_loss(combined, lbl, _hp["rank_max_pairs"])
        loss = feat_loss + lambda_struct * struct_loss + lambda_rank * rank_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rad_model.parameters(), _hp["grad_clip"])
        rad_opt.step()
        cnt = int(norm.sum())
        S["loss"]   += float(loss.detach())        * cnt
        S["feat"]   += float(feat_loss.detach())   * cnt
        S["struct"] += float(struct_loss.detach()) * cnt
        S["rank"]   += float(rank_loss.detach())   * cnt
        S["n"] += cnt; S["pairs"] += np_; S["steps"] += 1
        feat_buf.append(feat_err[norm].detach().cpu())
        struct_buf.append(struct_err[norm].detach().cpu())
        if _hp["max_steps_train"] and S["steps"] >= _hp["max_steps_train"]: break
    if S["n"] == 0: raise RuntimeError("No normal training samples.")
    all_f = torch.cat(feat_buf).numpy(); all_s = torch.cat(struct_buf).numpy()
    ns_stats = dict(feat_mu=float(np.mean(all_f)), feat_sd=float(np.std(all_f)),
                    struct_mu=float(np.mean(all_s)), struct_sd=float(np.std(all_s)))
    return {k: (v / S["n"] if k in ("loss", "feat", "struct", "rank") else v) for k, v in S.items()}, ns_stats


@torch.no_grad()
def _score(ldr, max_batches=None):
    rad_model.eval()
    feats, structs = [], []
    for i, batch in enumerate(ldr):
        if max_batches and i >= max_batches: break
        batch = batch.to(device)
        recon, target, z, n = rad_model(batch, rdl_entity_table, _rt_gpu)
        feats.append(((recon - target) ** 2).mean(dim=1).cpu())
        structs.append(rad_model.seed_struct_error(z, batch, n, rdl_entity_table).cpu())
    return torch.cat(feats).numpy(), torch.cat(structs).numpy()


def _combine(fe, se, ns):
    fz = (fe - ns["feat_mu"]) / max(ns["feat_sd"], 1e-8)
    sz = (se - ns["struct_mu"]) / max(ns["struct_sd"], 1e-8)
    return _hp["alpha_feat"] * fz + _hp["alpha_struct"] * sz


# ── Train RAD ────────────────────────────────────────────────────────────
best_metric, best_state, bad = -math.inf, None, 0
norm_stats = dict(feat_mu=0., feat_sd=1., struct_mu=0., struct_sd=1.)

for epoch in range(1, _hp["epochs"] + 1):
    in_warmup = epoch <= _hp["warmup_epochs"]
    cur_lambda_struct = _hp["lambda_struct"] * (_hp["warmup_struct_scale"] if in_warmup else 1.0)
    cur_lambda_rank = _hp["lambda_rank"] * (_hp["warmup_rank_scale"] if in_warmup else 1.0)

    tr, norm_stats = _train_epoch(cur_lambda_struct, cur_lambda_rank)
    vf, vs   = _score(loader_val)
    val_pred = _combine(vf, vs, norm_stats)
    val_met  = rdl_task.evaluate(val_pred, rdl_val_table)
    val_aupr = average_precision_score(_val_labels, val_pred)
    val_auroc_ep = roc_auc_score(_val_labels, val_pred) if np.unique(_val_labels).size > 1 else float("nan")
    val_target = val_aupr if rdl_tune_metric == "auprc" else val_met["roc_auc"]
    stage = "warmup" if in_warmup else "main"
    print(f"  RAD epoch {epoch:02d}/{_hp['epochs']} [{stage}] | steps={tr['steps']} | "
          f"loss={tr['loss']:.4f} | feat={tr['feat']:.4f} | struct={tr['struct']:.4f} | "
          f"rank={tr['rank']:.4f} | ls={cur_lambda_struct:.3f} | lrk={cur_lambda_rank:.3f} | "
          f"val_auprc={val_aupr:.4f} | val_auroc={val_auroc_ep:.4f}")
    if val_target > best_metric:
        best_metric = val_target
        best_state  = {k: v.detach().cpu().clone() for k, v in rad_model.state_dict().items()}
        best_norm   = dict(norm_stats)
        bad = 0
    else:
        bad += 1
        if bad >= _hp["patience"]:
            print(f"  Early stopping at epoch {epoch}."); break

rad_model.load_state_dict(best_state)
print(f"\n  RAD best val {rdl_tune_metric}: {best_metric:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14 — FINAL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
ns = best_norm
vf, vs   = _score(loader_test)
val_pred = _combine(vf, vs, ns)
rad_auprc = float(average_precision_score(_test_labels, val_pred)) if np.unique(_test_labels).size > 1 else float("nan")
rad_auroc = float(roc_auc_score(_test_labels, val_pred))           if np.unique(_test_labels).size > 1 else float("nan")

# Original (unfiltered) test labels
orig_vf, orig_vs   = _score(loader_orig_test)
orig_val_pred      = _combine(orig_vf, orig_vs, ns)
orig_rad_auprc     = float(average_precision_score(_orig_lbl, orig_val_pred)) if np.unique(_orig_lbl).size > 1 else float("nan")
orig_rad_auroc     = float(roc_auc_score(_orig_lbl, orig_val_pred))           if np.unique(_orig_lbl).size > 1 else float("nan")

# ─── Save RAD weights ────────────────────────────────────────────────────
save_path = os.environ.get("SAVE_RAW_RESULTS_PATH", "weights/rad_amazon_user_churn.pt")
os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
save_obj = {
    "model_state_dict": rad_model.state_dict(),
    "optimizer_state_dict": rad_opt.state_dict(),
    "best_val_metric": best_metric,
    "norm_stats": ns,
    "hyperparams": _hp,
    "at_risk_thresholds": rdl_at_risk_thresholds,
    "rule_cols": rule_cols,
    "n_rules": n_rules,
    "test_scores": val_pred,
    "test_labels": _test_labels,
    "orig_test_scores": orig_val_pred,
    "orig_test_labels": _orig_lbl,
    "rf_detector_scores": rf_detector_scores,
}
if "_final_sup_scores" in locals():
    save_obj["sup_val_scores"] = _final_sup_scores
if "_final_sup_orig_scores" in locals():
    save_obj["sup_orig_val_scores"] = _final_sup_orig_scores
torch.save(save_obj, save_path)
print(f"Saved → {save_path}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 15 — COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════
rad_pak  = _p_at_k(_test_labels, val_pred)
rad_lift = _lift(rad_auprc, _test_labels)

rf_det_auprc = float(average_precision_score(_test_labels, rf_detector_scores)) if np.unique(_test_labels).size > 1 else float("nan")
rf_det_auroc = float(roc_auc_score(_test_labels, rf_detector_scores))           if np.unique(_test_labels).size > 1 else float("nan")
rf_det_pak   = _p_at_k(_test_labels, rf_detector_scores) if np.unique(_test_labels).size > 1 else float("nan")
rf_det_lift  = _lift(rf_det_auprc, _test_labels)

mlp_det_auprc = float(average_precision_score(_test_labels, mlp_detector_scores)) if np.unique(_test_labels).size > 1 else float("nan")
mlp_det_auroc = float(roc_auc_score(_test_labels, mlp_detector_scores))           if np.unique(_test_labels).size > 1 else float("nan")
mlp_det_pak   = _p_at_k(_test_labels, mlp_detector_scores) if np.unique(_test_labels).size > 1 else float("nan")
mlp_det_lift  = _lift(mlp_det_auprc, _test_labels)

print("\n" + "=" * 70)
print(f"FINAL COMPARISON ({_eval_split_name} Set — At-Risk Filtered Labels)")
print("=" * 70)
print(f"  {_eval_split_name}:  {len(rdl_test_table.df):,} users  |  {int(_test_labels.sum()):,} anomalies  "
    f"({_test_labels.mean():.4f} rate)")
print(f"  eval_fingerprint: {eval_fingerprint(rdl_test_table.df, _test_labels)}")
print()
print(f"{'Model':<30}  {'AUPRC':>8}  {'AUROC':>8}  {'P@K':>8}  {'Lift':>8}")
print("-" * 72)
print(f"{'Supervised RDL GNN':<30}  {sup_auprc:>8.4f}  {sup_auroc:>8.4f}  {sup_pak:>8.4f}  {sup_lift:>8.4f}  (graph only — no rules)")
print(f"{'RF (supervised detector)':<30}  {rf_det_auprc:>8.4f}  {rf_det_auroc:>8.4f}  {rf_det_pak:>8.4f}  {rf_det_lift:>8.4f}  (same features/supervision as rules — no graph)")
print(f"{'MLP (supervised detector)':<30}  {mlp_det_auprc:>8.4f}  {mlp_det_auroc:>8.4f}  {mlp_det_pak:>8.4f}  {mlp_det_lift:>8.4f}  (same features/supervision as rules — no graph)")
print(f"{'RAD (rule-augmented)':<30}  {rad_auprc:>8.4f}  {rad_auroc:>8.4f}  {rad_pak:>8.4f}  {rad_lift:>8.4f}  (graph + {n_rules} rule features)")
print("=" * 70)
print(f"FINAL COMPARISON ({_eval_split_name} Set — Original Unfiltered Labels)")
print("=" * 70)
print(f"  {_eval_split_name}:  {len(_rdl_orig_test_table.df):,} users  |  {int(_orig_lbl.sum()):,} anomalies  "
    f"({_orig_lbl.mean():.4f} rate)")
print()
print(f"{'Model':<30}  {'AUPRC':>8}  {'AUROC':>8}")
print("-" * 52)
print(f"{'Supervised RDL GNN':<30}  {sup_orig_auprc:>8.4f}  {sup_orig_auroc:>8.4f}  (graph only — no rules)")
print(f"{'RAD (rule-augmented)':<30}  {orig_rad_auprc:>8.4f}  {orig_rad_auroc:>8.4f}  (graph + {n_rules} rule features)")
print(f"Sweep prevalence target: {_target_anom_rate:.4f}")
print(f"Latent dimension: {_hp['latent_dim']}")
print(f"At-risk thresholds: {rdl_at_risk_thresholds}")
print(f"Rule features used: {n_rules}")
print("=" * 70)

_common = {"task": "amazon-user-churn", "seed": RUN_SEED, "split": _eval_split_name, "rule_source": RF_RULE_MODE, "n_rules": n_rules}
for _method, _auprc, _auroc, _pak, _lift_v in [
    ("Supervised RDL GNN", sup_auprc, sup_auroc, sup_pak, sup_lift),
    ("RF (supervised detector)", rf_det_auprc, rf_det_auroc, rf_det_pak, rf_det_lift),
    ("MLP (supervised detector)", mlp_det_auprc, mlp_det_auroc, mlp_det_pak, mlp_det_lift),
    ("RAD (rule-augmented)", rad_auprc, rad_auroc, rad_pak, rad_lift),
]:
    append_results_jsonl({
        **_common, "method": _method,
        "auprc": _auprc, "auroc": _auroc, "p@k": _pak, "lift": _lift_v,
    })
