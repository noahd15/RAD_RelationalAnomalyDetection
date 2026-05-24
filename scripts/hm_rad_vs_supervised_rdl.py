#!/usr/bin/env python3
"""
RAD vs Supervised RDL — H&M User-Churn (At-Risk Filtering)
Trains both models on the same at-risk-filtered H&M dataset and compares side-by-side.
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
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from path_helpers import hm_data

ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)

RUN_SEED = int(os.environ.get("SEED", "42"))
POS_DOWNSAMPLE_MODE = os.environ.get("POS_DOWNSAMPLE_MODE", "random").strip().lower()

IDENTITY_PROXY_FEATURES = {"customer": {"postal_code"}}

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
rdl_graph_cache_path = hm_data("relbench_rdl_graph_cache_hm.pt")
rdl_stats_cache_path = hm_data("relbench_rdl_col_stats_cache_hm.pt")

def _sanitize_graph(data):
    for ntype in data.node_types:
        node = data[ntype]
        if hasattr(node, "tf") and node.tf is not None and hasattr(node.tf, "feat_dict"):
            for key, t in node.tf.feat_dict.items():
                if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                    node.tf.feat_dict[key] = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
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
    rdl_data         = _mask_identity_proxy_features(_sanitize_graph(torch.load(rdl_graph_cache_path, weights_only=False)))
    rdl_col_stats_dict = torch.load(rdl_stats_cache_path, weights_only=False)
    print("Loaded H&M graph cache from disk.")
else:
    rdl_data = None
    rdl_col_stats_dict = None
    print("No H&M cache found — will build graph from scratch.")

rdl_dataset = get_dataset("rel-hm", download=True)
rdl_task    = get_task("rel-hm", "user-churn", download=True)
rdl_db      = rdl_dataset.get_db()

rdl_train_table = rdl_task.get_table("train")
rdl_val_table   = rdl_task.get_table("val")
rdl_test_table  = rdl_task.get_table("test")

_rdl_orig_val_table   = copy.deepcopy(rdl_val_table)
_rdl_orig_train_table = copy.deepcopy(rdl_train_table)
_rdl_orig_test_table  = copy.deepcopy(rdl_test_table)

_eval_split_name = "Test"
if "churn" not in rdl_test_table.df.columns:
    # RelBench test labels are hidden for leaderboard usage; fallback to val for local evaluation.
    _eval_split_name = "Val"
    rdl_test_table = copy.deepcopy(rdl_val_table)
    _rdl_orig_test_table = copy.deepcopy(_rdl_orig_val_table)
    print("[warn] test split has no 'churn' labels; using validation split for final evaluation.")

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
txn_events = rdl_db.table_dict["transactions"].df[["customer_id", "t_dat"]].dropna().copy()
txn_events["customer_id"] = txn_events["customer_id"].astype(np.int64)
txn_events["t_dat"]       = pd.to_datetime(txn_events["t_dat"])
txn_events = txn_events.sort_values(["customer_id", "t_dat"]).reset_index(drop=True)
txn_events["cum_txns"] = txn_events.groupby("customer_id").cumcount() + 1

# Shared disk cache for causal per-timestamp price/article aggregations
_PA_CACHE_PATH = hm_data("hm_causal_price_article_cache.pkl")
import pickle as _pickle
_pa_cache: dict = {}
if os.path.exists(_PA_CACHE_PATH):
    with open(_PA_CACHE_PATH, "rb") as _f:
        _pa_cache = _pickle.load(_f)
    print(f"Loaded causal price/article cache: {len(_pa_cache)} timestamps")


def _pa_cache_key(ts) -> str:
    return pd.Timestamp(ts).isoformat()


def _compute_pa_for_ts(ts):
    txn_before = txn_df[txn_df["t_dat"] < ts]
    price_agg, art_agg = None, None
    if _price_col is not None and not txn_before.empty:
        rated = txn_before[["customer_id", _price_col]].copy()
        rated[_price_col] = pd.to_numeric(rated[_price_col], errors="coerce")
        rated = rated.dropna(subset=[_price_col])
        price_agg = rated.groupby("customer_id")[_price_col].agg(
            mean_price="mean", price_std="std", min_price="min", max_price="max"
        ).reset_index()
    if "article_id" in txn_df.columns and not txn_before.empty:
        art_agg = (txn_before.groupby("customer_id")["article_id"]
                   .nunique().reset_index(name="distinct_articles"))
    return price_agg, art_agg


_ATRISK_A     = float(os.environ.get("ATRISK_TAU_A",     "0.05"))
_ATRISK_B     = float(os.environ.get("ATRISK_TAU_B",     "10.0"))
_ATRISK_ALPHA = float(os.environ.get("ATRISK_TAU_ALPHA", "0.02"))


def _activity_features_for_targets(target_df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(np.int64)
    q["timestamp"]   = pd.to_datetime(q["timestamp"])
    q["_row_id"]     = np.arange(len(q), dtype=np.int64)
    q = q.sort_values(["timestamp", "customer_id", "_row_id"]).reset_index(drop=True)
    right = txn_events[["customer_id", "t_dat", "cum_txns"]]

    now_join = pd.merge_asof(
        q[["customer_id", "timestamp", "_row_id"]].sort_values(["timestamp", "customer_id"]),
        right.sort_values(["t_dat", "customer_id"]),
        left_on="timestamp", right_on="t_dat",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    q["txns_before_now"] = now_join["cum_txns"].fillna(0).astype(np.int64)

    # Recent 30-day window [T-30, T)
    q["window_start"] = q["timestamp"] - pd.Timedelta(days=lookback_days)
    start_join = pd.merge_asof(
        q[["customer_id", "window_start", "_row_id"]].sort_values(["window_start", "customer_id"]),
        right.sort_values(["t_dat", "customer_id"]),
        left_on="window_start", right_on="t_dat",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    _w30 = start_join["cum_txns"].fillna(0).astype(np.int64)
    q["recent_txns"] = np.maximum(0, q["txns_before_now"].values - _w30.values)

    # Previous 30-day window [T-60, T-30) for trend
    q["window_60"] = q["timestamp"] - pd.Timedelta(days=2 * lookback_days)
    w60_join = pd.merge_asof(
        q[["customer_id", "window_60", "_row_id"]].sort_values(["window_60", "customer_id"]),
        right.sort_values(["t_dat", "customer_id"]),
        left_on="window_60", right_on="t_dat",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    _w60 = w60_join["cum_txns"].fillna(0).astype(np.int64)
    q["prev_txns_30d"] = np.maximum(0, _w30.values - _w60.values)
    q["trend_w"]       = ((q["recent_txns"].values - q["prev_txns_30d"].values)
                          / (q["prev_txns_30d"].values + 1.0))
    q["avg_activity"]  = 0.5 * (q["recent_txns"].values + q["prev_txns_30d"].values)

    q = q.sort_values("_row_id").reset_index(drop=True)
    return q[["recent_txns", "prev_txns_30d", "trend_w", "avg_activity"]]


def _identify_at_risk_users(target_df, activity_df, thresholds=None):
    raw   = target_df["churn"].astype(np.int64).to_numpy()
    trend = activity_df["trend_w"].to_numpy(dtype=np.float64)
    a_bar = np.maximum(activity_df["avg_activity"].to_numpy(dtype=np.float64), 0.0)
    if thresholds is None:
        thresholds = {
            "a": _ATRISK_A,
            "b": _ATRISK_B,
            "alpha": _ATRISK_ALPHA,
            "tau_multiplier": float(os.environ.get("ATRISK_TAU_MULTIPLIER", "1.0")),
        }
    a, b, alpha = thresholds["a"], thresholds["b"], thresholds["alpha"]
    tau_multiplier = float(thresholds.get("tau_multiplier", 1.0))
    if tau_multiplier <= 0.0:
        raise ValueError(f"ATRISK_TAU_MULTIPLIER must be > 0, got {tau_multiplier}")
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


rdl_activity_train = _activity_features_for_targets(rdl_train_table.df, lookback_days=30)
rdl_activity_val   = _activity_features_for_targets(rdl_val_table.df,   lookback_days=30)
rdl_activity_test  = _activity_features_for_targets(rdl_test_table.df,  lookback_days=30)

rdl_train_anomaly_labels, rdl_train_keep, rdl_at_risk_thresholds = _identify_at_risk_users(
    rdl_train_table.df, rdl_activity_train)
rdl_val_anomaly_labels, rdl_val_keep, _ = _identify_at_risk_users(
    rdl_val_table.df, rdl_activity_val, thresholds=rdl_at_risk_thresholds)
rdl_test_anomaly_labels, rdl_test_keep, _ = _identify_at_risk_users(
    rdl_test_table.df, rdl_activity_test, thresholds=rdl_at_risk_thresholds)

rdl_train_table.df = rdl_train_table.df.loc[rdl_train_keep].copy()
rdl_train_table.df["churn"] = rdl_train_anomaly_labels[rdl_train_keep]
rdl_val_table.df   = rdl_val_table.df.loc[rdl_val_keep].copy()
rdl_val_table.df["churn"]   = rdl_val_anomaly_labels[rdl_val_keep]
rdl_test_table.df  = rdl_test_table.df.loc[rdl_test_keep].copy()
rdl_test_table.df["churn"]  = rdl_test_anomaly_labels[rdl_test_keep]

_target_anom_rate = float(os.environ.get("TARGET_ANOM_RATE", "0.015"))
if not (0.0 < _target_anom_rate < 1.0):
    raise ValueError(f"TARGET_ANOM_RATE must be in (0,1), got {_target_anom_rate}")

_rdl_train_pos_df = rdl_train_table.df[rdl_train_table.df["churn"] == 1]
_rdl_train_neg_df = rdl_train_table.df[rdl_train_table.df["churn"] == 0]
if len(_rdl_train_pos_df) > 0 and len(_rdl_train_neg_df) > 0:
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
if len(_rdl_val_pos_df) > 0 and len(_rdl_val_neg_df) > 0:
    _n_pos_tgt_vl = int(round((_target_anom_rate / (1.0 - _target_anom_rate)) * len(_rdl_val_neg_df)))
    _n_pos_tgt_vl = min(_n_pos_tgt_vl, len(_rdl_val_pos_df))
    if _n_pos_tgt_vl < len(_rdl_val_pos_df):
        _rng_vl = np.random.default_rng(RUN_SEED)
        _kp_idx_vl = _rng_vl.choice(_rdl_val_pos_df.index.to_numpy(), size=_n_pos_tgt_vl, replace=False)
        rdl_val_table.df = rdl_val_table.df.loc[
            np.sort(np.concatenate([_rdl_val_neg_df.index.to_numpy(), _kp_idx_vl]))].copy()

# When test labels are hidden in RelBench and we evaluate on validation,
# keep final-eval split aligned with the same filtered/downsampled val regime.
if _eval_split_name == "Val":
    rdl_test_table = copy.deepcopy(rdl_val_table)

_val_labels = rdl_val_table.df["churn"].astype(np.int64).to_numpy()
_test_labels = rdl_test_table.df["churn"].astype(np.int64).to_numpy()

print("=" * 70)
print(f"AT-RISK USER FILTERING (TARGET {_target_anom_rate:.4f} ANOMALY RATE)")
print("=" * 70)
print(f"  thresholds: {rdl_at_risk_thresholds}")
print(f"  pos downsample mode: {POS_DOWNSAMPLE_MODE}")
print(f"  Train: {len(rdl_train_table.df):,} users | {int(rdl_train_table.df['churn'].sum()):,} anomalies "
      f"({rdl_train_table.df['churn'].mean():.4f})")
print(f"  Val:   {len(rdl_val_table.df):,} users | {int(rdl_val_table.df['churn'].sum()):,} anomalies "
      f"({rdl_val_table.df['churn'].mean():.4f})")
print(f"  Test:  {len(rdl_test_table.df):,} users | {int(rdl_test_table.df['churn'].sum()):,} anomalies "
    f"({rdl_test_table.df['churn'].mean():.4f})")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — FLAT FEATURES (for RF rule mining)
# ═══════════════════════════════════════════════════════════════════════════
txn_df    = rdl_db.table_dict["transactions"].df.copy()
txn_df["customer_id"] = txn_df["customer_id"].astype(np.int64)
txn_df["t_dat"]       = pd.to_datetime(txn_df["t_dat"])
_price_col = "price" if "price" in txn_df.columns else None


def _build_customer_flat(target_df, activity_df, lookback_90=90):
    base = target_df[["customer_id", "timestamp"]].copy()
    base["customer_id"] = base["customer_id"].astype(np.int64)
    base["timestamp"]   = pd.to_datetime(base["timestamp"])
    base = base.reset_index(drop=True)

    base["recent_txns_30d"] = activity_df["recent_txns"].values
    base["trend_w"]         = activity_df["trend_w"].values
    base["avg_activity"]    = activity_df["avg_activity"].values

    _right = txn_events[["customer_id", "t_dat", "cum_txns"]]
    _j = pd.merge_asof(
        base[["customer_id", "timestamp"]].sort_values(["timestamp", "customer_id"]),
        _right.sort_values(["t_dat", "customer_id"]),
        left_on="timestamp", right_on="t_dat",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    base["total_txns"] = _j["cum_txns"].fillna(0).astype(np.int64).values

    _q2 = base[["customer_id", "timestamp"]].copy()
    _q2["window_90"] = _q2["timestamp"] - pd.Timedelta(days=lookback_90)
    _j2 = pd.merge_asof(
        _q2.sort_values(["window_90", "customer_id"]),
        _right.sort_values(["t_dat", "customer_id"]),
        left_on="window_90", right_on="t_dat",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    base["recent_txns_90d"] = np.maximum(
        0, base["total_txns"].values - _j2["cum_txns"].fillna(0).astype(np.int64).values)

    # Compute price and article features causally — use shared disk cache
    price_cols = ["mean_price", "price_std", "min_price", "max_price"]
    for c in price_cols:
        base[c] = 0.0
    base["distinct_articles"] = 0

    new_entries = False
    for ts, grp_idx in base.groupby("timestamp").groups.items():
        key = _pa_cache_key(ts)
        if key not in _pa_cache:
            _pa_cache[key] = _compute_pa_for_ts(ts)
            new_entries = True
        price_agg, art_agg = _pa_cache[key]

        if price_agg is not None:
            merged = base.loc[grp_idx, ["customer_id"]].merge(price_agg, on="customer_id", how="left")
            for c in price_cols:
                base.loc[grp_idx, c] = merged[c].fillna(0.0).values
        if art_agg is not None:
            merged2 = base.loc[grp_idx, ["customer_id"]].merge(art_agg, on="customer_id", how="left")
            base.loc[grp_idx, "distinct_articles"] = (
                merged2["distinct_articles"].fillna(0).astype(np.int64).values)

    if new_entries:
        with open(_PA_CACHE_PATH, "wb") as _f:
            _pickle.dump(_pa_cache, _f)
        print(f"Saved causal price/article cache: {len(_pa_cache)} timestamps → {_PA_CACHE_PATH}")

    return base


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
    print("Building H&M graph from scratch (this may take ~10 min)...")
    rdl_data, rdl_col_stats_dict = make_pkey_fkey_graph(
        rdl_db, col_to_stype_dict=rdl_col_to_stype,
        text_embedder_cfg=rdl_text_embedder_cfg, cache_dir=None)
    rdl_data = _mask_identity_proxy_features(_sanitize_graph(rdl_data))
    torch.save(rdl_data,          rdl_graph_cache_path)
    torch.save(rdl_col_stats_dict, rdl_stats_cache_path)
    print("H&M graph built and cached.")

gc.collect()
torch.cuda.empty_cache()
print(rdl_data)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — RF RULE MINING
# ═══════════════════════════════════════════════════════════════════════════
_flat = train_flat.copy()
_flat["churn"] = rdl_train_table.df["churn"].to_numpy(dtype=np.int64)
print(f"\nRF table: {_flat.shape}  anomaly_rate={_flat['churn'].mean():.4f}")

RF_PARAMS    = dict(n_estimators=200, max_depth=10, min_samples_leaf=5,
                    random_state=RUN_SEED, n_jobs=-1, class_weight="balanced_subsample")
RF_MAX_TRAIN = 200_000
DROP_COLS    = {"customer_id", "timestamp", "churn"}

feature_names = [c for c in _flat.columns if c not in DROP_COLS]
X_df = _flat[feature_names].apply(pd.to_numeric, errors="coerce")
X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)
X    = X_df.to_numpy(dtype=np.float32)
y_rf = _flat["churn"].to_numpy(dtype=np.int64)

rng = np.random.default_rng(RUN_SEED)
if len(X) > RF_MAX_TRAIN:
    pos_idx  = np.where(y_rf > 0)[0]; neg_idx = np.where(y_rf == 0)[0]
    pos_rate = float(y_rf.mean())
    pos_take = min(pos_idx.size, max(1, int(RF_MAX_TRAIN * pos_rate)))
    neg_take = min(neg_idx.size, RF_MAX_TRAIN - pos_take)
    sampled  = rng.permutation(np.concatenate([
        pos_idx[rng.permutation(pos_idx.size)[:pos_take]],
        neg_idx[rng.permutation(neg_idx.size)[:neg_take]],
    ]))
    X_fit, y_fit = X[sampled], y_rf[sampled]
else:
    X_fit, y_fit = X, y_rf

print(f"RF training: {len(X_fit):,} rows  pos={int(y_fit.sum()):,}  features={X.shape[1]}")
clf = RandomForestClassifier(**RF_PARAMS)
clf.fit(X_fit, y_fit)
print("RF trained.")


def _parent_map(tree_):
    parent  = np.full(tree_.node_count, -1, np.int32)
    is_left = np.zeros(tree_.node_count, bool)
    for p in range(tree_.node_count):
        l, r = int(tree_.children_left[p]), int(tree_.children_right[p])
        if l >= 0: parent[l] = p; is_left[l] = True
        if r >= 0: parent[r] = p
    return parent, is_left


def _leaf_sql(tree_est, leaf_id, fnames):
    tree_          = tree_est.tree_
    parent, is_left = _parent_map(tree_)
    conds, node    = [], int(leaf_id)
    while parent[node] != -1:
        p    = int(parent[node])
        fidx = int(tree_.feature[p])
        if fidx >= 0:
            op = "<=" if is_left[node] else ">"
            conds.append(f"{fnames[fidx]} {op} {tree_.threshold[p]:.6g}")
        node = p
    conds.reverse()
    return " AND ".join(conds)


MIN_LEAF_SAMPLES = 5
MIN_POS_PROB     = 0.50
POOL_TREES       = 100

all_rows         = []
leaf_assignments = np.stack([est.apply(X_fit) for est in clf.estimators_[:POOL_TREES]], axis=1)

for tid, tree_est in enumerate(clf.estimators_[:POOL_TREES]):
    tree_    = tree_est.tree_
    row_leaf = leaf_assignments[:, tid]
    for leaf_id in np.where(tree_.children_left == -1)[0]:
        if int(tree_.n_node_samples[leaf_id]) < MIN_LEAF_SAMPLES: continue
        vals     = tree_.value[leaf_id][0]
        pos_prob = float(vals[1] / vals.sum()) if (vals.sum() > 0 and vals.shape[0] > 1) else 0.0
        if pos_prob < MIN_POS_PROB: continue
        sql = _leaf_sql(tree_est, leaf_id, feature_names)
        if not sql: continue
        mask = row_leaf == leaf_id
        all_rows.append({
            "rule_id": f"rule_{len(all_rows)+1:04d}", "tree_id": tid,
            "pos_prob": pos_prob, "val_support": int(mask.sum()),
            "val_precision": float(y_fit[mask].mean()) if mask.any() else 0.0,
            "sql_predicate": sql,
        })

rule_paths_df = (
    pd.DataFrame(all_rows)
    .sort_values(["val_precision", "val_support"], ascending=False)
    .drop_duplicates(subset=["sql_predicate"])
    .reset_index(drop=True)
)
print(f"Extracted {len(rule_paths_df):,} unique rule candidates.")


def _sample_rules(df, n=30, seed=42):
    pool = df[df["val_support"] >= 1].head(2000).copy()
    if pool.empty: return []
    w = (np.sqrt(pool["val_precision"].clip(0)) * np.log1p(pool["val_support"].clip(0))).to_numpy() + 1e-9
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
PARSED_RULES_CSV     = hm_data("hm_parsed_rules_df.csv")
response_text = "[]"

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
        print(f"Using cached parsed LLM rules from {PARSED_RULES_CSV} ({len(_cached_rules)} rules).")
    else:
        tokenizer  = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        total_gib  = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
        llm_model  = AutoModelForCausalLM.from_pretrained(
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
Table: features_table  (customer-level activity for H&M user-churn detection)
Columns:
  recent_txns_30d      INTEGER  -- transactions in last 30 days
  prev_txns_30d        INTEGER  -- transactions in the prior 30-day window (days -60 to -30)
  trend_w              FLOAT    -- (recent_txns_30d - prev_txns_30d) / (prev_txns_30d + 1)
  avg_activity         FLOAT    -- average of recent and prior 30-day transaction counts
  total_txns           INTEGER  -- cumulative transactions before churn timestamp
  recent_txns_90d      INTEGER  -- transactions in last 90 days
  mean_price           FLOAT    -- average transaction price
  price_std            FLOAT    -- std-dev of transaction prices
  min_price            FLOAT    -- minimum transaction price
  max_price            FLOAT    -- maximum transaction price
  distinct_articles    INTEGER  -- distinct article IDs purchased
""".strip()

seed_text = json.dumps([{
    "rule_id": r.get("rule_id", f"seed_{i:03d}"),
    "sql_predicate": r.get("sql_predicate", ""),
    "val_precision": round(float(r.get("val_precision", 0)), 4),
    "val_support":   int(r.get("val_support", 0)),
} for i, r in enumerate(seed_rules[:20])], indent=2)

if RF_RULE_MODE == "llm" and not (USE_CACHED_LLM_RULES and os.path.exists(PARSED_RULES_CSV)):
    outputs = pipe(
        [
            {"role": "system", "content": (
                "You write SQL WHERE-clause predicates for an H&M customer anomaly detector. "
                "The positives are churn anomalies after at-risk users are removed. "
                "You receive RF seed rules and must generate refined or novel predicates. "
                "Return only a valid JSON array. No prose. Keep predicates simple (1-3 conditions) "
                "to maximize customer coverage."
            )},
            {"role": "user", "content": (
                f"SCHEMA:\n{SCHEMA}\n\n"
                f"SEED RULES:\n{seed_text}\n\n"
                "Generate 15 SQL WHERE predicates. Prefer broad, single-condition rules like "
                "'trend_w <= -0.5' or 'avg_activity <= 1.0' over highly specific multi-condition rules. "
                "Each object must have keys: "
                "\"candidate_id\" (string), \"sql_predicate\" (SQL), "
                "\"mutation_type\" (tighten|relax|combine|novel), \"rationale\" (one sentence). "
                "Return only the JSON array."
            )},
        ],
        max_new_tokens=int(os.environ.get("LLM_MAX_NEW_TOKENS", "4000")),
        do_sample=False, temperature=0.0, top_p=1.0,
    )
    response_text = outputs[0]["generated_text"][-1]["content"]
    print("LLM response (first 500 chars):")
    print(response_text[:500])

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
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b",  "true",  s)
    s = re.sub(r"\bFalse\b", "false", s)
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
            if esc:   esc = False
            elif c == "\\": esc = True
            elif c == '"':  in_str = False
        elif c == '"': in_str = True
        elif c == "[": depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                chunk = s[start:i+1]
                try:    return json.loads(chunk)
                except:
                    try:    return ast.literal_eval(chunk)
                    except: return []
    return []


# Broad hand-crafted rules using trend-based features — guaranteed high coverage
_broad_rules = [
    {"candidate_id": "broad_trend_neg",    "sql_predicate": "trend_w <= -0.5",                              "mutation_type": "novel", "rationale": "sharp activity decline"},
    {"candidate_id": "broad_trend_any",    "sql_predicate": "trend_w < 0.0",                               "mutation_type": "novel", "rationale": "any activity decline"},
    {"candidate_id": "broad_low_activity", "sql_predicate": "avg_activity <= 1.0",                         "mutation_type": "novel", "rationale": "very low average activity"},
    {"candidate_id": "broad_low_recent",   "sql_predicate": "recent_txns_30d <= 1",                        "mutation_type": "novel", "rationale": "almost no recent transactions"},
    {"candidate_id": "broad_trend_low_act","sql_predicate": "trend_w <= -0.3 AND avg_activity <= 3.0",     "mutation_type": "novel", "rationale": "declining + low overall activity"},
    {"candidate_id": "broad_zero_recent",  "sql_predicate": "recent_txns_30d = 0 AND total_txns > 0",      "mutation_type": "novel", "rationale": "previously active, now silent"},
]

llm_candidates = _parse_llm(response_text)
print(f"Parsed {len(llm_candidates)} rule candidates.")

if RF_RULE_MODE in {"llm", "rf_seed"} and not llm_candidates:
    print("Rule parsing failed — using seed rules as fallback.")
    llm_candidates = [
        {"candidate_id": f"seed_{i:03d}", "sql_predicate": r["sql_predicate"],
         "mutation_type": "seed_fallback", "rationale": "RF seed rule"}
        for i, r in enumerate(seed_rules[:20]) if r.get("sql_predicate", "").strip()
    ]

# Keep broad hand-crafted rules for LLM mode only.
if RF_RULE_MODE == "llm":
    llm_candidates = _broad_rules + llm_candidates

_act_eval = _activity_features_for_targets(rdl_train_table.df, lookback_days=30)
eval_df   = _build_customer_flat(rdl_train_table.df, _act_eval)
eval_df   = eval_df.drop(columns=["timestamp"], errors="ignore").copy()
eval_df["customer_id"] = eval_df["customer_id"].astype(np.int64)

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
    ref_cols = set()
    for cand in llm_candidates:
        for tok in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", str(cand.get("sql_predicate", ""))):
            if tok.upper() not in SQL_KW:
                ref_cols.add(tok)
    for c in sorted(ref_cols - set(eval_df.columns) - {"customer_id"}):
        eval_df[c] = 0

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
Path(PARSED_RULES_CSV).parent.mkdir(parents=True, exist_ok=True)
parsed_rules_df.to_csv(PARSED_RULES_CSV, index=False)
rule_cols = [c for c in result_df.columns if c != "customer_id"]
print(f"Rule columns: {rule_cols}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10 — RULE TABLE BUILDING
# ═══════════════════════════════════════════════════════════════════════════
RULE_TABLE_CACHE = hm_data("hm_customer_rule_table.pt")

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

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11 — BUILD SHARED LOADERS
# ═══════════════════════════════════════════════════════════════════════════
LOADER_BATCH_SIZE = int(os.environ.get("LOADER_BATCH_SIZE", "1024"))
ORIG_VAL_BATCH_SIZE = int(os.environ.get("ORIG_VAL_BATCH_SIZE", str(LOADER_BATCH_SIZE)))

def _make_loader(table, split):
    ti = get_node_train_table_input(table=table, task=rdl_task)
    return NeighborLoader(
        rdl_data,
        num_neighbors=[32, 32], time_attr="time",
        input_nodes=ti.nodes, input_time=ti.time, transform=ti.transform,
        batch_size=LOADER_BATCH_SIZE, temporal_strategy="uniform",
        shuffle=(split == "train"),
        num_workers=0, persistent_workers=False, pin_memory=True,
    )


loader_train = _make_loader(rdl_train_table, "train")
loader_val   = _make_loader(rdl_val_table,   "val")
loader_test  = _make_loader(rdl_test_table,  "test")
print(f"\nLoaders — train: {len(loader_train)} batches | val: {len(loader_val)} batches")

_orig_ti        = get_node_train_table_input(table=_rdl_orig_test_table, task=rdl_task)
loader_orig_test = NeighborLoader(
    rdl_data, num_neighbors=[32, 32], time_attr="time",
    input_nodes=_orig_ti.nodes, input_time=_orig_ti.time, transform=_orig_ti.transform,
    batch_size=ORIG_VAL_BATCH_SIZE, temporal_strategy="uniform",
    shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12 — SUPERVISED GNN (Baseline)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 1: SUPERVISED RDL GNN")
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
        self.gnn  = HeteroGraphSAGE(node_types=data.node_types, edge_types=data.edge_types,
                                    channels=channels, aggr=aggr, num_layers=num_layers)
        self.head = torch.nn.Linear(channels, 1)

    def forward(self, batch, entity_table):
        x_dict = self.encoder(batch.tf_dict)
        rel_t  = batch[entity_table].seed_time
        t_dict = self.temporal_encoder(rel_t, batch.time_dict, batch.batch_dict)
        x_dict = {k: (x_dict[k] + t_dict[k]) if k in t_dict else x_dict[k] for k in x_dict}
        x_dict = self.gnn(x_dict, batch.edge_index_dict,
                          num_sampled_nodes_dict=batch.num_sampled_nodes_dict,
                          num_sampled_edges_dict=batch.num_sampled_edges_dict)
        out    = self.head(x_dict[entity_table]).squeeze(-1)
        return out[:batch[entity_table].batch_size]


sup_model = SupervisedGNN(rdl_data, rdl_col_stats_dict, channels=128, num_layers=2).to(device)
sup_opt   = torch.optim.AdamW(sup_model.parameters(), lr=1e-3, weight_decay=1e-4)
print(f"SupervisedGNN | params={sum(p.numel() for p in sup_model.parameters() if p.requires_grad):,}")

SUP_EPOCHS    = int(os.environ.get("SUP_EPOCHS", "10"))
SUP_MAX_STEPS = 2000
SUP_PATIENCE  = int(os.environ.get("SUP_PATIENCE", "3"))
SUP_GRAD_CLIP = 2.0

_best_sup_auprc = -1.0; _best_sup_sd = None; _sup_patience = 0

for _epoch in range(1, SUP_EPOCHS + 1):
    sup_model.train()
    _steps, _run_loss = 0, 0.0
    for _batch in tqdm(loader_train, desc=f"  SUP epoch {_epoch:02d}", leave=False):
        _batch  = _batch.to(device)
        _logits = sup_model(_batch, rdl_entity_table)
        _lbl    = _batch[rdl_entity_table].y.float().view(-1)[:_logits.shape[0]]
        _loss   = F.binary_cross_entropy_with_logits(_logits, _lbl)
        sup_opt.zero_grad(set_to_none=True); _loss.backward()
        torch.nn.utils.clip_grad_norm_(sup_model.parameters(), SUP_GRAD_CLIP)
        sup_opt.step()
        _run_loss += float(_loss.detach()); _steps += 1
        if _steps >= SUP_MAX_STEPS: break

    sup_model.eval()
    _sup_scores = []
    with torch.no_grad():
        for _vb in loader_val:
            _sup_scores.append(torch.sigmoid(sup_model(_vb.to(device), rdl_entity_table)).detach().cpu().numpy())
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

if _best_sup_sd is not None:
    sup_model.load_state_dict(_best_sup_sd)
sup_model.eval()
_final_sup_scores = []
with torch.no_grad():
    for _vb in loader_test:
        _final_sup_scores.append(torch.sigmoid(sup_model(_vb.to(device), rdl_entity_table)).detach().cpu().numpy())
_final_sup_scores = np.concatenate(_final_sup_scores)
sup_auprc = float(average_precision_score(_test_labels, _final_sup_scores)) if np.unique(_test_labels).size > 1 else float("nan")
sup_auroc = float(roc_auc_score(_test_labels, _final_sup_scores))           if np.unique(_test_labels).size > 1 else float("nan")


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

_orig_lbl = _rdl_orig_test_table.df["churn"].astype(np.int64).to_numpy()
_final_sup_orig_scores = []
with torch.no_grad():
    for _vb in loader_orig_test:
        _final_sup_orig_scores.append(torch.sigmoid(sup_model(_vb.to(device), rdl_entity_table)).detach().cpu().numpy())
_final_sup_orig_scores = np.concatenate(_final_sup_orig_scores)
sup_orig_auprc = float(average_precision_score(_orig_lbl, _final_sup_orig_scores)) if np.unique(_orig_lbl).size > 1 else float("nan")
sup_orig_auroc = float(roc_auc_score(_orig_lbl, _final_sup_orig_scores))           if np.unique(_orig_lbl).size > 1 else float("nan")

print(f"\n  Supervised GNN val (at-risk filtered)  AUPRC={sup_auprc:.4f}  AUROC={sup_auroc:.4f}")

del sup_model, sup_opt, _best_sup_sd
gc.collect(); torch.cuda.empty_cache()
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
        self.to_latent   = nn.Linear(channels, latent_dim)
        self.decoder_gnn = HeteroGraphSAGE(
            node_types=data.node_types, edge_types=data.edge_types,
            channels=latent_dim, aggr=aggr, num_layers=decoder_num_layers)
        self.from_latent  = nn.Linear(latent_dim, channels)
        self.feat_decoder = MLP(channels, hidden_channels=channels, out_channels=channels,
                                norm=norm, num_layers=decoder_head_layers)
        self.rel_diag  = nn.ParameterDict(
            {self._ek(et): nn.Parameter(torch.ones(latent_dim)) for et in data.edge_types})
        self.rule_proj = nn.Linear(rule_dim, latent_dim, bias=False) if rule_dim > 0 else None
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.decoder_gnn.reset_parameters()
        self.feat_decoder.reset_parameters()
        nn.init.xavier_uniform_(self.to_latent.weight);   nn.init.zeros_(self.to_latent.bias)
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
        pre   = {k: v.detach() for k, v in x.items()}
        z     = self.gnn(x, batch.edge_index_dict,
                         batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict)
        z_lat = {k: self.to_latent(v) for k, v in z.items()}
        return pre, z_lat, seed_time.size(0)

    def forward(self, batch, entity_table, rule_table=None):
        pre, z_lat, n = self._encode(batch, entity_table)
        z_dec_in = {k: v for k, v in z_lat.items()}
        if rule_table is not None and self.rule_proj is not None:
            node_ids = batch[entity_table].n_id
            z_dec_in[entity_table] = (z_dec_in[entity_table]
                                      + self.rule_proj(rule_table[node_ids].to(z_dec_in[entity_table].device)))
        z_dec    = self.decoder_gnn(z_dec_in, batch.edge_index_dict,
                                    batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict)
        seed_dec = self.from_latent(z_dec[entity_table][:n])
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
        dev = z[entity_table].device
        es  = torch.zeros(n_seed, device=dev)
        ec  = torch.zeros(n_seed, device=dev)
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
    pos = scores[labels > 0.5]; neg = scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.new_zeros(()), 0
    n  = min(int(pos.numel()) * int(neg.numel()), max_pairs)
    pi = torch.randint(0, int(pos.numel()), (n,), device=scores.device)
    ni = torch.randint(0, int(neg.numel()), (n,), device=scores.device)
    return F.softplus(-(pos[pi] - neg[ni])).mean(), n


# ── Hyperparams + model build ─────────────────────────────────────────────
_hp = dict(
    channels=128, latent_dim=int(os.environ.get("LATENT_DIM", "48")),
    num_layers=2, decoder_num_layers=2, decoder_head_layers=2,
    aggr="sum", norm="batch_norm",
    lr=float(os.environ.get("LR", "1e-3")),
    weight_decay=float(os.environ.get("WEIGHT_DECAY", "1e-4")),
    epochs=int(os.environ.get("EPOCHS", "30")),
    patience=int(os.environ.get("PATIENCE", "5")),
    max_steps_train=int(os.environ.get("MAX_STEPS_TRAIN", "2000")),
    grad_clip=float(os.environ.get("GRAD_CLIP", "2.0")),
    lambda_struct=float(os.environ.get("LAMBDA_STRUCT", "0.5")),
    lambda_rank=float(os.environ.get("LAMBDA_RANK", "1.0")), rank_max_pairs=32768,
    neg_ratio=1.0,
    alpha_feat=float(os.environ.get("ALPHA_FEAT", "1.0")),
    alpha_struct=float(os.environ.get("ALPHA_STRUCT", "1.0")),
)

_warmup_epochs_env = os.environ.get("WARMUP_EPOCHS", "")
_hp["warmup_epochs"] = (int(_warmup_epochs_env) if _warmup_epochs_env.strip()
                        else min(5, max(0, _hp["epochs"] // 6)))
_hp["warmup_rank_scale"]   = float(os.environ.get("WARMUP_RANK_SCALE",   "1.0"))
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
    print("RAD rules disabled via RAD_DISABLE_RULES=1.")

seed_everything(RUN_SEED)
rad_model = RADAutoencoder(
    rdl_data, rdl_col_stats_dict,
    num_layers=_hp["num_layers"], channels=_hp["channels"],
    aggr=_hp["aggr"], norm=_hp["norm"],
    rule_dim=n_rules, latent_dim=_hp["latent_dim"],
    decoder_num_layers=_hp["decoder_num_layers"],
    decoder_head_layers=_hp["decoder_head_layers"],
).to(device)
rad_opt   = torch.optim.AdamW(rad_model.parameters(), lr=_hp["lr"], weight_decay=_hp["weight_decay"])
_rt_gpu   = rule_table.to(device) if rule_table is not None else None
print(f"RADAutoencoder | channels={_hp['channels']} | latent={_hp['latent_dim']} | "
      f"rule_dim={n_rules} | "
      f"params={sum(p.numel() for p in rad_model.parameters() if p.requires_grad):,}")


def _train_epoch(lambda_struct: float, lambda_rank: float):
    rad_model.train()
    S = dict(loss=0., feat=0., struct=0., rank=0., n=0, pairs=0, steps=0)
    feat_buf, struct_buf = [], []
    for batch in tqdm(loader_train, desc="  RAD train", leave=False):
        batch = batch.to(device)
        rad_opt.zero_grad(set_to_none=True)
        recon, target, z, n = rad_model(batch, rdl_entity_table, _rt_gpu)
        lbl    = batch[rdl_entity_table].y.float().view(-1)
        norm   = lbl < 0.5
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
    fz = (fe - ns["feat_mu"]) / max(ns["feat_sd"],   1e-8)
    sz = (se - ns["struct_mu"]) / max(ns["struct_sd"], 1e-8)
    return _hp["alpha_feat"] * fz + _hp["alpha_struct"] * sz


# ── Train RAD ─────────────────────────────────────────────────────────────
best_metric, best_state, bad = -math.inf, None, 0
norm_stats = dict(feat_mu=0., feat_sd=1., struct_mu=0., struct_sd=1.)

for epoch in range(1, _hp["epochs"] + 1):
    in_warmup         = epoch <= _hp["warmup_epochs"]
    cur_lambda_struct = _hp["lambda_struct"] * (_hp["warmup_struct_scale"] if in_warmup else 1.0)
    cur_lambda_rank   = _hp["lambda_rank"]   * (_hp["warmup_rank_scale"]   if in_warmup else 1.0)

    tr, norm_stats = _train_epoch(cur_lambda_struct, cur_lambda_rank)
    vf, vs   = _score(loader_val)
    val_pred = _combine(vf, vs, norm_stats)
    val_aupr = average_precision_score(_val_labels, val_pred)
    val_auroc_ep = roc_auc_score(_val_labels, val_pred) if np.unique(_val_labels).size > 1 else float("nan")
    val_target = val_aupr if rdl_tune_metric == "auprc" else val_auroc_ep
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

orig_vf, orig_vs = _score(loader_orig_test)
orig_val_pred    = _combine(orig_vf, orig_vs, ns)
orig_rad_auprc   = float(average_precision_score(_orig_lbl, orig_val_pred)) if np.unique(_orig_lbl).size > 1 else float("nan")
orig_rad_auroc   = float(roc_auc_score(_orig_lbl, orig_val_pred))           if np.unique(_orig_lbl).size > 1 else float("nan")

save_path = os.environ.get("SAVE_RAW_RESULTS_PATH", "weights/rad_hm_user_churn.pt")
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

print("\n" + "=" * 70)
print(f"FINAL COMPARISON ({_eval_split_name} Set — At-Risk Filtered Labels)")
print("=" * 70)
print(f"  {_eval_split_name}:  {len(rdl_test_table.df):,} users  |  {int(_test_labels.sum()):,} anomalies  "
    f"({_test_labels.mean():.4f} rate)")
print()
print(f"{'Model':<30}  {'AUPRC':>8}  {'AUROC':>8}  {'P@K':>8}  {'Lift':>8}")
print("-" * 72)
print(f"{'Supervised RDL GNN':<30}  {sup_auprc:>8.4f}  {sup_auroc:>8.4f}  {sup_pak:>8.4f}  {sup_lift:>8.4f}  (graph only)")
print(f"{'RAD (rule-augmented)':<30}  {rad_auprc:>8.4f}  {rad_auroc:>8.4f}  {rad_pak:>8.4f}  {rad_lift:>8.4f}  (graph + {n_rules} rule features)")
print("=" * 70)
print(f"FINAL COMPARISON ({_eval_split_name} Set — Original Unfiltered Labels)")
print("=" * 70)
print(f"  {_eval_split_name}:  {len(_rdl_orig_test_table.df):,} users  |  {int(_orig_lbl.sum()):,} anomalies  "
      f"({_orig_lbl.mean():.4f} rate)")
print()
print(f"{'Model':<30}  {'AUPRC':>8}  {'AUROC':>8}")
print("-" * 52)
print(f"{'Supervised RDL GNN':<30}  {sup_orig_auprc:>8.4f}  {sup_orig_auroc:>8.4f}  (graph only)")
print(f"{'RAD (rule-augmented)':<30}  {orig_rad_auprc:>8.4f}  {orig_rad_auroc:>8.4f}  (graph + {n_rules} rules)")
print(f"Sweep prevalence target: {_target_anom_rate:.4f}")
print(f"Latent dimension: {_hp['latent_dim']}")
print(f"At-risk thresholds: {rdl_at_risk_thresholds}")
print(f"Rule features used: {n_rules}")
print("=" * 70)
