#!/usr/bin/env python3
"""
Multi-seed Amazon user-churn baseline runner.

Loads the RelBench database ONCE, builds feature matrices ONCE, computes
at-risk labels ONCE, then loops over each requested seed for the
seed-dependent parts (equalization, normalisation, model training).

Usage (via env vars):
  MULTI_SEED=2,3,4,5,6,7,8,9 python run_baselines_multiseed.py
  SEED=42                     python run_baselines_multiseed.py  # single-seed

Writes:
    results/amazon/full_ablation/baselines_metrics_multiseed.csv
    columns: seed, model, auroc, auprc, p@k, lift, source
"""

from __future__ import annotations

import copy, gc, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from path_helpers import amazon_data
from eval_split import fresh_table

# ── Device setup ─────────────────────────────────────────────────────────────
_device_env = os.environ.get("BASELINES_DEVICE", "auto").strip().lower()
if _device_env in {"cpu"}:
    DEVICE = "cpu"
elif _device_env in {"cuda", "gpu", "auto", ""}:
    if torch.cuda.is_available():
        _cuda_index_env = os.environ.get("BASELINES_CUDA_INDEX", "").strip()
        if _cuda_index_env:
            DEVICE = f"cuda:{int(_cuda_index_env)}"
        elif os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            DEVICE = "cuda:0"
        else:
            DEVICE = "cuda:1"
    else:
        DEVICE = "cpu"
else:
    raise ValueError(f"Unsupported BASELINES_DEVICE={_device_env}")

print("device:", DEVICE)

# ── Env-driven constants (seed-independent) ───────────────────────────────────
CAUSAL_GUARD         = os.environ.get("CAUSAL_GUARD", "1").strip() != "0"
CAUSAL_GUARD_SAMPLES = int(os.environ.get("CAUSAL_GUARD_SAMPLES", "128"))
_ATRISK_A            = float(os.environ.get("ATRISK_TAU_A",     "0.05"))
_ATRISK_B            = float(os.environ.get("ATRISK_TAU_B",     "10.0"))
_ATRISK_ALPHA        = float(os.environ.get("ATRISK_TAU_ALPHA", "0.02"))
TARGET_ANOM_RATE     = float(os.environ.get("TARGET_ANOM_RATE", "0.015"))
TRAIN_EQ_RATE        = os.environ.get("TRAIN_EQ_RATE", "1").strip() != "0"
DRL_EPOCHS           = int(os.environ.get("DRL_EPOCHS", "10"))
SVDD_EPOCHS          = int(os.environ.get("SVDD_EPOCHS", "100"))
AE_RANK_POOL_MODE    = os.environ.get("AE_RANK_POOL_MODE", "labeled").strip().lower()
EXIT_AFTER_AE        = os.environ.get("EXIT_AFTER_AE", "0").strip() == "1"
INCLUDE_SUPERVISED   = os.environ.get("INCLUDE_SUPERVISED_RDL", "1").strip() != "0"
LOOKBACK_DAYS        = 30
PARTIAL_AUROC_MAX_FPR = float(os.environ.get("PARTIAL_AUROC_MAX_FPR", "0.1"))

# ── Load dataset & task (once) ────────────────────────────────────────────────
print("Loading RelBench dataset and task...")
dataset     = get_dataset("rel-amazon", download=True)
task        = get_task("rel-amazon", "user-churn", download=True)
db          = dataset.get_db()
train_table = task.get_table("train")
val_table   = task.get_table("val")
print(f"train: {len(train_table.df):,} | val: {len(val_table.df):,}")

# ── Feature engineering helpers (seed-independent) ───────────────────────────
review_df = db.table_dict["review"].df.copy()
review_df["customer_id"] = review_df["customer_id"].astype(np.int64)
review_df["review_time"] = pd.to_datetime(review_df["review_time"])
_rating_col = next(
    (c for c in ["rating", "star_rating", "stars"] if c in review_df.columns), None
)

_event_cols = ["customer_id", "review_time"]
if "product_id" in review_df.columns:
    _event_cols.append("product_id")
if _rating_col is not None:
    _event_cols.append(_rating_col)

review_events = (
    review_df[_event_cols]
    .dropna(subset=["customer_id", "review_time"])
    .copy()
    .sort_values(["customer_id", "review_time"])
    .reset_index(drop=True)
)
review_events["cum_reviews"] = review_events.groupby("customer_id").cumcount() + 1
review_events["first_review_time"] = review_events.groupby("customer_id")["review_time"].transform("first")

if _rating_col is not None:
    review_events["_rating_num"] = pd.to_numeric(
        review_events[_rating_col], errors="coerce"
    ).fillna(3.0)
else:
    review_events["_rating_num"] = 3.0

review_events["cum_rating_sum"]   = review_events.groupby("customer_id")["_rating_num"].cumsum()
review_events["cum_rating_sumsq"] = review_events.groupby("customer_id")["_rating_num"].transform(
    lambda s: (s * s).cumsum()
)

if "product_id" in review_events.columns:
    review_events["_is_new_product"] = (
        ~review_events.duplicated(subset=["customer_id", "product_id"])
    ).astype(np.int64)
    review_events["cum_distinct_products"] = review_events.groupby("customer_id")[
        "_is_new_product"
    ].cumsum()
else:
    review_events["cum_distinct_products"] = 0


def activity_features(target_df, lookback_days=LOOKBACK_DAYS, lookback_90=90):
    """Return per-row customer activity features (causal)."""
    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(np.int64)
    q["timestamp"]   = pd.to_datetime(q["timestamp"])
    q["_row_id"]     = np.arange(len(q), dtype=np.int64)
    q = q.sort_values(["timestamp", "customer_id", "_row_id"]).reset_index(drop=True)

    right = review_events[["customer_id", "review_time", "cum_reviews"]]

    now_join = pd.merge_asof(
        q[["customer_id", "timestamp", "_row_id"]].sort_values(["timestamp", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="timestamp", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    q["total_reviews"]    = now_join["cum_reviews"].fillna(0).astype(np.int64)
    q["last_review_time"] = now_join["review_time"]

    q["window_30"] = q["timestamp"] - pd.Timedelta(days=lookback_days)
    w30_join = pd.merge_asof(
        q[["customer_id", "window_30", "_row_id"]].sort_values(["window_30", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="window_30", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w30_cumrev = w30_join["cum_reviews"].fillna(0).astype(np.int64)
    q["recent_reviews_30d"] = np.maximum(0, q["total_reviews"].values - _w30_cumrev.values)

    q["window_60"] = q["timestamp"] - pd.Timedelta(days=2 * lookback_days)
    w60_join = pd.merge_asof(
        q[["customer_id", "window_60", "_row_id"]].sort_values(["window_60", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="window_60", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w60_cumrev = w60_join["cum_reviews"].fillna(0).astype(np.int64)
    q["prev_reviews_30d"] = np.maximum(0, _w30_cumrev.values - _w60_cumrev.values)
    q["trend_w"]    = (q["recent_reviews_30d"].values - q["prev_reviews_30d"].values) / (
        q["prev_reviews_30d"].values + 1.0
    )
    q["avg_activity"] = 0.5 * (q["recent_reviews_30d"].values + q["prev_reviews_30d"].values)

    q["window_90"] = q["timestamp"] - pd.Timedelta(days=lookback_90)
    w90_join = pd.merge_asof(
        q[["customer_id", "window_90", "_row_id"]].sort_values(["window_90", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="window_90", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    q["recent_reviews_90d"] = np.maximum(
        0, q["total_reviews"].values - w90_join["cum_reviews"].fillna(0).astype(np.int64).values
    )

    q["days_since_last_review"] = (
        (q["timestamp"] - q["last_review_time"]).dt.days.fillna(9999).astype(np.int64)
    )

    return q.sort_values("_row_id").reset_index(drop=True)[
        ["recent_reviews_30d", "recent_reviews_90d",
         "days_since_last_review", "total_reviews",
         "prev_reviews_30d", "trend_w", "avg_activity"]
    ]


def build_X(target_df):
    """Build feature matrix (causal, no leakage)."""
    act = activity_features(target_df)
    feat = pd.DataFrame(index=range(len(target_df)))
    feat["recent_reviews_30d"]     = act["recent_reviews_30d"].values
    feat["recent_reviews_90d"]     = act["recent_reviews_90d"].values
    feat["days_since_last_review"] = act["days_since_last_review"].values
    feat["total_reviews"]          = act["total_reviews"].values

    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(np.int64)
    q["timestamp"]   = pd.to_datetime(q["timestamp"])
    q["_row_id"]     = np.arange(len(q), dtype=np.int64)
    q = q.sort_values(["timestamp", "customer_id", "_row_id"]).reset_index(drop=True)

    right = review_events[[
        "customer_id", "review_time", "cum_reviews",
        "cum_rating_sum", "cum_rating_sumsq",
        "cum_distinct_products", "first_review_time",
    ]]
    state_join = pd.merge_asof(
        q[["customer_id", "timestamp", "_row_id"]].sort_values(["timestamp", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="timestamp", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    ).sort_values("_row_id").reset_index(drop=True)

    n_reviews = state_join["cum_reviews"].fillna(0).to_numpy(dtype=np.float64)
    r_sum     = state_join["cum_rating_sum"].fillna(0.0).to_numpy(dtype=np.float64)
    r_sumsq   = state_join["cum_rating_sumsq"].fillna(0.0).to_numpy(dtype=np.float64)

    mean_rating = np.where(n_reviews > 0, r_sum / np.maximum(n_reviews, 1.0), 3.0)
    var_num = r_sumsq - (r_sum * r_sum) / np.maximum(n_reviews, 1.0)
    var_den = np.maximum(n_reviews - 1.0, 1.0)
    rating_var = np.where(n_reviews > 1, np.maximum(var_num / var_den, 0.0), 0.0)
    rating_std = np.sqrt(rating_var)

    distinct_products = state_join["cum_distinct_products"].fillna(0).to_numpy(dtype=np.int64)
    span_days = (state_join["review_time"] - state_join["first_review_time"]).dt.days
    review_span_days = span_days.fillna(0).clip(lower=0).to_numpy(dtype=np.int64)

    feat["mean_rating"]        = mean_rating.astype(np.float32)
    feat["rating_std"]         = rating_std.astype(np.float32)
    feat["distinct_products"]  = distinct_products
    feat["review_span_days"]   = review_span_days
    feat["low_rating_flag"]    = (feat["mean_rating"] < 3.0).astype(np.int8)

    FEAT_COLS = [
        "recent_reviews_30d", "recent_reviews_90d", "days_since_last_review",
        "total_reviews", "mean_rating", "rating_std",
        "distinct_products", "review_span_days", "low_rating_flag",
    ]
    X = feat[FEAT_COLS].fillna(0).values.astype(np.float32)
    return X, act


# Causal guard lookup
_review_times_by_customer = {
    int(cid): grp["review_time"].to_numpy(dtype="datetime64[ns]")
    for cid, grp in review_events.groupby("customer_id", sort=False)
}


def _causal_sanity_check(
    target_df: pd.DataFrame, X: np.ndarray, sample_size: int, split_name: str, rng_seed: int = 0
) -> None:
    if len(target_df) == 0 or sample_size <= 0:
        return
    rng = np.random.default_rng(rng_seed)
    n   = min(sample_size, len(target_df))
    idxs = rng.choice(len(target_df), size=n, replace=False)
    tdf  = target_df.reset_index(drop=True)
    for i in idxs:
        cid = int(tdf.at[int(i), "customer_id"])
        ts  = np.datetime64(pd.to_datetime(tdf.at[int(i), "timestamp"]))
        arr = _review_times_by_customer.get(cid)
        if arr is None or arr.size == 0:
            total_reviews = recent_30 = recent_90 = 0
            days_since = 9999
        else:
            left       = int(np.searchsorted(arr, ts, side="left"))
            total_reviews = left
            w30        = ts - np.timedelta64(30, "D")
            w90        = ts - np.timedelta64(90, "D")
            left_30    = int(np.searchsorted(arr, w30, side="left"))
            left_90    = int(np.searchsorted(arr, w90, side="left"))
            recent_30  = max(0, left - left_30)
            recent_90  = max(0, left - left_90)
            days_since = 9999 if left == 0 else int((ts - arr[left - 1]) / np.timedelta64(1, "D"))
        xr = X[int(i)]
        if (
            int(xr[0]) != recent_30
            or int(xr[1]) != recent_90
            or int(xr[2]) != days_since
            or int(xr[3]) != total_reviews
        ):
            raise RuntimeError(
                f"Causal leakage guard FAILED for split={split_name}, row={int(i)}, cid={cid}. "
                f"Expected (30d={recent_30}, 90d={recent_90}, days={days_since}, tot={total_reviews}) "
                f"but got (30d={int(xr[0])}, 90d={int(xr[1])}, days={int(xr[2])}, tot={int(xr[3])})."
            )
    print(f"[causal-guard] split={split_name}: checked {n} rows, no leakage found")


# ── Build feature matrices once ───────────────────────────────────────────────
print("Computing train features...")
_X_tr_raw, _act_tr_raw = build_X(train_table.df)
print("Computing val features...")
_X_vl_raw, _act_vl_raw = build_X(val_table.df)
print(f"Feature matrix — train: {_X_tr_raw.shape}, val: {_X_vl_raw.shape}")

if CAUSAL_GUARD:
    _causal_sanity_check(train_table.df, _X_tr_raw, CAUSAL_GUARD_SAMPLES, "train", rng_seed=0)
    _causal_sanity_check(val_table.df,   _X_vl_raw, CAUSAL_GUARD_SAMPLES, "val",   rng_seed=1)

# ── At-risk labels (seed-independent) ────────────────────────────────────────
def atrisk_labels(target_df, activity_df, thresholds=None):
    raw   = target_df["churn"].astype(np.int64).to_numpy()
    trend = activity_df["trend_w"].to_numpy(dtype=np.float64)
    a_bar = np.maximum(activity_df["avg_activity"].to_numpy(dtype=np.float64), 0.0)
    if thresholds is None:
        thresholds = {"a": _ATRISK_A, "b": _ATRISK_B, "alpha": _ATRISK_ALPHA}
    a, b, alpha = thresholds["a"], thresholds["b"], thresholds["alpha"]
    tau     = a + (np.log(a_bar + 1.0) / np.log(b)) * (1.0 + alpha * a_bar)
    at_risk = trend <= (-tau)
    return raw * (~at_risk).astype(np.int64), ~at_risk, thresholds


_y_tr_raw, _train_keep, _atrisk_thresh = atrisk_labels(train_table.df, _act_tr_raw)
_y_vl_raw, _val_keep,   _              = atrisk_labels(val_table.df,   _act_vl_raw, thresholds=_atrisk_thresh)

# Pre-equalization filtered arrays (seed-independent)
_X_tr_filt    = _X_tr_raw[_train_keep].copy()
_act_tr_filt  = _act_tr_raw.iloc[_train_keep].reset_index(drop=True).copy()
_y_tr_filt    = _y_tr_raw[_train_keep].copy()

_X_vl_filt    = _X_vl_raw[_val_keep].copy()
_act_vl_filt  = _act_vl_raw.iloc[_val_keep].reset_index(drop=True).copy()
_y_vl_raw_filt = _y_vl_raw[_val_keep].copy()

print(f"After at-risk filter — train: {len(_X_tr_filt):,} | val: {len(_X_vl_filt):,}")

# ── Module-level model class definitions ─────────────────────────────────────
DRL_REPO = "./DRL"
if DRL_REPO not in sys.path:
    sys.path.insert(0, DRL_REPO)
from PrototypeNet.Model import PN


class DeepSVDDNet(nn.Module):
    def __init__(self, in_dim, rep_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),     nn.ReLU(),
            nn.Linear(64, rep_dim),
        )
    def forward(self, x):
        return self.net(x)


class VanillaAE(nn.Module):
    def __init__(self, in_dim, latent_dim=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, in_dim))
    def forward(self, x):
        return self.dec(self.enc(x))


# ── Supervised RDL graph cache (loaded once) ─────────────────────────────────
_GRAPH_CACHE = amazon_data("relbench_rdl_graph_cache.pt")
_STATS_CACHE = amazon_data("relbench_rdl_col_stats_cache.pt")

_rdl_data = _rdl_col_stats = None
if INCLUDE_SUPERVISED:
    if os.path.exists(_GRAPH_CACHE) and os.path.exists(_STATS_CACHE):
        _rdl_data      = torch.load(_GRAPH_CACHE,  weights_only=False)
        _rdl_col_stats = torch.load(_STATS_CACHE,  weights_only=False)
        print("Supervised RDL: graph and col-stats loaded from cache.")
    else:
        print("WARNING: Supervised RDL graph cache not found; will skip Supervised RDL.")
        INCLUDE_SUPERVISED = False

# Shared review event table for RDL activity (identical to main review_events, subset of cols)
_rev_ev = (
    db.table_dict["review"].df[["customer_id", "review_time"]]
    .dropna()
    .copy()
    .assign(
        customer_id=lambda df: df["customer_id"].astype(np.int64),
        review_time=lambda df: pd.to_datetime(df["review_time"]),
    )
    .sort_values(["customer_id", "review_time"])
    .reset_index(drop=True)
)
_rev_ev["cum_reviews"] = _rev_ev.groupby("customer_id").cumcount() + 1


def _rdl_activity(target_df, lookback=30):
    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(np.int64)
    q["timestamp"]   = pd.to_datetime(q["timestamp"])
    q["_row_id"]     = np.arange(len(q), dtype=np.int64)
    q = q.sort_values(["timestamp", "customer_id", "_row_id"]).reset_index(drop=True)
    r = _rev_ev[["customer_id", "review_time", "cum_reviews"]]

    j1 = pd.merge_asof(
        q[["customer_id", "timestamp", "_row_id"]].sort_values(["timestamp", "customer_id"]),
        r.sort_values(["review_time", "customer_id"]),
        left_on="timestamp", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    q["tot"] = j1["cum_reviews"].fillna(0).astype(np.int64)
    q["lrt"] = j1["review_time"]

    q["win"] = q["timestamp"] - pd.Timedelta(days=lookback)
    j2 = pd.merge_asof(
        q[["customer_id", "win", "_row_id"]].sort_values(["win", "customer_id"]),
        r.sort_values(["review_time", "customer_id"]),
        left_on="win", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w30 = j2["cum_reviews"].fillna(0).astype(np.int64)
    q["recent"] = np.maximum(0, q["tot"].values - _w30.values)

    q["win_prev"] = q["timestamp"] - pd.Timedelta(days=2 * lookback)
    j3 = pd.merge_asof(
        q[["customer_id", "win_prev", "_row_id"]].sort_values(["win_prev", "customer_id"]),
        r.sort_values(["review_time", "customer_id"]),
        left_on="win_prev", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w60 = j3["cum_reviews"].fillna(0).astype(np.int64)
    q["prev"] = np.maximum(0, _w30.values - _w60.values)

    q["trend_w"]      = (q["recent"].values - q["prev"].values) / (q["prev"].values + 1.0)
    q["avg_activity"] = 0.5 * (q["recent"].values + q["prev"].values)
    q["days_since"]   = (q["timestamp"] - q["lrt"]).dt.days.fillna(9999).astype(np.int64)
    return q.sort_values("_row_id").reset_index(drop=True)[
        ["recent", "prev", "trend_w", "avg_activity", "days_since"]
    ]


def _rdl_atrisk_labels(target_df, act, thresholds=None):
    raw = target_df["churn"].astype(np.int64).to_numpy()
    if thresholds is None:
        thresholds = {"a": _ATRISK_A, "b": _ATRISK_B, "alpha": _ATRISK_ALPHA}
    a, b, alpha = float(thresholds["a"]), float(thresholds["b"]), float(thresholds["alpha"])
    trend = act["trend_w"].to_numpy(dtype=np.float64)
    a_bar = np.maximum(act["avg_activity"].to_numpy(dtype=np.float64), 0.0)
    tau   = a + (np.log(a_bar + 1.0) / np.log(b)) * (1.0 + alpha * a_bar)
    at_risk = trend <= (-tau)
    keep    = ~at_risk
    return raw * keep.astype(np.int64), keep, thresholds


# ── Shared eval metric ────────────────────────────────────────────────────────
def eval_metrics(y_true, scores):
    y_true = y_true.astype(int)
    scores = scores.astype(float)
    auroc  = float(roc_auc_score(y_true, scores))
    auprc  = float(average_precision_score(y_true, scores))
    pauroc = (
        float(roc_auc_score(y_true, scores, max_fpr=PARTIAL_AUROC_MAX_FPR))
        if np.unique(y_true).size > 1 and 0.0 < PARTIAL_AUROC_MAX_FPR < 1.0
        else float("nan")
    )
    k      = max(int(y_true.sum()), 1)
    top_k  = np.argsort(-scores)[:k]
    p_at_k = float(y_true[top_k].mean())
    prev   = float(y_true.mean())
    lift   = auprc / prev if prev > 0 else float("nan")
    return {"auroc": auroc, "auprc": auprc, "pauroc": pauroc, "p@k": p_at_k, "lift": lift}


def _persist_raw_results(seed: int, raw_results: dict[str, dict[str, object]]) -> None:
    raw_dir = os.environ.get("SAVE_RAW_RESULTS_DIR", "").strip()
    if not raw_dir:
        return
    raw_path = Path(raw_dir) / f"baselines_seed{seed}.pt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw_results, raw_path)
    print(f"[seed={seed}] Wrote raw baseline scores: {raw_path}")


# ── Per-seed runner ───────────────────────────────────────────────────────────
def _run_one_seed(seed: int) -> dict[str, dict[str, float]]:
    """Run all baseline models for a single seed value.

    Uses module-level pre-equalization data (_X_tr_filt etc.) and makes
    seed-specific copies for equalization, normalisation, and training.
    Returns RESULTS dict: {model_name -> {auroc, auprc, p@k, lift}}.
    """
    print(f"\n{'='*60}\n[seed={seed}] Starting baseline training\n{'='*60}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Copies of pre-equalization data ──────────────────────────────────────
    X_train_all  = _X_tr_filt.copy()
    act_train    = _act_tr_filt.copy()
    y_train_heur = _y_tr_filt.copy()
    X_val_all    = _X_vl_filt.copy()
    act_val      = _act_vl_filt.copy()
    y_val_raw    = _y_vl_raw_filt.copy()

    # ── Equalize train positives to TARGET_ANOM_RATE ─────────────────────────
    if TRAIN_EQ_RATE:
        _tr_neg_idx  = np.where(y_train_heur == 0)[0]
        _tr_pos_idx  = np.where(y_train_heur == 1)[0]
        _n_pos_tgt   = int(round(TARGET_ANOM_RATE / (1.0 - TARGET_ANOM_RATE) * len(_tr_neg_idx)))
        _n_pos_tgt   = min(_n_pos_tgt, len(_tr_pos_idx))
        _rng_tr      = np.random.default_rng(seed)
        _keep_pos_tr = _rng_tr.choice(_tr_pos_idx, size=_n_pos_tgt, replace=False)
        _tr_mask     = np.sort(np.concatenate([_tr_neg_idx, _keep_pos_tr]))
        y_train_heur = y_train_heur[_tr_mask]
        X_train_all  = X_train_all[_tr_mask]
        act_train    = act_train.iloc[_tr_mask].reset_index(drop=True)

    # ── Equalize val positives to TARGET_ANOM_RATE ───────────────────────────
    _val_neg_idx  = np.where(y_val_raw == 0)[0]
    _val_pos_idx  = np.where(y_val_raw == 1)[0]
    _n_pos_tgt_v  = int(round(TARGET_ANOM_RATE / (1.0 - TARGET_ANOM_RATE) * len(_val_neg_idx)))
    _n_pos_tgt_v  = min(_n_pos_tgt_v, len(_val_pos_idx))
    _rng_eq       = np.random.default_rng(seed)
    _keep_pos_v   = _rng_eq.choice(_val_pos_idx, size=_n_pos_tgt_v, replace=False)
    _val_mask     = np.sort(np.concatenate([_val_neg_idx, _keep_pos_v]))
    y_val_heur    = y_val_raw[_val_mask]
    X_val_all     = X_val_all[_val_mask]
    act_val       = act_val.iloc[_val_mask].reset_index(drop=True)

    print(f"[seed={seed}] At-Risk thresholds: {_atrisk_thresh}")
    print(f"[seed={seed}] Train EQ: {TRAIN_EQ_RATE}")
    print(f"[seed={seed}] Train — total: {len(X_train_all):,} | pos: {int(y_train_heur.sum()):,} ({y_train_heur.mean():.4f})")
    print(f"[seed={seed}] Val   — total: {len(y_val_heur):,}  | pos: {int(y_val_heur.sum()):,} ({y_val_heur.mean():.4f})")

    # ── Normalise features ────────────────────────────────────────────────────
    _normal_mask = y_train_heur == 0
    X_mu    = X_train_all[_normal_mask].mean(axis=0, keepdims=True)
    X_sigma = X_train_all[_normal_mask].std(axis=0, keepdims=True)
    X_train_n = (X_train_all - X_mu) / (X_sigma + 1e-6)
    X_val_n   = (X_val_all   - X_mu) / (X_sigma + 1e-6)
    X_norm    = X_train_n[_normal_mask]
    print(f"[seed={seed}] Train normal pool: {len(X_norm):,} | Val positives: {int(y_val_heur.sum()):,}")

    RESULTS: dict[str, dict] = {}
    RAW_RESULTS: dict[str, dict[str, object]] = {}

    # ── Baseline 1: Isolation Forest ─────────────────────────────────────────
    print(f"[seed={seed}] Isolation Forest...")
    iforest    = IsolationForest(n_estimators=300, contamination="auto",
                                 random_state=seed, n_jobs=-1)
    iforest.fit(X_norm)
    if_scores  = -iforest.decision_function(X_val_n)
    RESULTS["IForest"] = eval_metrics(y_val_heur, if_scores)
    RAW_RESULTS["IForest"] = {"labels": y_val_heur.copy(), "scores": if_scores.copy()}
    print(f"  IForest: {RESULTS['IForest']}")

    # ── Baseline 2: DRL / PrototypeNet ───────────────────────────────────────
    print(f"[seed={seed}] DRL / PrototypeNet (epochs={DRL_EPOCHS})...")
    _DRL_BATCH = 8192
    _DRL_LR    = 5e-3
    _DRL_CFG   = {
        "data_dim": X_norm.shape[1], "hidden_dim": 64,
        "en_nlayers": 3, "de_nlayers": 3, "prototype_num": 5,
        "diversity": True, "plearn": False,
        "input_info": True, "input_info_ratio": 0.1,
        "cl": True, "cl_ratio": 0.06,
    }
    drl_model = PN(_DRL_CFG).to(DEVICE)
    drl_opt   = torch.optim.Adam(drl_model.parameters(), lr=_DRL_LR, weight_decay=1e-5)
    drl_sched = torch.optim.lr_scheduler.ExponentialLR(drl_opt, gamma=0.98)

    X_t      = torch.tensor(X_norm, dtype=torch.float32)
    tr_loader = DataLoader(TensorDataset(X_t), batch_size=_DRL_BATCH, shuffle=True)

    drl_model.train()
    for epoch in range(DRL_EPOCHS):
        running = 0.0; steps = 0
        for (xb,) in tr_loader:
            xb   = xb.to(DEVICE)
            loss = drl_model(xb).mean()
            if _DRL_CFG["input_info"]:
                h      = drl_model.encoder(xb)
                x_tild = drl_model.decoder(h)
                loss   = loss + _DRL_CFG["input_info_ratio"] * (
                    1 - F.cosine_similarity(x_tild, xb, dim=-1)
                ).mean()
            if _DRL_CFG["cl"]:
                h_   = drl_model.phi(xb)
                pick = max(2, int(h_.shape[0] * 0.8))
                sel  = torch.randperm(h_.shape[0], device=h_.device)[:pick]
                h_   = h_[sel]
                sim  = h_ @ h_.T
                denom = (
                    torch.sqrt((h_**2).sum(-1, keepdim=True))
                    @ torch.sqrt((h_.T**2).sum(0, keepdim=True))
                )
                sim  = sim / (denom + 1e-8)
                loss = loss + _DRL_CFG["cl_ratio"] * (
                    sim - torch.eye(h_.shape[0], device=h_.device)
                ).pow(2).mean()
            drl_opt.zero_grad(set_to_none=True)
            loss.backward(); drl_opt.step()
            running += float(loss.item()); steps += 1
        drl_sched.step()
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f"  epoch={epoch+1:03d} loss={running/max(steps,1):.6f}")

    drl_model.eval()
    X_val_t    = torch.tensor(X_val_n, dtype=torch.float32)
    val_loader = DataLoader(TensorDataset(X_val_t), batch_size=_DRL_BATCH, shuffle=False)
    drl_scores = []
    with torch.no_grad():
        for (xb,) in val_loader:
            s = drl_model(xb.to(DEVICE))
            if isinstance(s, tuple): s = s[0]
            drl_scores.append(s.view(-1).detach().cpu().numpy())
    drl_scores = np.concatenate(drl_scores)
    RESULTS["DRL"] = eval_metrics(y_val_heur, drl_scores)
    RAW_RESULTS["DRL"] = {"labels": y_val_heur.copy(), "scores": drl_scores.copy()}
    print(f"  DRL: {RESULTS['DRL']}")
    del drl_model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Baseline 3: Deep SVDD ─────────────────────────────────────────────────
    print(f"[seed={seed}] Deep SVDD (epochs={SVDD_EPOCHS})...")
    _SVDD_BATCH  = 4096
    _SVDD_LR     = 5e-3
    _SVDD_REP    = 32
    _SVDD_PAT    = 10
    _SVDD_EVAL   = 10

    svdd_model = DeepSVDDNet(X_norm.shape[1], rep_dim=_SVDD_REP).to(DEVICE)
    svdd_opt   = torch.optim.Adam(svdd_model.parameters(), lr=_SVDD_LR, weight_decay=1e-5)

    tr_loader_svdd = DataLoader(TensorDataset(X_t), batch_size=_SVDD_BATCH, shuffle=True)
    svdd_model.eval()
    with torch.no_grad():
        z0 = torch.cat([svdd_model(xb.to(DEVICE)) for (xb,) in tr_loader_svdd], dim=0)
        c  = z0.mean(dim=0)
        eps = 0.1
        c[(c.abs() < eps) & (c < 0)]  = -eps
        c[(c.abs() < eps) & (c >= 0)] =  eps

    rng_svdd = np.random.default_rng(seed)
    hold_idx  = rng_svdd.choice(len(X_norm), size=max(1, len(X_norm) // 10), replace=False)
    X_t_sub   = torch.tensor(np.delete(X_norm, hold_idx, axis=0), dtype=torch.float32)
    X_val_t   = torch.tensor(X_val_n, dtype=torch.float32)
    tr_loader_sub   = DataLoader(TensorDataset(X_t_sub), batch_size=_SVDD_BATCH, shuffle=True)
    val_loader_svdd = DataLoader(TensorDataset(X_val_t), batch_size=_SVDD_BATCH, shuffle=False)

    best_svdd_ap = -1.0; best_svdd_sd = None; patience_cnt = 0
    svdd_model.train()
    for epoch in range(SVDD_EPOCHS):
        running = 0.0; n_b = 0
        for (xb,) in tr_loader_sub:
            xb   = xb.to(DEVICE)
            z    = svdd_model(xb)
            loss = torch.sum((z - c) ** 2, dim=1).mean()
            svdd_opt.zero_grad(set_to_none=True)
            loss.backward(); svdd_opt.step()
            running += float(loss.item()); n_b += 1
        tr_loss = running / max(1, n_b)
        if tr_loss < 1e-5:
            print(f"  SVDD collapse at epoch {epoch+1}"); break
        if (epoch + 1) % _SVDD_EVAL == 0 or epoch == 0:
            svdd_model.eval()
            with torch.no_grad():
                vs = np.concatenate([
                    torch.sum((svdd_model(xb.to(DEVICE)) - c)**2, dim=1)
                    .detach().cpu().numpy()
                    for (xb,) in val_loader_svdd
                ])
            val_ap = (
                float(average_precision_score(y_val_heur, vs))
                if np.unique(y_val_heur).size > 1 else float("nan")
            )
            print(f"  epoch={epoch+1:03d} loss={tr_loss:.6f} val_auprc={val_ap:.6f}")
            if np.isfinite(val_ap) and val_ap > best_svdd_ap + 1e-8:
                best_svdd_ap = val_ap
                best_svdd_sd = copy.deepcopy(svdd_model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= _SVDD_PAT:
                    print(f"  Early stop at epoch {epoch+1}"); break
            svdd_model.train()

    if best_svdd_sd is not None:
        svdd_model.load_state_dict(best_svdd_sd)
    svdd_model.eval()
    with torch.no_grad():
        svdd_scores = np.concatenate([
            torch.sum((svdd_model(xb.to(DEVICE)) - c)**2, dim=1).detach().cpu().numpy()
            for (xb,) in val_loader_svdd
        ])
    RESULTS["DeepSVDD"] = eval_metrics(y_val_heur, svdd_scores)
    RAW_RESULTS["DeepSVDD"] = {"labels": y_val_heur.copy(), "scores": svdd_scores.copy()}
    print(f"  DeepSVDD: {RESULTS['DeepSVDD']}")
    del svdd_model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Baseline 4 & 5: Vanilla AE and AE + pairwise rank loss ───────────────
    print(f"[seed={seed}] Vanilla AE / AE+Rank...")
    _AE_EPOCHS    = 10
    _AE_BATCH     = 10000
    _AE_LR        = 1e-3
    _AE_LATENT    = 8
    _AE_PATIENCE  = 10
    _AE_EVAL_EVERY = 10
    _RANK_LAMBDA  = 1.5
    _RANK_MARGIN  = 0.1

    if AE_RANK_POOL_MODE == "none":
        X_anom_pool = np.empty((0, X_train_n.shape[1]), dtype=X_train_n.dtype)
    else:
        X_anom_pool = X_train_n[y_train_heur == 1]
    print(f"  Anomaly pool: {len(X_anom_pool):,} (mode={AE_RANK_POOL_MODE})")

    def _pairwise_rank_loss(scores, labels, margin=_RANK_MARGIN, max_pairs=32768):
        pos = scores[labels > 0.5]; neg = scores[labels <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return scores.new_zeros(()), 0
        total = int(pos.numel()) * int(neg.numel())
        if total <= max_pairs:
            diff = pos[:, None] - neg[None, :]
        else:
            n  = max_pairs
            pi = torch.randint(0, pos.numel(), (n,), device=scores.device)
            ni = torch.randint(0, neg.numel(), (n,), device=scores.device)
            diff = pos[pi] - neg[ni]
        return F.softplus(-(diff - margin)).mean(), min(total, max_pairs)

    def _train_ae(use_rank=False):
        torch.manual_seed(seed)
        model     = VanillaAE(X_norm.shape[1], _AE_LATENT).to(DEVICE)
        opt       = torch.optim.Adam(model.parameters(), lr=_AE_LR, weight_decay=1e-5)
        crit      = nn.MSELoss()
        tr_ldr    = DataLoader(
            TensorDataset(torch.tensor(X_norm, dtype=torch.float32)),
            batch_size=_AE_BATCH, shuffle=True,
        )
        vl_ldr    = DataLoader(
            TensorDataset(torch.tensor(X_val_n, dtype=torch.float32)),
            batch_size=_AE_BATCH, shuffle=False,
        )
        anom_t = (
            torch.tensor(X_anom_pool, dtype=torch.float32)
            if (use_rank and len(X_anom_pool) > 0) else None
        )
        best_ap = -1.0; best_sd = None; pat = 0

        for epoch in range(_AE_EPOCHS):
            model.train()
            running = 0.0; n_b = 0
            for (xb,) in tr_ldr:
                xb    = xb.to(DEVICE)
                recon = model(xb)
                loss  = crit(recon, xb)
                if use_rank and anom_t is not None:
                    n_anom  = min(len(xb), len(anom_t))
                    idx     = torch.randperm(len(anom_t))[:n_anom]
                    xa      = anom_t[idx].to(DEVICE)
                    recon_a = model(xa)
                    mse_n   = ((recon   - xb) ** 2).mean(dim=1)
                    mse_a   = ((recon_a - xa) ** 2).mean(dim=1)
                    all_sc  = torch.cat([mse_n, mse_a])
                    all_lb  = torch.cat([
                        torch.zeros(len(mse_n), device=DEVICE),
                        torch.ones( len(mse_a), device=DEVICE),
                    ])
                    rank_loss, _ = _pairwise_rank_loss(all_sc, all_lb)
                    loss = loss + _RANK_LAMBDA * rank_loss
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                running += float(loss.item()); n_b += 1

            if (epoch + 1) % _AE_EVAL_EVERY == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    vs = np.concatenate([
                        ((model(xb.to(DEVICE)) - xb.to(DEVICE))**2).mean(dim=1)
                        .detach().cpu().numpy()
                        for (xb,) in vl_ldr
                    ])
                val_ap = (
                    float(average_precision_score(y_val_heur, vs))
                    if np.unique(y_val_heur).size > 1 else float("nan")
                )
                label = "AE-rank" if use_rank else "AE"
                print(f"  [{label}] epoch={epoch+1:03d} loss={running/max(1,n_b):.6f} val_auprc={val_ap:.6f}")
                if np.isfinite(val_ap) and val_ap > best_ap + 1e-8:
                    best_ap = val_ap; best_sd = copy.deepcopy(model.state_dict()); pat = 0
                else:
                    pat += 1
                    if pat >= _AE_PATIENCE:
                        print(f"  Early stop at epoch {epoch+1}"); break

        if best_sd is not None:
            model.load_state_dict(best_sd)
        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                ((model(xb.to(DEVICE)) - xb.to(DEVICE))**2).mean(dim=1)
                .detach().cpu().numpy()
                for (xb,) in vl_ldr
            ])
        return scores

    ae_scores      = _train_ae(use_rank=False)
    ae_rank_scores = _train_ae(use_rank=True)

    RESULTS["VanillaAE"] = eval_metrics(y_val_heur, ae_scores)
    RESULTS["AE-Rank"]   = eval_metrics(y_val_heur, ae_rank_scores)
    RAW_RESULTS["VanillaAE"] = {"labels": y_val_heur.copy(), "scores": ae_scores.copy()}
    RAW_RESULTS["AE-Rank"] = {"labels": y_val_heur.copy(), "scores": ae_rank_scores.copy()}
    print(f"  VanillaAE: {RESULTS['VanillaAE']}")
    print(f"  AE-Rank:   {RESULTS['AE-Rank']}")

    if EXIT_AFTER_AE:
        print(f"[seed={seed}] EXIT_AFTER_AE=1 — skipping Supervised RDL")
        _persist_raw_results(seed, RAW_RESULTS)
        return RESULTS

    # ── Baseline 6: Supervised RDL GNN ───────────────────────────────────────
    if not INCLUDE_SUPERVISED or _rdl_data is None:
        print(f"[seed={seed}] Supervised RDL skipped.")
        _persist_raw_results(seed, RAW_RESULTS)
        return RESULTS

    print(f"[seed={seed}] Supervised RDL GNN...")
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.seed import seed_everything as _seed_pyg
    from relbench.modeling.graph import get_node_train_table_input
    from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder

    _seed_pyg(seed)

    # Fresh table copies per seed: get_task()/get_table() are cached, and these tables are
    # filtered in place below.
    _rdl_task        = get_task("rel-amazon", "user-churn", download=True)
    _rdl_train_table = fresh_table(_rdl_task.get_table("train"))
    _rdl_val_table   = fresh_table(_rdl_task.get_table("val"))
    _entity_table    = _rdl_task.entity_table

    # Apply at-risk filtering with the same thresholds
    _act_tr_rdl = _rdl_activity(_rdl_train_table.df)
    _act_vl_rdl = _rdl_activity(_rdl_val_table.df)
    _unexp_tr, _keep_tr, _thr = _rdl_atrisk_labels(_rdl_train_table.df, _act_tr_rdl)
    _unexp_vl, _keep_vl, _    = _rdl_atrisk_labels(_rdl_val_table.df,   _act_vl_rdl,
                                                    thresholds=_thr)

    _rdl_train_table.df = _rdl_train_table.df.loc[_keep_tr].copy()
    _rdl_train_table.df["churn"] = _unexp_tr[_keep_tr]
    _rdl_val_table.df   = _rdl_val_table.df.loc[_keep_vl].copy()
    _rdl_val_table.df["churn"]   = _unexp_vl[_keep_vl]

    # Equalize val positives to TARGET_ANOM_RATE
    _rdl_val_pos = _rdl_val_table.df[_rdl_val_table.df["churn"] == 1]
    _rdl_val_neg = _rdl_val_table.df[_rdl_val_table.df["churn"] == 0]
    if 0.0 < TARGET_ANOM_RATE < 1.0 and len(_rdl_val_pos) > 0 and len(_rdl_val_neg) > 0:
        _n_pos_t = int(round((TARGET_ANOM_RATE / (1.0 - TARGET_ANOM_RATE)) * len(_rdl_val_neg)))
        _n_pos_t = min(_n_pos_t, len(_rdl_val_pos))
        if _n_pos_t < len(_rdl_val_pos):
            _rng_rdl = np.random.default_rng(seed)
            _kp_idx  = _rng_rdl.choice(
                _rdl_val_pos.index.to_numpy(), size=_n_pos_t, replace=False
            )
            _rdl_val_table.df = _rdl_val_table.df.loc[
                np.sort(np.concatenate([_rdl_val_neg.index.to_numpy(), _kp_idx]))
            ].copy()

    _rdl_val_labels = _rdl_val_table.df["churn"].astype(np.int64).to_numpy()
    print(
        f"  RDL train: {len(_rdl_train_table.df):,} ({int(_rdl_train_table.df['churn'].sum()):,} pos) | "
        f"val: {len(_rdl_val_table.df):,} ({int(_rdl_val_table.df['churn'].sum()):,} pos)"
    )

    def _make_rdl_loader(table, shuffle):
        ti = get_node_train_table_input(table=table, task=_rdl_task)
        return NeighborLoader(
            _rdl_data, num_neighbors=[32, 32], time_attr="time",
            input_nodes=ti.nodes, input_time=ti.time, transform=ti.transform,
            batch_size=1024, temporal_strategy="uniform",
            shuffle=shuffle, num_workers=4, persistent_workers=True,
        )

    _tr_ldr_rdl = _make_rdl_loader(_rdl_train_table, shuffle=True)
    _vl_ldr_rdl = _make_rdl_loader(_rdl_val_table,   shuffle=False)
    print(f"  RDL loaders — train: {len(_tr_ldr_rdl)} batches | val: {len(_vl_ldr_rdl)} batches")

    class _SupervisedGNN(torch.nn.Module):
        def __init__(self, data, col_stats, channels=128, num_layers=2, aggr="sum"):
            super().__init__()
            self.encoder = HeteroEncoder(
                channels=channels,
                node_to_col_names_dict={n: data[n].tf.col_names_dict for n in data.node_types},
                node_to_col_stats=col_stats,
            )
            self.temporal_encoder = HeteroTemporalEncoder(
                node_types=[n for n in data.node_types if data[n].get("time") is not None],
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
            x_dict = {k: (x_dict[k] + t_dict[k]) if k in t_dict else x_dict[k]
                      for k in x_dict}
            x_dict = self.gnn(
                x_dict, batch.edge_index_dict,
                num_sampled_nodes_dict=getattr(batch, "num_sampled_nodes", None),
                num_sampled_edges_dict=getattr(batch, "num_sampled_edges", None),
            )
            out    = self.head(x_dict[entity_table]).squeeze(-1)
            return out[:batch[entity_table].batch_size]

    _sup_model = _SupervisedGNN(_rdl_data, _rdl_col_stats, channels=128, num_layers=2).to(DEVICE)
    _sup_opt   = torch.optim.AdamW(_sup_model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"  SupervisedGNN params={sum(p.numel() for p in _sup_model.parameters() if p.requires_grad):,}")

    _RDL_EPOCHS    = 10
    _RDL_MAX_STEPS = 2000
    _RDL_PATIENCE  = 3
    _RDL_GRAD_CLIP = 2.0
    _best_rdl_ap   = -1.0
    _best_rdl_sd   = None
    _rdl_pat       = 0

    for _ep in range(1, _RDL_EPOCHS + 1):
        _sup_model.train()
        _steps = 0; _running = 0.0
        for _batch in tqdm(_tr_ldr_rdl, desc=f"  RDL epoch {_ep}", leave=False):
            _batch  = _batch.to(DEVICE)
            _logits = _sup_model(_batch, _entity_table)
            _lbl    = _batch[_entity_table].y.float().view(-1)[:_logits.shape[0]]
            _loss   = F.binary_cross_entropy_with_logits(_logits, _lbl)
            _sup_opt.zero_grad(set_to_none=True)
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(_sup_model.parameters(), _RDL_GRAD_CLIP)
            _sup_opt.step()
            _running += float(_loss.detach()); _steps += 1
            if _steps >= _RDL_MAX_STEPS:
                break
        _sup_model.eval()
        _all_sc = []
        with torch.no_grad():
            for _vb in _vl_ldr_rdl:
                _vb = _vb.to(DEVICE)
                _all_sc.append(torch.sigmoid(_sup_model(_vb, _entity_table)).detach().cpu().numpy())
        _vsc  = np.concatenate(_all_sc)
        _vap  = float(average_precision_score(_rdl_val_labels, _vsc)) if np.unique(_rdl_val_labels).size > 1 else float("nan")
        _vauc = float(roc_auc_score(_rdl_val_labels, _vsc)) if np.unique(_rdl_val_labels).size > 1 else float("nan")
        print(f"  RDL ep {_ep:02d} | steps={_steps} | loss={_running/_steps:.4f} | val_auprc={_vap:.4f} | auroc={_vauc:.4f}")
        if np.isfinite(_vap) and _vap > _best_rdl_ap + 1e-8:
            _best_rdl_ap = _vap
            _best_rdl_sd = {k: v.detach().cpu().clone() for k, v in _sup_model.state_dict().items()}
            _rdl_pat = 0
        else:
            _rdl_pat += 1
            if _rdl_pat >= _RDL_PATIENCE:
                print(f"  RDL early stop at epoch {_ep}"); break

    if _best_rdl_sd is not None:
        _sup_model.load_state_dict(_best_rdl_sd)
    _sup_model.eval()
    _final_sc = []
    with torch.no_grad():
        for _vb in _vl_ldr_rdl:
            _vb = _vb.to(DEVICE)
            _final_sc.append(torch.sigmoid(_sup_model(_vb, _entity_table)).detach().cpu().numpy())
    _final_sc = np.concatenate(_final_sc)

    RESULTS["Supervised RDL"] = eval_metrics(_rdl_val_labels, _final_sc)
    RAW_RESULTS["Supervised RDL"] = {"labels": _rdl_val_labels.copy(), "scores": _final_sc.copy()}
    print(f"  Supervised RDL: {RESULTS['Supervised RDL']}")
    del _sup_model, _tr_ldr_rdl, _vl_ldr_rdl; gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    _persist_raw_results(seed, RAW_RESULTS)

    return RESULTS


# ── Persistence helpers ───────────────────────────────────────────────────────
_OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "amazon" / "full_ablation"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_MS_CSV  = _OUT_DIR / "baselines_metrics_multiseed.csv"
_SS_CSV  = _OUT_DIR / "baselines_metrics.csv"


def _write_multiseed_csv(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    _MS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_MS_CSV, index=False)
    print(f"[multi-seed] Wrote {_MS_CSV} ({len(df)} rows)")


def _write_singleseed_csv(results: dict[str, dict]) -> None:
    rows = []
    for model, met in results.items():
        if isinstance(met, dict):
            rows.append({
                "model": model,
                "auroc": float(met.get("auroc", float("nan"))),
                "auprc": float(met.get("auprc", float("nan"))),
                "pauroc": float(met.get("pauroc", float("nan"))),
                "p@k":   float(met.get("p@k",   float("nan"))),
                "lift":  float(met.get("lift",  float("nan"))),
                "source": "baselines_multiseed_script",
            })
    pd.DataFrame(rows).to_csv(_SS_CSV, index=False)
    print(f"[single-seed] Wrote {_SS_CSV}")


# ── Main dispatch ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _MULTI_SEED_ENV = os.environ.get("MULTI_SEED", "").strip()
    _DEFAULT_SEED   = int(os.environ.get("SEED", os.environ.get("RANDOM_STATE", "42")))

    if _MULTI_SEED_ENV:
        _seeds = [int(s.strip()) for s in _MULTI_SEED_ENV.split(",") if s.strip()]
    else:
        _seeds = [_DEFAULT_SEED]

    print(f"Seeds to run: {_seeds}")

    _all_rows: list[dict] = []

    for _seed in _seeds:
        _res = _run_one_seed(_seed)
        for _model, _met in _res.items():
            _all_rows.append({"seed": _seed, "model": _model, **_met})

        # Also write single-seed CSV after each seed so single-seed readers stay current
        _write_singleseed_csv(_res)

    _write_multiseed_csv(_all_rows)

    print("\n" + "=" * 70)
    print(f"{'Model':<22} {'Seed':>6} {'AUROC':>8} {'AUPRC':>8} {'P@K':>8} {'Lift':>8}")
    print("-" * 70)
    for r in _all_rows:
        print(
            f"{r['model']:<22} {r['seed']:>6} {r['auroc']:8.4f} {r['auprc']:8.4f} "
            f"{r['p@k']:8.4f} {r['lift']:8.2f}"
        )
    print("=" * 70)
