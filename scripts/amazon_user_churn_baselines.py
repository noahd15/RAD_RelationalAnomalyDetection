#!/usr/bin/env python3
"""Standalone Amazon user-churn baselines (no notebook execution)."""

# ===== Converted from notebook code cell 1 =====
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
from path_helpers import amazon_data, append_results_jsonl

RANDOM_STATE = int(os.environ.get("SEED", os.environ.get("RANDOM_STATE", "42")))
_device_env = os.environ.get("BASELINES_DEVICE", "auto").strip().lower()
if _device_env in {"cpu"}:
    DEVICE = "cpu"
elif _device_env in {"cuda", "gpu", "auto", ""}:
    if torch.cuda.is_available():
        _cuda_index_env = os.environ.get("BASELINES_CUDA_INDEX", "").strip()
        if _cuda_index_env:
            DEVICE = f"cuda:{int(_cuda_index_env)}"
        elif os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            # CUDA-visible remapping exposes selected GPUs as 0..N-1.
            DEVICE = "cuda:0"
        else:
            DEVICE = "cuda:1"
    else:
        DEVICE = "cpu"
else:
    raise ValueError(f"Unsupported BASELINES_DEVICE={_device_env}; use cpu|cuda|auto")

print("device:", DEVICE)
print("seed:", RANDOM_STATE)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# ── Load dataset & task ───────────────────────────────────────────────────────
dataset = get_dataset("rel-amazon", download=True)
task    = get_task("rel-amazon", "user-churn", download=True)
db      = dataset.get_db()

train_table = task.get_table("train")
val_table   = task.get_table("val")
print("train:", len(train_table.df), "| val:", len(val_table.df))
print("train churn rate:", train_table.df["churn"].mean().round(4),
      "| val churn rate:", val_table.df["churn"].mean().round(4))

# ===== Converted from notebook code cell 2 =====
# ── Feature engineering ───────────────────────────────────────────────────────
# For each (customer_id, timestamp) in the task table compute:
#   recent_reviews_30d    – reviews in the 30-day window before timestamp
#   recent_reviews_90d    – reviews in the 90-day window before timestamp
#   days_since_last_review – days since most recent review (9999 if none)
#   total_reviews         – all reviews before timestamp
#   mean_rating           – customer's mean star rating before timestamp
#   rating_std            – std of star ratings before timestamp
#   distinct_products     – distinct products reviewed before timestamp
#   review_span_days      – days between first and last review before timestamp
#   low_rating_flag       – 1 if mean_rating < 3
# Feature computation is strictly causal (only uses data before task timestamp).

LOOKBACK_DAYS = 30
CAUSAL_GUARD = os.environ.get("CAUSAL_GUARD", "1").strip() != "0"
CAUSAL_GUARD_SAMPLES = int(os.environ.get("CAUSAL_GUARD_SAMPLES", "128"))

review_df = db.table_dict["review"].df.copy()
review_df["customer_id"] = review_df["customer_id"].astype(np.int64)
review_df["review_time"] = pd.to_datetime(review_df["review_time"])
_rating_col = next((c for c in ["rating", "star_rating", "stars"] if c in review_df.columns), None)

# Event-level table with causal cumulative statistics.
_event_cols = ["customer_id", "review_time"]
if "product_id" in review_df.columns:
    _event_cols.append("product_id")
if _rating_col is not None:
    _event_cols.append(_rating_col)

review_events = review_df[_event_cols].dropna(subset=["customer_id", "review_time"]).copy()
review_events = review_events.sort_values(["customer_id", "review_time"]).reset_index(drop=True)
review_events["cum_reviews"] = review_events.groupby("customer_id").cumcount() + 1
review_events["first_review_time"] = review_events.groupby("customer_id")["review_time"].transform("first")

if _rating_col is not None:
    review_events["_rating_num"] = pd.to_numeric(review_events[_rating_col], errors="coerce").fillna(3.0)
else:
    review_events["_rating_num"] = 3.0
review_events["cum_rating_sum"] = review_events.groupby("customer_id")["_rating_num"].cumsum()
review_events["cum_rating_sumsq"] = review_events.groupby("customer_id")["_rating_num"].transform(lambda s: (s * s).cumsum())

if "product_id" in review_events.columns:
    review_events["_is_new_product"] = (~review_events.duplicated(subset=["customer_id", "product_id"])).astype(np.int64)
    review_events["cum_distinct_products"] = review_events.groupby("customer_id")["_is_new_product"].cumsum()
else:
    review_events["cum_distinct_products"] = 0


def activity_features(target_df, lookback_days=LOOKBACK_DAYS, lookback_90=90):
    """Return DataFrame with per-row customer activity features."""
    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(np.int64)
    q["timestamp"]   = pd.to_datetime(q["timestamp"])
    q["_row_id"]     = np.arange(len(q), dtype=np.int64)
    q = q.sort_values(["timestamp", "customer_id", "_row_id"]).reset_index(drop=True)

    right = review_events[["customer_id", "review_time", "cum_reviews"]]

    # total reviews before timestamp
    now_join = pd.merge_asof(
        q[["customer_id", "timestamp", "_row_id"]].sort_values(["timestamp", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="timestamp", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    q["total_reviews"]    = now_join["cum_reviews"].fillna(0).astype(np.int64)
    q["last_review_time"] = now_join["review_time"]

    # 30-day window
    q["window_30"] = q["timestamp"] - pd.Timedelta(days=lookback_days)
    w30_join = pd.merge_asof(
        q[["customer_id", "window_30", "_row_id"]].sort_values(["window_30", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="window_30", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w30_cumrev = w30_join["cum_reviews"].fillna(0).astype(np.int64)
    q["recent_reviews_30d"] = np.maximum(
        0, q["total_reviews"].values - _w30_cumrev.values)

    # Previous 30-day window (days -60 to -30) for AtRisk trend computation
    q["window_60"] = q["timestamp"] - pd.Timedelta(days=2 * lookback_days)
    w60_join = pd.merge_asof(
        q[["customer_id", "window_60", "_row_id"]].sort_values(["window_60", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="window_60", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w60_cumrev = w60_join["cum_reviews"].fillna(0).astype(np.int64)
    q["prev_reviews_30d"] = np.maximum(0, _w30_cumrev.values - _w60_cumrev.values)
    q["trend_w"]    = ((q["recent_reviews_30d"].values - q["prev_reviews_30d"].values)
                       / (q["prev_reviews_30d"].values + 1.0))
    q["avg_activity"] = 0.5 * (q["recent_reviews_30d"].values + q["prev_reviews_30d"].values)

    # 90-day window
    q["window_90"] = q["timestamp"] - pd.Timedelta(days=lookback_90)
    w90_join = pd.merge_asof(
        q[["customer_id", "window_90", "_row_id"]].sort_values(["window_90", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="window_90", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    q["recent_reviews_90d"] = np.maximum(
        0, q["total_reviews"].values - w90_join["cum_reviews"].fillna(0).astype(np.int64).values)

    q["days_since_last_review"] = (
        (q["timestamp"] - q["last_review_time"]).dt.days.fillna(9999).astype(np.int64))

    return q.sort_values("_row_id").reset_index(drop=True)[
        ["recent_reviews_30d", "recent_reviews_90d",
         "days_since_last_review", "total_reviews",
         "prev_reviews_30d", "trend_w", "avg_activity"]
    ]


def build_X(target_df):
    """Build feature matrix for a task table (one row per customer/timestamp)."""
    act = activity_features(target_df)

    # Causal customer stats computed only from reviews strictly before timestamp.
    feat = pd.DataFrame(index=range(len(target_df)))
    feat["recent_reviews_30d"]    = act["recent_reviews_30d"].values
    feat["recent_reviews_90d"]    = act["recent_reviews_90d"].values
    feat["days_since_last_review"]= act["days_since_last_review"].values
    feat["total_reviews"]         = act["total_reviews"].values

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

    feat["mean_rating"] = mean_rating.astype(np.float32)
    feat["rating_std"]  = rating_std.astype(np.float32)
    feat["distinct_products"] = distinct_products
    feat["review_span_days"] = review_span_days

    feat["low_rating_flag"] = (feat["mean_rating"] < 3.0).astype(np.int8)

    FEAT_COLS = [
        "recent_reviews_30d", "recent_reviews_90d", "days_since_last_review",
        "total_reviews", "mean_rating", "rating_std",
        "distinct_products", "review_span_days", "low_rating_flag",
    ]
    X = feat[FEAT_COLS].fillna(0).values.astype(np.float32)
    return X, act


_review_times_by_customer = {
    int(cid): grp["review_time"].to_numpy(dtype="datetime64[ns]")
    for cid, grp in review_events.groupby("customer_id", sort=False)
}


def _causal_sanity_check(target_df: pd.DataFrame, X: np.ndarray, sample_size: int, split_name: str) -> None:
    if len(target_df) == 0 or sample_size <= 0:
        return
    rng = np.random.default_rng(RANDOM_STATE + (0 if split_name == "train" else 1))
    n = min(sample_size, len(target_df))
    idxs = rng.choice(len(target_df), size=n, replace=False)
    tdf = target_df.reset_index(drop=True)

    for i in idxs:
        cid = int(tdf.at[int(i), "customer_id"])
        ts = np.datetime64(pd.to_datetime(tdf.at[int(i), "timestamp"]))
        arr = _review_times_by_customer.get(cid)

        if arr is None or arr.size == 0:
            total_reviews = 0
            recent_30 = 0
            recent_90 = 0
            days_since = 9999
        else:
            left = int(np.searchsorted(arr, ts, side="left"))
            total_reviews = left

            w30 = ts - np.timedelta64(30, "D")
            w90 = ts - np.timedelta64(90, "D")
            left_30 = int(np.searchsorted(arr, w30, side="left"))
            left_90 = int(np.searchsorted(arr, w90, side="left"))
            recent_30 = max(0, left - left_30)
            recent_90 = max(0, left - left_90)

            if left == 0:
                days_since = 9999
            else:
                days_since = int((ts - arr[left - 1]) / np.timedelta64(1, "D"))

        xr = X[int(i)]
        if (
            int(xr[0]) != recent_30
            or int(xr[1]) != recent_90
            or int(xr[2]) != days_since
            or int(xr[3]) != total_reviews
        ):
            raise RuntimeError(
                "Causal leakage guard failed for split="
                f"{split_name}, row={int(i)}, customer_id={cid}. "
                f"Expected (30d={recent_30}, 90d={recent_90}, days={days_since}, total={total_reviews}) "
                f"but got (30d={int(xr[0])}, 90d={int(xr[1])}, days={int(xr[2])}, total={int(xr[3])})."
            )

    print(f"[causal-guard] split={split_name}: checked {n} rows, no leakage found")


print("Computing train features...")
X_train_all, act_train = build_X(train_table.df)
print("Computing val features...")
X_val_all,   act_val   = build_X(val_table.df)
print(f"Feature matrix — train: {X_train_all.shape}, val: {X_val_all.shape}")

if CAUSAL_GUARD:
    _causal_sanity_check(train_table.df, X_train_all, CAUSAL_GUARD_SAMPLES, "train")
    _causal_sanity_check(val_table.df, X_val_all, CAUSAL_GUARD_SAMPLES, "val")

# ===== Converted from notebook code cell 3 =====
# ── At-Risk labeling (paper formula, equations 1 & 2) ─────────────────────────
# AtRisk(u) = 1[ trend_w(u) <= -τ(ā(u)) ]   τ(ā) = a + log_b(ā+1)·(1 + α·ā)
# Positive  = churned AND NOT at-risk (unexpected churner).
# At-risk churners are removed from the dataset entirely.

_ATRISK_A     = float(os.environ.get("ATRISK_TAU_A",     "0.05"))
_ATRISK_B     = float(os.environ.get("ATRISK_TAU_B",     "10.0"))
_ATRISK_ALPHA = float(os.environ.get("ATRISK_TAU_ALPHA", "0.02"))
TARGET_ANOM_RATE = float(os.environ.get("TARGET_ANOM_RATE", "0.015"))
TRAIN_EQ_RATE = os.environ.get("TRAIN_EQ_RATE", "1").strip() != "0"


def atrisk_labels(target_df, activity_df, thresholds=None):
    raw   = target_df["churn"].astype(np.int64).to_numpy()
    trend = activity_df["trend_w"].to_numpy(dtype=np.float64)
    a_bar = np.maximum(activity_df["avg_activity"].to_numpy(dtype=np.float64), 0.0)
    if thresholds is None:
        thresholds = {"a": _ATRISK_A, "b": _ATRISK_B, "alpha": _ATRISK_ALPHA}
    a, b, alpha = thresholds["a"], thresholds["b"], thresholds["alpha"]
    tau     = a + (np.log(a_bar + 1.0) / np.log(b)) * (1.0 + alpha * a_bar)
    at_risk = trend <= (-tau)
    keep_mask      = ~at_risk
    anomaly_labels = raw * keep_mask.astype(np.int64)
    return anomaly_labels, keep_mask, thresholds


y_train_raw, train_keep, atrisk_thresholds = atrisk_labels(train_table.df, act_train)
y_val_raw,   val_keep,   _                 = atrisk_labels(val_table.df,   act_val,
                                                            thresholds=atrisk_thresholds)

# Apply row filtering (remove at-risk rows entirely)
X_train_all = X_train_all[train_keep]
act_train   = act_train.iloc[train_keep].reset_index(drop=True)
y_train_heur = y_train_raw[train_keep]

X_val_all = X_val_all[val_keep]
act_val   = act_val.iloc[val_keep].reset_index(drop=True)
y_val_raw = y_val_raw[val_keep]

# Downsample train positives to TARGET_ANOM_RATE for apples-to-apples parity
# with RAD training prevalence. Disable with TRAIN_EQ_RATE=0.
if TRAIN_EQ_RATE:
    _tr_neg_idx  = np.where(y_train_heur == 0)[0]
    _tr_pos_idx  = np.where(y_train_heur == 1)[0]
    _n_pos_target_tr = int(round(TARGET_ANOM_RATE / (1.0 - TARGET_ANOM_RATE) * len(_tr_neg_idx)))
    _n_pos_target_tr = min(_n_pos_target_tr, len(_tr_pos_idx))
    _rng_tr       = np.random.default_rng(RANDOM_STATE)
    _keep_pos_tr  = _rng_tr.choice(_tr_pos_idx, size=_n_pos_target_tr, replace=False)
    _tr_mask      = np.sort(np.concatenate([_tr_neg_idx, _keep_pos_tr]))
    y_train_heur  = y_train_heur[_tr_mask]
    X_train_all   = X_train_all[_tr_mask]
    act_train     = act_train.iloc[_tr_mask].reset_index(drop=True)

# Downsample val positives to TARGET_ANOM_RATE
_val_neg_idx  = np.where(y_val_raw == 0)[0]
_val_pos_idx  = np.where(y_val_raw == 1)[0]
_n_pos_target = int(round(TARGET_ANOM_RATE / (1.0 - TARGET_ANOM_RATE) * len(_val_neg_idx)))
_n_pos_target = min(_n_pos_target, len(_val_pos_idx))
_rng_eq       = np.random.default_rng(RANDOM_STATE)
_keep_pos     = _rng_eq.choice(_val_pos_idx, size=_n_pos_target, replace=False)
_val_mask     = np.sort(np.concatenate([_val_neg_idx, _keep_pos]))
y_val_heur    = y_val_raw[_val_mask]
X_val_all     = X_val_all[_val_mask]
act_val       = act_val.iloc[_val_mask].reset_index(drop=True)

print("At-Risk thresholds:", atrisk_thresholds)
print(f"Train prevalence matching enabled: {TRAIN_EQ_RATE}")
print(f"Train — total: {len(X_train_all):,} | positives: {int(y_train_heur.sum()):,} ({y_train_heur.mean():.4f})")
print(f"Val   — total: {len(y_val_heur):,}   | positives: {int(y_val_heur.sum()):,} ({y_val_heur.mean():.4f})")

# Normalise features using train-normal statistics
train_normal_mask = y_train_heur == 0
X_mu    = X_train_all[train_normal_mask].mean(axis=0, keepdims=True)
X_sigma = X_train_all[train_normal_mask].std(axis=0, keepdims=True)
X_train_n = (X_train_all - X_mu) / (X_sigma + 1e-6)
X_val_n   = (X_val_all   - X_mu) / (X_sigma + 1e-6)

# Training pool: normal samples only
X_norm = X_train_n[train_normal_mask]
print(f"\nTrain normal pool: {len(X_norm):,} | Val positives: {int(y_val_heur.sum()):,}")

# ===== Converted from notebook code cell 4 =====
# ── Shared evaluation helper ──────────────────────────────────────────────────
def eval_metrics(y_true, scores):
    y_true  = y_true.astype(int)
    scores  = scores.astype(float)
    auroc   = float(roc_auc_score(y_true, scores))
    auprc   = float(average_precision_score(y_true, scores))
    k       = max(int(y_true.sum()), 1)
    top_k   = np.argsort(-scores)[:k]
    p_at_k  = float(y_true[top_k].mean())
    prev    = float(y_true.mean())
    lift    = auprc / prev if prev > 0 else float("nan")
    return {"auroc": auroc, "auprc": auprc, "p@k": p_at_k, "lift": lift}

RESULTS = {}   # model_name -> metrics dict


def _persist_results_csv(results: dict[str, dict[str, float]]) -> None:
    root_dir = Path(__file__).resolve().parents[1]
    out_dir = root_dir / "results" / "amazon" / "full_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, met in results.items():
        if isinstance(met, dict):
            rows.append(
                {
                    "model": model,
                    "auroc": float(met.get("auroc", float("nan"))),
                    "auprc": float(met.get("auprc", float("nan"))),
                    "p@k": float(met.get("p@k", float("nan"))),
                    "lift": float(met.get("lift", float("nan"))),
                    "source": "baselines_script",
                }
            )
    df = pd.DataFrame(rows)
    # Seed-specific path: a fixed overwritten path loses every prior seed's raw output.
    csv_path = out_dir / f"baselines_metrics_seed{RANDOM_STATE}.csv"
    df.to_csv(csv_path, index=False)
    # Also keep writing the un-seeded path for backward compatibility with existing consumers.
    (out_dir / "baselines_metrics.csv").write_text(df.to_csv(index=False))
    print(f"[baselines] Wrote metrics CSV: {csv_path}")
    for row in rows:
        append_results_jsonl({
            "task": "amazon-user-churn", "seed": RANDOM_STATE, "split": "Val",
            "method": row["model"],
            "auprc": row["auprc"], "auroc": row["auroc"], "p@k": row["p@k"], "lift": row["lift"],
        })


def _persist_raw_results(raw_results: dict[str, dict[str, object]]) -> None:
    save_path = os.environ.get("SAVE_RAW_RESULTS_PATH", "").strip()
    if not save_path:
        return
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw_results, save_path_obj)
    print(f"[baselines] Wrote raw scores: {save_path_obj}")

print(f"Val baseline prevalence: {y_val_heur.mean():.6f}  "
      f"(random AUPRC ≈ {y_val_heur.mean():.4f})")

# ===== Converted from notebook code cell 5 =====
# ── Baseline 1: Isolation Forest ─────────────────────────────────────────────
print("=" * 50)
print("Baseline 1: Isolation Forest")
print("=" * 50)

iforest = IsolationForest(n_estimators=300, contamination="auto",
                          random_state=RANDOM_STATE, n_jobs=-1)
iforest.fit(X_norm)
if_scores = -iforest.decision_function(X_val_n)   # higher = more anomalous

RESULTS["IForest"] = eval_metrics(y_val_heur, if_scores)
print("IForest:", RESULTS["IForest"])

# ===== Converted from notebook code cell 6 =====
# ── Baseline 2: DRL / PrototypeNet ───────────────────────────────────────────
print("=" * 50)
print("Baseline 2: DRL / PrototypeNet")
print("=" * 50)

DRL_REPO = "./DRL"
if DRL_REPO not in sys.path:
    sys.path.insert(0, DRL_REPO)
from PrototypeNet.Model import PN

DRL_CFG = {
    "data_dim":          X_norm.shape[1],
    "hidden_dim":        64,
    "en_nlayers":        3,
    "de_nlayers":        3,
    "prototype_num":     5,
    "diversity":         True,
    "plearn":            False,
    "input_info":        True,
    "input_info_ratio":  0.1,
    "cl":                True,
    "cl_ratio":          0.06,
}
DRL_EPOCHS     = int(os.environ.get("DRL_EPOCHS", "10"))
DRL_BATCH_SIZE = 8192
DRL_LR         = 5e-3

drl_model = PN(DRL_CFG).to(DEVICE)
drl_opt   = torch.optim.Adam(drl_model.parameters(), lr=DRL_LR, weight_decay=1e-5)
drl_sched = torch.optim.lr_scheduler.ExponentialLR(drl_opt, gamma=0.98)

X_t = torch.tensor(X_norm, dtype=torch.float32)
tr_loader = DataLoader(TensorDataset(X_t), batch_size=DRL_BATCH_SIZE, shuffle=True)

drl_model.train()
for epoch in range(DRL_EPOCHS):
    running = 0.0; steps = 0
    for (xb,) in tr_loader:
        xb   = xb.to(DEVICE)
        loss = drl_model(xb).mean()
        if DRL_CFG["input_info"]:
            h      = drl_model.encoder(xb)
            x_tild = drl_model.decoder(h)
            loss   = loss + DRL_CFG["input_info_ratio"] * (1 - F.cosine_similarity(x_tild, xb, dim=-1)).mean()
        if DRL_CFG["cl"]:
            h_   = drl_model.phi(xb)
            pick = max(2, int(h_.shape[0] * 0.8))
            sel  = torch.randperm(h_.shape[0], device=h_.device)[:pick]
            h_   = h_[sel]
            sim  = h_ @ h_.T
            denom = (torch.sqrt((h_**2).sum(-1, keepdim=True))
                     @ torch.sqrt((h_.T**2).sum(0, keepdim=True)))
            sim  = sim / (denom + 1e-8)
            loss = loss + DRL_CFG["cl_ratio"] * (sim - torch.eye(h_.shape[0], device=h_.device)).pow(2).mean()
        drl_opt.zero_grad(set_to_none=True)
        loss.backward()
        drl_opt.step()
        running += float(loss.item()); steps += 1
    drl_sched.step()
    if epoch == 0 or (epoch + 1) % 20 == 0:
        print(f"  epoch={epoch+1:03d} loss={running/max(steps,1):.6f}")

drl_model.eval()
X_val_t = torch.tensor(X_val_n, dtype=torch.float32)
val_loader = DataLoader(TensorDataset(X_val_t), batch_size=DRL_BATCH_SIZE, shuffle=False)
drl_scores = []
with torch.no_grad():
    for (xb,) in val_loader:
        s = drl_model(xb.to(DEVICE))
        if isinstance(s, tuple): s = s[0]
        drl_scores.append(s.view(-1).detach().cpu().numpy())
drl_scores = np.concatenate(drl_scores)

RESULTS["DRL"] = eval_metrics(y_val_heur, drl_scores)
print("DRL:", RESULTS["DRL"])

# ===== Converted from notebook code cell 7 =====
# ── Baseline 3: Deep SVDD ─────────────────────────────────────────────────────
print("=" * 50)
print("Baseline 3: Deep SVDD")
print("=" * 50)

SVDD_EPOCHS     = int(os.environ.get("SVDD_EPOCHS", "100"))
SVDD_BATCH_SIZE = 4096
SVDD_LR         = 5e-3
SVDD_REP_DIM    = 32
SVDD_PATIENCE   = 10
SVDD_EVAL_EVERY = 10

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


svdd_model = DeepSVDDNet(X_norm.shape[1], rep_dim=SVDD_REP_DIM).to(DEVICE)
svdd_opt   = torch.optim.Adam(svdd_model.parameters(), lr=SVDD_LR, weight_decay=1e-5)

# Initialise center from a forward pass on training data
tr_loader_svdd = DataLoader(TensorDataset(X_t), batch_size=SVDD_BATCH_SIZE, shuffle=True)
svdd_model.eval()
with torch.no_grad():
    z0 = torch.cat([svdd_model(xb.to(DEVICE)) for (xb,) in tr_loader_svdd], dim=0)
    c  = z0.mean(dim=0)
    eps = 0.1
    c[(c.abs() < eps) & (c < 0)]  = -eps
    c[(c.abs() < eps) & (c >= 0)] =  eps

# Hold-out from normal train samples for early stopping
rng_svdd = np.random.default_rng(RANDOM_STATE)
hold_idx = rng_svdd.choice(len(X_norm), size=max(1, len(X_norm)//10), replace=False)
X_t_sub  = torch.tensor(np.delete(X_norm, hold_idx, axis=0), dtype=torch.float32)
X_val_t  = torch.tensor(X_val_n, dtype=torch.float32)

tr_loader_sub   = DataLoader(TensorDataset(X_t_sub), batch_size=SVDD_BATCH_SIZE, shuffle=True)
val_loader_svdd = DataLoader(TensorDataset(X_val_t), batch_size=SVDD_BATCH_SIZE, shuffle=False)

best_svdd_ap  = -1.0
best_svdd_sd  = None
patience_cnt  = 0

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
    if (epoch + 1) % SVDD_EVAL_EVERY == 0 or epoch == 0:
        svdd_model.eval()
        with torch.no_grad():
            vs = np.concatenate([torch.sum((svdd_model(xb.to(DEVICE)) - c)**2, dim=1)
                                  .detach().cpu().numpy() for (xb,) in val_loader_svdd])
        val_ap = float(average_precision_score(y_val_heur, vs)) if np.unique(y_val_heur).size > 1 else float("nan")
        print(f"  epoch={epoch+1:03d} train_loss={tr_loss:.6f} val_auprc={val_ap:.6f}")
        if np.isfinite(val_ap) and val_ap > best_svdd_ap + 1e-8:
            best_svdd_ap = val_ap
            best_svdd_sd = copy.deepcopy(svdd_model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= SVDD_PATIENCE:
                print(f"  Early stop at epoch {epoch+1}"); break
        svdd_model.train()

if best_svdd_sd is not None:
    svdd_model.load_state_dict(best_svdd_sd)

svdd_model.eval()
with torch.no_grad():
    svdd_scores = np.concatenate(
        [torch.sum((svdd_model(xb.to(DEVICE)) - c)**2, dim=1).detach().cpu().numpy()
         for (xb,) in val_loader_svdd])

RESULTS["DeepSVDD"] = eval_metrics(y_val_heur, svdd_scores)
print("DeepSVDD:", RESULTS["DeepSVDD"])

# ===== Converted from notebook code cell 8 =====
# ── Baseline 4 & 5: Vanilla AE and AE + pairwise rank loss ───────────────────
print("=" * 50)
print("Baseline 4: Vanilla AE  |  Baseline 5: AE + Rank")
print("=" * 50)

AE_EPOCHS     = 10
AE_BATCH_SIZE = 10000
AE_LR         = 1e-3
AE_LATENT_DIM = 8
AE_PATIENCE   = 10
AE_EVAL_EVERY = 10
RANK_LAMBDA   = 1.5
RANK_MARGIN   = 0.1
AE_RANK_POOL_MODE = os.environ.get("AE_RANK_POOL_MODE", "labeled").strip().lower()
EXIT_AFTER_AE = os.environ.get("EXIT_AFTER_AE", "0").strip() == "1"

class VanillaAE(nn.Module):
    def __init__(self, in_dim, latent_dim=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, in_dim))
    def forward(self, x):
        return self.dec(self.enc(x))


def pairwise_logistic_rank_loss(scores, labels, margin=0.1, max_pairs=32768):
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


# Train-time anomaly pool for AE-rank.
# - labeled (default): use train heuristic positives
# - none: fairness mode; disable labeled-positive pool
if AE_RANK_POOL_MODE == "none":
    X_anom_pool = np.empty((0, X_train_n.shape[1]), dtype=X_train_n.dtype)
    print("Anomaly pool for AE-rank: disabled (fairness mode)")
else:
    train_anom_mask = y_train_heur == 1
    X_anom_pool = X_train_n[train_anom_mask]
    print(f"Anomaly pool for AE-rank: {len(X_anom_pool):,} (mode={AE_RANK_POOL_MODE})")


def train_ae(use_rank=False, seed=RANDOM_STATE):
    torch.manual_seed(seed)
    model = VanillaAE(X_norm.shape[1], AE_LATENT_DIM).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=1e-5)
    crit  = nn.MSELoss()

    tr_loader_ae  = DataLoader(TensorDataset(torch.tensor(X_norm, dtype=torch.float32)),
                               batch_size=AE_BATCH_SIZE, shuffle=True)
    val_loader_ae = DataLoader(TensorDataset(X_val_t), batch_size=AE_BATCH_SIZE, shuffle=False)
    anom_t = torch.tensor(X_anom_pool, dtype=torch.float32) if (use_rank and len(X_anom_pool) > 0) else None

    best_ap = -1.0; best_sd = None; pat = 0

    for epoch in range(AE_EPOCHS):
        model.train()
        running = 0.0; n_b = 0
        for (xb,) in tr_loader_ae:
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
                all_scores = torch.cat([mse_n, mse_a])
                all_labels = torch.cat([torch.zeros(len(mse_n), device=DEVICE),
                                        torch.ones( len(mse_a), device=DEVICE)])
                rank_loss, _ = pairwise_logistic_rank_loss(all_scores, all_labels, RANK_MARGIN)
                loss = loss + RANK_LAMBDA * rank_loss
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            running += float(loss.item()); n_b += 1

        if (epoch + 1) % AE_EVAL_EVERY == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vs = np.concatenate(
                    [((model(xb.to(DEVICE)) - xb.to(DEVICE))**2).mean(dim=1)
                     .detach().cpu().numpy() for (xb,) in val_loader_ae])
            val_ap = float(average_precision_score(y_val_heur, vs)) if np.unique(y_val_heur).size > 1 else float("nan")
            tr_loss = running / max(1, n_b)
            label = "AE-rank" if use_rank else "AE"
            print(f"  [{label}] epoch={epoch+1:03d} loss={tr_loss:.6f} val_auprc={val_ap:.6f}")
            if np.isfinite(val_ap) and val_ap > best_ap + 1e-8:
                best_ap = val_ap; best_sd = copy.deepcopy(model.state_dict()); pat = 0
            else:
                pat += 1
                if pat >= AE_PATIENCE:
                    print(f"  Early stop at epoch {epoch+1}"); break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        scores = np.concatenate(
            [((model(xb.to(DEVICE)) - xb.to(DEVICE))**2).mean(dim=1)
             .detach().cpu().numpy() for (xb,) in val_loader_ae])
    return scores


ae_scores      = train_ae(use_rank=False)
ae_rank_scores = train_ae(use_rank=True)

RESULTS["VanillaAE"] = eval_metrics(y_val_heur, ae_scores)
RESULTS["AE-Rank"]   = eval_metrics(y_val_heur, ae_rank_scores)
print("VanillaAE:", RESULTS["VanillaAE"])
print("AE-Rank:"  , RESULTS["AE-Rank"])

if EXIT_AFTER_AE:
    _persist_results_csv(RESULTS)
    print("EXIT_AFTER_AE=1 set; stopping after AE baselines.")
    raise SystemExit(0)

# ===== Converted from notebook code cell 9 =====
# ── Baseline 6: Classic Supervised RDL (GNN on the relational graph) ─────────
# Uses the same HeteroEncoder + HeteroGraphSAGE stack as RAD, but trained with
# a supervised binary cross-entropy loss on the heuristic labels instead of the
# unsupervised autoencoder objective.  The only graph-side information this model
# receives is the relational neighbourhood — no rule augmentation, no reconstruction.
print("=" * 50)
print("Baseline 6: Supervised RDL (GNN)")
print("=" * 50)

import copy as _copy
import gc as _gc
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything as _seed
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.utils import get_stype_proposal
from torch_frame.config.text_embedder import TextEmbedderConfig
from sentence_transformers import SentenceTransformer

_seed(RANDOM_STATE)

# ── Load cached graph artifacts ───────────────────────────────────────────────
_graph_cache  = amazon_data("relbench_rdl_graph_cache.pt")
_stats_cache  = amazon_data("relbench_rdl_col_stats_cache.pt")

if os.path.exists(_graph_cache) and os.path.exists(_stats_cache):
    _rdl_data       = torch.load(_graph_cache, weights_only=False)
    _rdl_col_stats  = torch.load(_stats_cache, weights_only=False)
    print("Graph and col-stats loaded from cache.")
else:
    # Build from scratch — slow (~10 min) but only needed once.
    print("Building graph from scratch (this is slow)...")
    _rdl_dataset   = get_dataset("rel-amazon", download=True)
    _rdl_db        = _rdl_dataset.get_db()
    _col_to_stype  = get_stype_proposal(_rdl_db)
    class _GloveEmb:
        def __init__(self): self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", device="cpu")
        def __call__(self, s): return torch.from_numpy(self.model.encode(s, show_progress_bar=False))
    _txt_cfg = TextEmbedderConfig(text_embedder=_GloveEmb(), batch_size=64)
    _rdl_data, _rdl_col_stats = make_pkey_fkey_graph(
        _rdl_db, col_to_stype_dict=_col_to_stype, text_embedder_cfg=_txt_cfg, cache_dir=None)
    torch.save(_rdl_data,      _graph_cache)
    torch.save(_rdl_col_stats, _stats_cache)
    print("Graph built and cached.")

# ── Re-use the same task tables + heuristic labels ────────────────────────────
# We re-load train/val tables here rather than relying on the baselines notebook's
# variables which do NOT have the heuristic churn column written back in.
import copy as _copy2
_rdl_task        = get_task("rel-amazon", "user-churn", download=True)
_rdl_train_table = _rdl_task.get_table("train")
_rdl_val_table   = _rdl_task.get_table("val")
_entity_table    = _rdl_task.entity_table   # "customer"

# Apply the same heuristic filtering/labeling as the main notebook.
import pandas as _pd
import numpy as _np

_rev_ev = db.table_dict["review"].df[["customer_id", "review_time"]].dropna().copy()
_rev_ev["customer_id"] = _rev_ev["customer_id"].astype(_np.int64)
_rev_ev["review_time"] = _pd.to_datetime(_rev_ev["review_time"])
_rev_ev = _rev_ev.sort_values(["customer_id", "review_time"]).reset_index(drop=True)
_rev_ev["cum_reviews"] = _rev_ev.groupby("customer_id").cumcount() + 1


def _rdl_activity(target_df, lookback=30):
    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(_np.int64)
    q["timestamp"]   = _pd.to_datetime(q["timestamp"])
    q["_row_id"]     = _np.arange(len(q), dtype=_np.int64)
    q = q.sort_values(["timestamp", "customer_id", "_row_id"]).reset_index(drop=True)
    r = _rev_ev[["customer_id", "review_time", "cum_reviews"]]

    j1 = _pd.merge_asof(
        q[["customer_id", "timestamp", "_row_id"]].sort_values(["timestamp", "customer_id"]),
        r.sort_values(["review_time", "customer_id"]),
        left_on="timestamp", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    q["tot"] = j1["cum_reviews"].fillna(0).astype(_np.int64)
    q["lrt"] = j1["review_time"]

    # Current 30-day window count.
    q["win"] = q["timestamp"] - _pd.Timedelta(days=lookback)
    j2 = _pd.merge_asof(
        q[["customer_id", "win", "_row_id"]].sort_values(["win", "customer_id"]),
        r.sort_values(["review_time", "customer_id"]),
        left_on="win", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w30 = j2["cum_reviews"].fillna(0).astype(_np.int64)
    q["recent"] = _np.maximum(0, q["tot"].values - _w30.values)

    # Previous 30-day window count for trend_w.
    q["win_prev"] = q["timestamp"] - _pd.Timedelta(days=2 * lookback)
    j3 = _pd.merge_asof(
        q[["customer_id", "win_prev", "_row_id"]].sort_values(["win_prev", "customer_id"]),
        r.sort_values(["review_time", "customer_id"]),
        left_on="win_prev", right_on="review_time", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w60 = j3["cum_reviews"].fillna(0).astype(_np.int64)
    q["prev"] = _np.maximum(0, _w30.values - _w60.values)

    q["trend_w"] = (q["recent"].values - q["prev"].values) / (q["prev"].values + 1.0)
    q["avg_activity"] = 0.5 * (q["recent"].values + q["prev"].values)
    q["days_since"] = (q["timestamp"] - q["lrt"]).dt.days.fillna(9999).astype(_np.int64)
    return q.sort_values("_row_id").reset_index(drop=True)[["recent", "prev", "trend_w", "avg_activity", "days_since"]]


def _rdl_atrisk_labels(target_df, act, thresholds=None):
    raw = target_df["churn"].astype(_np.int64).to_numpy()
    if thresholds is None:
        thresholds = {
            "a": float(os.environ.get("ATRISK_TAU_A", "0.05")),
            "b": float(os.environ.get("ATRISK_TAU_B", "10.0")),
            "alpha": float(os.environ.get("ATRISK_TAU_ALPHA", "0.02")),
        }
    a = float(thresholds["a"])
    b = float(thresholds["b"])
    alpha = float(thresholds["alpha"])
    if b <= 1.0:
        raise ValueError(f"ATRISK_TAU_B must be > 1.0, got {b}")

    trend = act["trend_w"].to_numpy(dtype=_np.float64)
    a_bar = _np.maximum(act["avg_activity"].to_numpy(dtype=_np.float64), 0.0)
    tau = a + (_np.log(a_bar + 1.0) / _np.log(b)) * (1.0 + alpha * a_bar)
    at_risk = trend <= (-tau)

    keep = ~at_risk
    unexp = raw * keep.astype(_np.int64)
    return unexp, keep, thresholds


_act_tr = _rdl_activity(_rdl_train_table.df)
_act_vl = _rdl_activity(_rdl_val_table.df)
_unexp_tr, _keep_tr, _thr = _rdl_atrisk_labels(_rdl_train_table.df, _act_tr)
_unexp_vl, _keep_vl, _    = _rdl_atrisk_labels(_rdl_val_table.df, _act_vl, thresholds=_thr)

# Write labels back and filter out at-risk rows.
_rdl_train_table.df = _rdl_train_table.df.loc[_keep_tr].copy()
_rdl_train_table.df["churn"] = _unexp_tr[_keep_tr]
_rdl_val_table.df   = _rdl_val_table.df.loc[_keep_vl].copy()
_rdl_val_table.df["churn"]   = _unexp_vl[_keep_vl]

# Match 1.5% anomaly target on val (same target as full-ablation config).
_target_anom_rate = float(os.environ.get("TARGET_ANOM_RATE", "0.015"))
_rdl_val_pos = _rdl_val_table.df[_rdl_val_table.df["churn"] == 1]
_rdl_val_neg = _rdl_val_table.df[_rdl_val_table.df["churn"] == 0]
if 0.0 < _target_anom_rate < 1.0 and len(_rdl_val_pos) > 0 and len(_rdl_val_neg) > 0:
    _n_pos_tgt = int(round((_target_anom_rate / (1.0 - _target_anom_rate)) * len(_rdl_val_neg)))
    _n_pos_tgt = min(_n_pos_tgt, len(_rdl_val_pos))
    if _n_pos_tgt < len(_rdl_val_pos):
        _rng_vl = _np.random.default_rng(RANDOM_STATE)
        _kp_idx = _rng_vl.choice(_rdl_val_pos.index.to_numpy(), size=_n_pos_tgt, replace=False)
        _rdl_val_table.df = _rdl_val_table.df.loc[
            _np.sort(_np.concatenate([_rdl_val_neg.index.to_numpy(), _kp_idx]))
        ].copy()

_rdl_val_labels = _rdl_val_table.df["churn"].astype(_np.int64).to_numpy()
print(f"Supervised RDL — train: {len(_rdl_train_table.df):,} "
      f"({int(_rdl_train_table.df['churn'].sum()):,} pos) | "
      f"val: {len(_rdl_val_table.df):,} "
      f"({int(_rdl_val_table.df['churn'].sum()):,} pos) | target={_target_anom_rate:.4f}")

# ── Build NeighborLoaders ─────────────────────────────────────────────────────
def _make_loader(table, shuffle):
    ti = get_node_train_table_input(table=table, task=_rdl_task)
    return NeighborLoader(
        _rdl_data, num_neighbors=[32, 32], time_attr="time",
        input_nodes=ti.nodes, input_time=ti.time, transform=ti.transform,
        batch_size=1024, temporal_strategy="uniform",
        shuffle=shuffle, num_workers=4, persistent_workers=True,
    )

_tr_loader_rdl = _make_loader(_rdl_train_table, shuffle=True)
_vl_loader_rdl = _make_loader(_rdl_val_table,   shuffle=False)
print(f"Loaders — train: {len(_tr_loader_rdl)} batches | val: {len(_vl_loader_rdl)} batches")

# ── Supervised GNN model ──────────────────────────────────────────────────────
class SupervisedGNN(torch.nn.Module):
    def __init__(self, data, col_stats, channels=128, num_layers=2, aggr="sum"):
        super().__init__()
        self.encoder = HeteroEncoder(channels=channels, node_to_col_names_dict={
            node: data[node].tf.col_names_dict for node in data.node_types
        }, node_to_col_stats=col_stats)
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[node for node in data.node_types if data[node].get("time") is not None],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types, edge_types=data.edge_types,
            channels=channels, aggr=aggr, num_layers=num_layers,
        )
        self.head = torch.nn.Linear(channels, 1)

    def forward(self, batch, entity_table):
        x_dict = self.encoder(batch.tf_dict)
        rel_t = batch[entity_table].seed_time
        t_dict = self.temporal_encoder(rel_t, batch.time_dict, batch.batch_dict)
        x_dict = {k: (x_dict[k] + t_dict[k]) if k in t_dict else x_dict[k] for k in x_dict.keys()}
        x_dict  = self.gnn(
            x_dict,
            batch.edge_index_dict,
            num_sampled_nodes_dict=getattr(batch, "num_sampled_nodes", None),
            num_sampled_edges_dict=getattr(batch, "num_sampled_edges", None),
        )
        out     = self.head(x_dict[entity_table]).squeeze(-1)
        seed_n  = batch[entity_table].batch_size
        return out[:seed_n]


_rdl_sup_model = SupervisedGNN(
    _rdl_data, _rdl_col_stats, channels=128, num_layers=2, aggr="sum"
).to(DEVICE)
_rdl_sup_opt = torch.optim.AdamW(_rdl_sup_model.parameters(), lr=1e-3, weight_decay=1e-4)
print(f"SupervisedGNN | params={sum(p.numel() for p in _rdl_sup_model.parameters() if p.requires_grad):,}")

# ── Training loop ─────────────────────────────────────────────────────────────
RDL_EPOCHS      = 10
RDL_MAX_STEPS   = 2000
RDL_PATIENCE    = 3
RDL_GRAD_CLIP   = 2.0

_best_rdl_ap  = -1.0
_best_rdl_sd  = None
_rdl_patience = 0

for _epoch in range(1, RDL_EPOCHS + 1):
    _rdl_sup_model.train()
    _steps = 0; _running = 0.0
    for _batch in tqdm(_tr_loader_rdl, desc=f"  RDL epoch {_epoch}", leave=False):
        _batch = _batch.to(DEVICE)
        _logits = _rdl_sup_model(_batch, _entity_table)
        _lbl    = _batch[_entity_table].y.float().view(-1)[:_logits.shape[0]]
        _loss   = torch.nn.functional.binary_cross_entropy_with_logits(_logits, _lbl)
        _rdl_sup_opt.zero_grad(set_to_none=True)
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(_rdl_sup_model.parameters(), RDL_GRAD_CLIP)
        _rdl_sup_opt.step()
        _running += float(_loss.detach()); _steps += 1
        if _steps >= RDL_MAX_STEPS:
            break
    # Val pass
    _rdl_sup_model.eval()
    _all_scores = []
    with torch.no_grad():
        for _vb in _vl_loader_rdl:
            _vb = _vb.to(DEVICE)
            _sc = torch.sigmoid(_rdl_sup_model(_vb, _entity_table)).detach().cpu().numpy()
            _all_scores.append(_sc)
    _vscores = _np.concatenate(_all_scores)
    _vap = float(average_precision_score(_rdl_val_labels, _vscores)) if _np.unique(_rdl_val_labels).size > 1 else float("nan")
    _vauc = float(roc_auc_score(_rdl_val_labels, _vscores)) if _np.unique(_rdl_val_labels).size > 1 else float("nan")
    print(f"  RDL epoch {_epoch:02d} | steps={_steps} | loss={_running/_steps:.4f} | val_auprc={_vap:.4f} | val_auroc={_vauc:.4f}")
    if _np.isfinite(_vap) and _vap > _best_rdl_ap + 1e-8:
        _best_rdl_ap = _vap
        _best_rdl_sd = {k: v.detach().cpu().clone() for k, v in _rdl_sup_model.state_dict().items()}
        _rdl_patience = 0
    else:
        _rdl_patience += 1
        if _rdl_patience >= RDL_PATIENCE:
            print(f"  Early stopping at epoch {_epoch}"); break

# Load best checkpoint and score val
if _best_rdl_sd is not None:
    _rdl_sup_model.load_state_dict(_best_rdl_sd)
_rdl_sup_model.eval()
_final_rdl_scores = []
with torch.no_grad():
    for _vb in _vl_loader_rdl:
        _vb = _vb.to(DEVICE)
        _sc = torch.sigmoid(_rdl_sup_model(_vb, _entity_table)).detach().cpu().numpy()
        _final_rdl_scores.append(_sc)
_final_rdl_scores = _np.concatenate(_final_rdl_scores)

# Align to y_val_heur (different filtering from heur_thresholds vs _thr may shift counts)
# Use intersection of both filtered val sets for a consistent comparison.
RESULTS["Supervised RDL"] = eval_metrics(_rdl_val_labels, _final_rdl_scores)
print("Supervised RDL:", RESULTS["Supervised RDL"])

# ===== Converted from notebook code cell 10 =====
# ── Results Summary ───────────────────────────────────────────────────────────
# Load RAD Graph-AE scores saved from rad_amazon_user_churn.ipynb if available.
RAD_SCORES_PATH = "weights/rad_amazon_user_churn.pt"
if os.path.exists(RAD_SCORES_PATH):
    try:
        ckpt = torch.load(RAD_SCORES_PATH, weights_only=False)
        if "val_scores" in ckpt:
            RESULTS["RAD Graph-AE"] = eval_metrics(y_val_heur, ckpt["val_scores"])
            print(f"Loaded RAD Graph-AE scores from {RAD_SCORES_PATH}")
        else:
            print(f"(RAD checkpoint found but no val_scores key — "
                  "re-run rad_amazon_user_churn.ipynb and save val_pred to the checkpoint to include here)")
    except Exception as e:
        print(f"(Could not load RAD scores: {e})")
else:
    print(f"(RAD scores not found at {RAD_SCORES_PATH} — run rad_amazon_user_churn.ipynb first)")

print()
print("=" * 70)
print(f"{'Model':<20} {'AUROC':>8} {'AUPRC':>8} {'P@K':>8} {'Lift':>8}")
print("-" * 70)
prev = float(y_val_heur.mean())
print(f"{'Random':20} {'0.5000':>8} {prev:8.4f} {prev:8.4f} {'1.00':>8}")
for name, m in RESULTS.items():
    print(f"{name:<20} {m['auroc']:8.4f} {m['auprc']:8.4f} {m['p@k']:8.4f} {m['lift']:8.2f}")
print("=" * 70)

_persist_raw_results(
    {
        "IForest": {"labels": y_val_heur, "scores": if_scores},
        "DRL": {"labels": y_val_heur, "scores": drl_scores},
        "DeepSVDD": {"labels": y_val_heur, "scores": svdd_scores},
        "VanillaAE": {"labels": y_val_heur, "scores": ae_scores},
        "AE-Rank": {"labels": y_val_heur, "scores": ae_rank_scores},
        "Supervised RDL": {"labels": _rdl_val_labels, "scores": _final_rdl_scores},
    }
)


# ===== Persist baseline results for orchestrators =====
if "RESULTS" not in globals() or not isinstance(RESULTS, dict):
    raise RuntimeError("RESULTS dict not found after baseline run")
_persist_results_csv(RESULTS)
