#!/usr/bin/env python3
"""Standalone H&M user-churn baselines (no notebook execution)."""

import copy, gc, os, sys
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
from path_helpers import hm_data, append_results_jsonl
from eval_split import check_disjoint, downsample_positions, eval_fingerprint, fresh_table, partition_val

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
dataset = get_dataset("rel-hm", download=True)
task    = get_task("rel-hm", "user-churn", download=True)
db      = dataset.get_db()

train_table = task.get_table("train")
val_table   = fresh_table(task.get_table("val"))  # fresh: .df is reassigned below
# Same fit/report partition of val as the RAD scripts: fit rows drive early stopping,
# report rows are the only ones scored.
_val_fit_df, _val_report_df = partition_val(val_table.df, RANDOM_STATE)
check_disjoint(_val_fit_df, _val_report_df, "val fit vs report")
val_table.df = _val_report_df
print("train:", len(train_table.df), "| val (report):", len(val_table.df), "| val (fit):", len(_val_fit_df))
print("train churn rate:", train_table.df["churn"].mean().round(4),
      "| val churn rate:", val_table.df["churn"].mean().round(4))

# ── Feature engineering ───────────────────────────────────────────────────────
LOOKBACK_DAYS = 30

txn_df = db.table_dict["transactions"].df.copy()
txn_df["customer_id"] = txn_df["customer_id"].astype(np.int64)
txn_df["t_dat"] = pd.to_datetime(txn_df["t_dat"])
_price_col = "price" if "price" in txn_df.columns else None

txn_events = txn_df[["customer_id", "t_dat"]].dropna().copy()
txn_events = txn_events.sort_values(["customer_id", "t_dat"]).reset_index(drop=True)
txn_events["cum_txns"] = txn_events.groupby("customer_id").cumcount() + 1


def activity_features(target_df, lookback_days=LOOKBACK_DAYS, lookback_90=90):
    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(np.int64)
    q["timestamp"]   = pd.to_datetime(q["timestamp"])
    q["_row_id"]     = np.arange(len(q), dtype=np.int64)
    q = q.sort_values(["timestamp", "customer_id", "_row_id"]).reset_index(drop=True)

    right = txn_events[["customer_id", "t_dat", "cum_txns"]]

    now_join = pd.merge_asof(
        q[["customer_id", "timestamp", "_row_id"]].sort_values(["timestamp", "customer_id"]),
        right.sort_values(["t_dat", "customer_id"]),
        left_on="timestamp", right_on="t_dat", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    q["total_txns"] = now_join["cum_txns"].fillna(0).astype(np.int64)
    q["last_t_dat"] = now_join["t_dat"]

    # Recent 30-day window [T-30, T)
    q["window_30"] = q["timestamp"] - pd.Timedelta(days=lookback_days)
    w30_join = pd.merge_asof(
        q[["customer_id", "window_30", "_row_id"]].sort_values(["window_30", "customer_id"]),
        right.sort_values(["t_dat", "customer_id"]),
        left_on="window_30", right_on="t_dat", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w30_cum = w30_join["cum_txns"].fillna(0).astype(np.int64)
    q["recent_txns"] = np.maximum(0, q["total_txns"].values - _w30_cum.values)

    # Previous 30-day window [T-60, T-30) for trend computation
    q["window_60"] = q["timestamp"] - pd.Timedelta(days=2 * lookback_days)
    w60_join = pd.merge_asof(
        q[["customer_id", "window_60", "_row_id"]].sort_values(["window_60", "customer_id"]),
        right.sort_values(["t_dat", "customer_id"]),
        left_on="window_60", right_on="t_dat", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w60_cum = w60_join["cum_txns"].fillna(0).astype(np.int64)
    q["prev_txns_30d"] = np.maximum(0, _w30_cum.values - _w60_cum.values)
    q["trend_w"]       = ((q["recent_txns"].values - q["prev_txns_30d"].values)
                          / (q["prev_txns_30d"].values + 1.0))
    q["avg_activity"]  = 0.5 * (q["recent_txns"].values + q["prev_txns_30d"].values)

    # 90-day window
    q["window_90"] = q["timestamp"] - pd.Timedelta(days=lookback_90)
    w90_join = pd.merge_asof(
        q[["customer_id", "window_90", "_row_id"]].sort_values(["window_90", "customer_id"]),
        right.sort_values(["t_dat", "customer_id"]),
        left_on="window_90", right_on="t_dat", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    q["recent_txns_90d"] = np.maximum(
        0, q["total_txns"].values - w90_join["cum_txns"].fillna(0).astype(np.int64).values)

    q["days_since_last_txn"] = (
        (q["timestamp"] - q["last_t_dat"]).dt.days.fillna(9999).astype(np.int64))

    return q.sort_values("_row_id").reset_index(drop=True)[
        ["recent_txns", "recent_txns_90d", "days_since_last_txn", "total_txns",
         "prev_txns_30d", "trend_w", "avg_activity"]
    ]


_PA_CACHE_PATH = hm_data("hm_causal_price_article_cache.pkl")
_pa_cache: dict = {}
if os.path.exists(_PA_CACHE_PATH):
    import pickle as _pickle
    with open(_PA_CACHE_PATH, "rb") as _f:
        _pa_cache = _pickle.load(_f)
    print(f"Loaded causal price/article cache: {len(_pa_cache)} timestamps")


def _pa_cache_key(ts) -> str:
    return pd.Timestamp(ts).isoformat()


def _compute_pa_for_ts(ts):
    """Compute per-customer price stats and distinct_articles for all transactions before ts."""
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


def _price_and_article_features(target_df):
    """Compute price stats and distinct_articles causally per prediction timestamp."""
    global _pa_cache
    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(np.int64)
    q["timestamp"]   = pd.to_datetime(q["timestamp"])
    q["_row_id"]     = np.arange(len(q), dtype=np.int64)

    price_cols = ["mean_price", "price_std", "min_price", "max_price"]
    out = pd.DataFrame(0.0, index=q["_row_id"], columns=price_cols + ["distinct_articles"])

    new_entries = False
    for ts, grp in q.groupby("timestamp"):
        key = _pa_cache_key(ts)
        row_ids = grp["_row_id"].values

        if key not in _pa_cache:
            price_agg, art_agg = _compute_pa_for_ts(ts)
            _pa_cache[key] = (price_agg, art_agg)
            new_entries = True
        else:
            price_agg, art_agg = _pa_cache[key]

        if price_agg is not None:
            merged = grp[["customer_id", "_row_id"]].merge(price_agg, on="customer_id", how="left")
            for c in price_cols:
                out.loc[row_ids, c] = merged[c].fillna(0.0).values

        if art_agg is not None:
            merged2 = grp[["customer_id", "_row_id"]].merge(art_agg, on="customer_id", how="left")
            out.loc[row_ids, "distinct_articles"] = (
                merged2["distinct_articles"].fillna(0).astype(np.int64).values)

    if new_entries:
        import pickle as _pickle
        with open(_PA_CACHE_PATH, "wb") as _f:
            _pickle.dump(_pa_cache, _f)
        print(f"Saved causal price/article cache: {len(_pa_cache)} timestamps → {_PA_CACHE_PATH}")

    return out.sort_index()


def build_X(target_df):
    act = activity_features(target_df)

    feat = pd.DataFrame(index=range(len(target_df)))
    feat["recent_txns_30d"]     = act["recent_txns"].values
    feat["recent_txns_90d"]     = act["recent_txns_90d"].values
    feat["days_since_last_txn"] = act["days_since_last_txn"].values
    feat["total_txns"]          = act["total_txns"].values

    pa = _price_and_article_features(target_df)
    for c in ["mean_price", "price_std", "min_price", "max_price", "distinct_articles"]:
        feat[c] = pa[c].values

    FEAT_COLS = [
        "recent_txns_30d", "recent_txns_90d", "days_since_last_txn", "total_txns",
        "mean_price", "price_std", "min_price", "max_price", "distinct_articles",
    ]
    X = feat[FEAT_COLS].fillna(0).values.astype(np.float32)
    return X, act


print("Computing train features...")
X_train_all, act_train = build_X(train_table.df)
print("Computing val features...")
X_val_all,   act_val   = build_X(val_table.df)
X_fit_all,   act_fit   = build_X(_val_fit_df)
print(f"Feature matrix — train: {X_train_all.shape}, val: {X_val_all.shape}")

# ── At-Risk labeling (quantile-based for H&M) ─────────────────────────────────
TARGET_ANOM_RATE = float(os.environ.get("TARGET_ANOM_RATE", "0.015"))
TRAIN_EQ_RATE    = os.environ.get("TRAIN_EQ_RATE", "1").strip() != "0"

# AtRisk(u) = 1[ trend_w(u) <= -tau(avg_activity) ]
# tau(a_bar) = a + log_b(a_bar+1) * (1 + alpha*a_bar)
_ATRISK_A     = float(os.environ.get("ATRISK_TAU_A",     "0.05"))
_ATRISK_B     = float(os.environ.get("ATRISK_TAU_B",     "10.0"))
_ATRISK_ALPHA = float(os.environ.get("ATRISK_TAU_ALPHA", "0.02"))


def atrisk_labels(target_df, activity_df, thresholds=None):
    raw     = target_df["churn"].astype(np.int64).to_numpy()
    trend   = activity_df["trend_w"].to_numpy(dtype=np.float64)
    a_bar   = np.maximum(activity_df["avg_activity"].to_numpy(dtype=np.float64), 0.0)
    if thresholds is None:
        thresholds = {
            "a": _ATRISK_A,
            "b": _ATRISK_B,
            "alpha": _ATRISK_ALPHA,
            "tau_multiplier": float(os.environ.get("ATRISK_TAU_MULTIPLIER", "1.0")),
        }
    a, b, alpha = thresholds["a"], thresholds["b"], thresholds["alpha"]
    # Must match the RAD scripts, otherwise baselines and RAD are scored on different anomaly labels.
    tau_multiplier = float(thresholds.get("tau_multiplier", 1.0))
    if tau_multiplier <= 0.0:
        raise ValueError(f"ATRISK_TAU_MULTIPLIER must be > 0, got {tau_multiplier}")
    tau     = (a + (np.log(a_bar + 1.0) / np.log(b)) * (1.0 + alpha * a_bar)) * tau_multiplier
    at_risk = trend <= (-tau)
    keep_mask      = ~at_risk
    anomaly_labels = raw * keep_mask.astype(np.int64)
    return anomaly_labels, keep_mask, thresholds


y_train_raw, train_keep, atrisk_thresholds = atrisk_labels(train_table.df, act_train)

X_train_all  = X_train_all[train_keep]
act_train    = act_train.iloc[train_keep].reset_index(drop=True)
y_train_heur = y_train_raw[train_keep]

if TRAIN_EQ_RATE:
    _tr_neg_idx      = np.where(y_train_heur == 0)[0]
    _tr_pos_idx      = np.where(y_train_heur == 1)[0]
    _n_pos_target_tr = int(round(TARGET_ANOM_RATE / (1.0 - TARGET_ANOM_RATE) * len(_tr_neg_idx)))
    _n_pos_target_tr = min(_n_pos_target_tr, len(_tr_pos_idx))
    _rng_tr          = np.random.default_rng(RANDOM_STATE)
    _keep_pos_tr     = _rng_tr.choice(_tr_pos_idx, size=_n_pos_target_tr, replace=False)
    _tr_mask         = np.sort(np.concatenate([_tr_neg_idx, _keep_pos_tr]))
    y_train_heur     = y_train_heur[_tr_mask]
    X_train_all      = X_train_all[_tr_mask]
    act_train        = act_train.iloc[_tr_mask].reset_index(drop=True)



def _atrisk_and_downsample(target_df, X, act):
    """At-risk filter, then downsample positives to TARGET_ANOM_RATE (as the RAD scripts do)."""
    y_raw, keep, _ = atrisk_labels(target_df, act, thresholds=atrisk_thresholds)
    keep_pos = np.where(keep)[0]
    m = downsample_positions(y_raw[keep_pos], TARGET_ANOM_RATE, RANDOM_STATE)
    rows = keep_pos[m]
    return X[rows], act.iloc[rows].reset_index(drop=True), y_raw[rows], target_df.iloc[rows]


X_val_all, act_val, y_val_heur, _val_scored_df = _atrisk_and_downsample(val_table.df, X_val_all, act_val)
X_fit_all, act_fit, y_fit_heur, _              = _atrisk_and_downsample(_val_fit_df, X_fit_all, act_fit)

print("At-Risk thresholds:", atrisk_thresholds)
print(f"Train — total: {len(X_train_all):,} | positives: {int(y_train_heur.sum()):,} ({y_train_heur.mean():.4f})")
print(f"Val   — total: {len(y_val_heur):,}   | positives: {int(y_val_heur.sum()):,} ({y_val_heur.mean():.4f})")
print(f"Val fit (early stopping) — total: {len(y_fit_heur):,} | positives: {int(y_fit_heur.sum()):,}")
VAL_EVAL_FP = eval_fingerprint(_val_scored_df, y_val_heur)
print(f"  eval_fingerprint: {VAL_EVAL_FP}")

train_normal_mask = y_train_heur == 0
X_mu    = X_train_all[train_normal_mask].mean(axis=0, keepdims=True)
X_sigma = X_train_all[train_normal_mask].std(axis=0, keepdims=True)
X_train_n = (X_train_all - X_mu) / (X_sigma + 1e-6)
X_val_n   = (X_val_all   - X_mu) / (X_sigma + 1e-6)
X_fit_n   = (X_fit_all   - X_mu) / (X_sigma + 1e-6)
X_norm    = X_train_n[train_normal_mask]
print(f"Train normal pool: {len(X_norm):,} | Val positives: {int(y_val_heur.sum()):,}")


def eval_metrics(y_true, scores, eval_fp=None):
    y_true = y_true.astype(int)
    scores = scores.astype(float)
    auroc  = float(roc_auc_score(y_true, scores))
    auprc  = float(average_precision_score(y_true, scores))
    k      = max(int(y_true.sum()), 1)
    top_k  = np.argsort(-scores)[:k]
    p_at_k = float(y_true[top_k].mean())
    prev   = float(y_true.mean())
    lift   = auprc / prev if prev > 0 else float("nan")
    return {"auroc": auroc, "auprc": auprc, "p@k": p_at_k, "lift": lift,
            "n_eval": int(len(y_true)), "n_pos": int(y_true.sum()),
            "eval_fp": eval_fp or VAL_EVAL_FP}


RESULTS: dict[str, dict] = {}


def _persist_results_csv(results: dict) -> None:
    root_dir = Path(__file__).resolve().parents[1]
    out_dir  = Path(os.environ.get("BASELINES_OUT_DIR", "") or (root_dir / "results" / "hm" / "full_ablation_hm"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, met in results.items():
        if isinstance(met, dict):
            rows.append({
                "model": model,
                "auroc": float(met.get("auroc", float("nan"))),
                "auprc": float(met.get("auprc", float("nan"))),
                "p@k":   float(met.get("p@k",   float("nan"))),
                "lift":  float(met.get("lift",  float("nan"))),
                "n_eval": met.get("n_eval"),
                "n_pos": met.get("n_pos"),
                "eval_fp": met.get("eval_fp"),
                "source": "baselines_script",
            })
    df = pd.DataFrame(rows)
    # Seed-specific path: a fixed overwritten path loses every prior seed's raw output.
    csv_path = out_dir / f"baselines_metrics_seed{RANDOM_STATE}.csv"
    df.to_csv(csv_path, index=False)
    # Also keep writing the un-seeded path for backward compatibility with existing consumers.
    (out_dir / "baselines_metrics.csv").write_text(df.to_csv(index=False))
    print(f"[baselines] Wrote metrics CSV: {csv_path}")
    for row in rows:
        append_results_jsonl({
            "task": "hm-user-churn", "seed": RANDOM_STATE, "split": "Val",
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


print(f"\nVal baseline prevalence: {y_val_heur.mean():.6f}")

# ── Baseline 1: Isolation Forest ──────────────────────────────────────────────
print("=" * 50)
print("Baseline 1: Isolation Forest")
print("=" * 50)

iforest   = IsolationForest(n_estimators=300, contamination="auto",
                            random_state=RANDOM_STATE, n_jobs=-1)
iforest.fit(X_norm)
if_scores = -iforest.decision_function(X_val_n)
RESULTS["IForest"] = eval_metrics(y_val_heur, if_scores)
print("IForest:", RESULTS["IForest"])

# ── Baseline 2: DRL / PrototypeNet ────────────────────────────────────────────
print("=" * 50)
print("Baseline 2: DRL / PrototypeNet")
print("=" * 50)

DRL_REPO = "./DRL"
if DRL_REPO not in sys.path:
    sys.path.insert(0, DRL_REPO)
from PrototypeNet.Model import PN

DRL_CFG = {
    "data_dim":         X_norm.shape[1],
    "hidden_dim":       64,
    "en_nlayers":       3,
    "de_nlayers":       3,
    "prototype_num":    5,
    "diversity":        True,
    "plearn":           False,
    "input_info":       True,
    "input_info_ratio": 0.1,
    "cl":               True,
    "cl_ratio":         0.06,
}
DRL_EPOCHS     = int(os.environ.get("DRL_EPOCHS", "10"))
DRL_BATCH_SIZE = 8192
DRL_LR         = 5e-3

drl_model = PN(DRL_CFG).to(DEVICE)
drl_opt   = torch.optim.Adam(drl_model.parameters(), lr=DRL_LR, weight_decay=1e-5)
drl_sched = torch.optim.lr_scheduler.ExponentialLR(drl_opt, gamma=0.98)

X_t       = torch.tensor(X_norm, dtype=torch.float32)
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
            h_    = drl_model.phi(xb)
            pick  = max(2, int(h_.shape[0] * 0.8))
            sel   = torch.randperm(h_.shape[0], device=h_.device)[:pick]
            h_    = h_[sel]
            sim   = h_ @ h_.T
            denom = (torch.sqrt((h_**2).sum(-1, keepdim=True))
                     @ torch.sqrt((h_.T**2).sum(0, keepdim=True)))
            sim   = sim / (denom + 1e-8)
            loss  = loss + DRL_CFG["cl_ratio"] * (sim - torch.eye(h_.shape[0], device=h_.device)).pow(2).mean()
        drl_opt.zero_grad(set_to_none=True); loss.backward(); drl_opt.step()
        running += float(loss.item()); steps += 1
    drl_sched.step()
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"  epoch={epoch+1:03d} loss={running/max(steps,1):.6f}")

drl_model.eval()
X_val_t    = torch.tensor(X_val_n, dtype=torch.float32)
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
    def forward(self, x): return self.net(x)


svdd_model     = DeepSVDDNet(X_norm.shape[1], rep_dim=SVDD_REP_DIM).to(DEVICE)
svdd_opt       = torch.optim.Adam(svdd_model.parameters(), lr=SVDD_LR, weight_decay=1e-5)
tr_loader_svdd = DataLoader(TensorDataset(X_t), batch_size=SVDD_BATCH_SIZE, shuffle=True)

svdd_model.eval()
with torch.no_grad():
    z0  = torch.cat([svdd_model(xb.to(DEVICE)) for (xb,) in tr_loader_svdd], dim=0)
    c   = z0.mean(dim=0)
    eps = 0.1
    c[(c.abs() < eps) & (c < 0)]  = -eps
    c[(c.abs() < eps) & (c >= 0)] =  eps

rng_svdd     = np.random.default_rng(RANDOM_STATE)
hold_idx     = rng_svdd.choice(len(X_norm), size=max(1, len(X_norm)//10), replace=False)
X_t_sub      = torch.tensor(np.delete(X_norm, hold_idx, axis=0), dtype=torch.float32)
X_val_t_svdd = torch.tensor(X_val_n, dtype=torch.float32)
X_fit_t_svdd = torch.tensor(X_fit_n, dtype=torch.float32)

tr_loader_sub   = DataLoader(TensorDataset(X_t_sub),      batch_size=SVDD_BATCH_SIZE, shuffle=True)
val_loader_svdd = DataLoader(TensorDataset(X_val_t_svdd), batch_size=SVDD_BATCH_SIZE, shuffle=False)
fit_loader_svdd = DataLoader(TensorDataset(X_fit_t_svdd), batch_size=SVDD_BATCH_SIZE, shuffle=False)

best_svdd_ap = -1.0; best_svdd_sd = None; patience_cnt = 0
svdd_model.train()
for epoch in range(SVDD_EPOCHS):
    running = 0.0; n_b = 0
    for (xb,) in tr_loader_sub:
        xb   = xb.to(DEVICE)
        z    = svdd_model(xb)
        loss = torch.sum((z - c) ** 2, dim=1).mean()
        svdd_opt.zero_grad(set_to_none=True); loss.backward(); svdd_opt.step()
        running += float(loss.item()); n_b += 1
    tr_loss = running / max(1, n_b)
    if tr_loss < 1e-5:
        print(f"  SVDD collapse at epoch {epoch+1}"); break
    if (epoch + 1) % SVDD_EVAL_EVERY == 0 or epoch == 0:
        svdd_model.eval()
        with torch.no_grad():
            vs = np.concatenate([
                torch.sum((svdd_model(xb.to(DEVICE)) - c)**2, dim=1).detach().cpu().numpy()
                for (xb,) in fit_loader_svdd
            ])
        val_ap = float(average_precision_score(y_fit_heur, vs)) if np.unique(y_fit_heur).size > 1 else float("nan")
        print(f"  epoch={epoch+1:03d} train_loss={tr_loss:.6f} val_auprc={val_ap:.6f}")
        if np.isfinite(val_ap) and val_ap > best_svdd_ap + 1e-8:
            best_svdd_ap = val_ap; best_svdd_sd = copy.deepcopy(svdd_model.state_dict()); patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= SVDD_PATIENCE:
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
print("DeepSVDD:", RESULTS["DeepSVDD"])

# ── Baseline 4 & 5: Vanilla AE and AE + pairwise rank loss ───────────────────
print("=" * 50)
print("Baseline 4: Vanilla AE  |  Baseline 5: AE + Rank")
print("=" * 50)

AE_EPOCHS         = 10
AE_BATCH_SIZE     = 10000
AE_LR             = 1e-3
AE_LATENT_DIM     = 8
AE_PATIENCE       = 10
AE_EVAL_EVERY     = 10
RANK_LAMBDA       = 1.5
RANK_MARGIN       = 0.1
AE_RANK_POOL_MODE = os.environ.get("AE_RANK_POOL_MODE", "labeled").strip().lower()
EXIT_AFTER_AE     = os.environ.get("EXIT_AFTER_AE", "0").strip() == "1"


class VanillaAE(nn.Module):
    def __init__(self, in_dim, latent_dim=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, in_dim))
    def forward(self, x): return self.dec(self.enc(x))


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


if AE_RANK_POOL_MODE == "none":
    X_anom_pool = np.empty((0, X_train_n.shape[1]), dtype=X_train_n.dtype)
    print("Anomaly pool for AE-rank: disabled")
else:
    X_anom_pool = X_train_n[y_train_heur == 1]
    print(f"Anomaly pool for AE-rank: {len(X_anom_pool):,}")


def train_ae(use_rank=False, seed=RANDOM_STATE):
    torch.manual_seed(seed)
    model = VanillaAE(X_norm.shape[1], AE_LATENT_DIM).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=1e-5)
    crit  = nn.MSELoss()

    tr_loader_ae  = DataLoader(TensorDataset(torch.tensor(X_norm, dtype=torch.float32)),
                               batch_size=AE_BATCH_SIZE, shuffle=True)
    val_loader_ae = DataLoader(TensorDataset(X_val_t), batch_size=AE_BATCH_SIZE, shuffle=False)
    fit_loader_ae = DataLoader(TensorDataset(torch.tensor(X_fit_n, dtype=torch.float32)),
                               batch_size=AE_BATCH_SIZE, shuffle=False)
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
                n_anom     = min(len(xb), len(anom_t))
                idx        = torch.randperm(len(anom_t))[:n_anom]
                xa         = anom_t[idx].to(DEVICE)
                recon_a    = model(xa)
                mse_n      = ((recon   - xb) ** 2).mean(dim=1)
                mse_a      = ((recon_a - xa) ** 2).mean(dim=1)
                all_s      = torch.cat([mse_n, mse_a])
                all_l      = torch.cat([torch.zeros(len(mse_n), device=DEVICE),
                                        torch.ones( len(mse_a), device=DEVICE)])
                rl, _      = pairwise_logistic_rank_loss(all_s, all_l, RANK_MARGIN)
                loss       = loss + RANK_LAMBDA * rl
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            running += float(loss.item()); n_b += 1

        if (epoch + 1) % AE_EVAL_EVERY == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vs = np.concatenate([
                    ((model(xb.to(DEVICE)) - xb.to(DEVICE))**2).mean(dim=1).detach().cpu().numpy()
                    for (xb,) in fit_loader_ae
                ])
            val_ap  = float(average_precision_score(y_fit_heur, vs)) if np.unique(y_fit_heur).size > 1 else float("nan")
            tr_loss = running / max(1, n_b)
            label   = "AE-rank" if use_rank else "AE"
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
        scores = np.concatenate([
            ((model(xb.to(DEVICE)) - xb.to(DEVICE))**2).mean(dim=1).detach().cpu().numpy()
            for (xb,) in val_loader_ae
        ])
    return scores


ae_scores      = train_ae(use_rank=False)
ae_rank_scores = train_ae(use_rank=True)

RESULTS["VanillaAE"] = eval_metrics(y_val_heur, ae_scores)
RESULTS["AE-Rank"]   = eval_metrics(y_val_heur, ae_rank_scores)
print("VanillaAE:", RESULTS["VanillaAE"])
print("AE-Rank:",   RESULTS["AE-Rank"])

if EXIT_AFTER_AE:
    _persist_results_csv(RESULTS)
    print("EXIT_AFTER_AE=1; stopping after AE baselines.")
    raise SystemExit(0)

# ── Baseline 6: Supervised RDL (GNN on the relational graph) ─────────────────
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

_graph_cache = hm_data("relbench_rdl_graph_cache_hm.pt")
_stats_cache = hm_data("relbench_rdl_col_stats_cache_hm.pt")

def _sanitize_graph(data):
    for ntype in data.node_types:
        node = data[ntype]
        if hasattr(node, "tf") and node.tf is not None and hasattr(node.tf, "feat_dict"):
            for key, t in node.tf.feat_dict.items():
                if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                    node.tf.feat_dict[key] = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return data


if os.path.exists(_graph_cache) and os.path.exists(_stats_cache):
    _rdl_data      = _sanitize_graph(torch.load(_graph_cache, weights_only=False))
    _rdl_col_stats = torch.load(_stats_cache, weights_only=False)
    print("H&M graph and col-stats loaded from cache.")
else:
    print("Building H&M graph from scratch (slow, ~10 min)...")
    _rdl_dataset  = get_dataset("rel-hm", download=True)
    _rdl_db       = _rdl_dataset.get_db()
    _col_to_stype = get_stype_proposal(_rdl_db)

    class _GloveEmb:
        def __init__(self):
            self.model = SentenceTransformer(
                "sentence-transformers/average_word_embeddings_glove.6B.300d", device="cpu")
        def __call__(self, s):
            return torch.from_numpy(self.model.encode(s, show_progress_bar=False))

    _txt_cfg = TextEmbedderConfig(text_embedder=_GloveEmb(), batch_size=64)
    _rdl_data, _rdl_col_stats = make_pkey_fkey_graph(
        _rdl_db, col_to_stype_dict=_col_to_stype, text_embedder_cfg=_txt_cfg, cache_dir=None)
    torch.save(_rdl_data,      _graph_cache)
    torch.save(_rdl_col_stats, _stats_cache)
    print("H&M graph built and cached.")

_rdl_task        = get_task("rel-hm", "user-churn", download=True)
# fresh_table: get_table() is cached, and these tables are filtered in place below.
_rdl_train_table = fresh_table(_rdl_task.get_table("train"))
_rdl_val_table   = fresh_table(_rdl_task.get_table("val"))
_rdl_val_fit_df, _rdl_val_report_df = partition_val(_rdl_val_table.df, RANDOM_STATE)
check_disjoint(_rdl_val_fit_df, _rdl_val_report_df, "Supervised RDL val fit vs report")
_rdl_val_fit_table = _copy.deepcopy(_rdl_val_table)
_rdl_val_fit_table.df = _rdl_val_fit_df
_rdl_val_table.df = _rdl_val_report_df
_entity_table    = _rdl_task.entity_table

import pandas as _pd
import numpy as _np

_txn_ev = db.table_dict["transactions"].df[["customer_id", "t_dat"]].dropna().copy()
_txn_ev["customer_id"] = _txn_ev["customer_id"].astype(_np.int64)
_txn_ev["t_dat"]       = _pd.to_datetime(_txn_ev["t_dat"])
_txn_ev = _txn_ev.sort_values(["customer_id", "t_dat"]).reset_index(drop=True)
_txn_ev["cum_txns"] = _txn_ev.groupby("customer_id").cumcount() + 1

def _rdl_activity(target_df, lookback=30):
    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(_np.int64)
    q["timestamp"]   = _pd.to_datetime(q["timestamp"])
    q["_row_id"]     = _np.arange(len(q), dtype=_np.int64)
    q = q.sort_values(["timestamp", "customer_id", "_row_id"]).reset_index(drop=True)
    r = _txn_ev[["customer_id", "t_dat", "cum_txns"]]

    j1 = _pd.merge_asof(
        q[["customer_id", "timestamp", "_row_id"]].sort_values(["timestamp", "customer_id"]),
        r.sort_values(["t_dat", "customer_id"]),
        left_on="timestamp", right_on="t_dat", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    q["tot"] = j1["cum_txns"].fillna(0).astype(_np.int64)

    q["win30"] = q["timestamp"] - _pd.Timedelta(days=lookback)
    j2 = _pd.merge_asof(
        q[["customer_id", "win30", "_row_id"]].sort_values(["win30", "customer_id"]),
        r.sort_values(["t_dat", "customer_id"]),
        left_on="win30", right_on="t_dat", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w30 = j2["cum_txns"].fillna(0).astype(_np.int64)
    q["recent_txns"] = _np.maximum(0, q["tot"].values - _w30.values)

    q["win60"] = q["timestamp"] - _pd.Timedelta(days=2 * lookback)
    j3 = _pd.merge_asof(
        q[["customer_id", "win60", "_row_id"]].sort_values(["win60", "customer_id"]),
        r.sort_values(["t_dat", "customer_id"]),
        left_on="win60", right_on="t_dat", by="customer_id",
        direction="backward", allow_exact_matches=False,
    )
    _w60 = j3["cum_txns"].fillna(0).astype(_np.int64)
    q["prev_txns_30d"] = _np.maximum(0, _w30.values - _w60.values)
    q["trend_w"]      = (q["recent_txns"].values - q["prev_txns_30d"].values) / (q["prev_txns_30d"].values + 1.0)
    q["avg_activity"] = 0.5 * (q["recent_txns"].values + q["prev_txns_30d"].values)

    return q.sort_values("_row_id").reset_index(drop=True)[["recent_txns", "prev_txns_30d", "trend_w", "avg_activity"]]


def _rdl_atrisk(target_df, act, thresholds=None):
    raw   = target_df["churn"].astype(_np.int64).to_numpy()
    trend = act["trend_w"].to_numpy(dtype=_np.float64)
    a_bar = _np.maximum(act["avg_activity"].to_numpy(dtype=_np.float64), 0.0)
    if thresholds is None:
        thresholds = {"a": _ATRISK_A, "b": _ATRISK_B, "alpha": _ATRISK_ALPHA,
                      "tau_multiplier": float(os.environ.get("ATRISK_TAU_MULTIPLIER", "1.0"))}
    a, b, alpha = thresholds["a"], thresholds["b"], thresholds["alpha"]
    tau   = (a + (_np.log(a_bar + 1.0) / _np.log(b)) * (1.0 + alpha * a_bar)) * float(thresholds.get("tau_multiplier", 1.0))
    at_risk = trend <= (-tau)
    keep  = ~at_risk
    unexp = raw * keep.astype(_np.int64)
    return unexp, keep, thresholds


_act_tr = _rdl_activity(_rdl_train_table.df)
_unexp_tr, _keep_tr, _thr = _rdl_atrisk(_rdl_train_table.df, _act_tr)
_rdl_train_table.df = _rdl_train_table.df.loc[_keep_tr].copy()
_rdl_train_table.df["churn"] = _unexp_tr[_keep_tr]

_target_anom_rate = float(os.environ.get("TARGET_ANOM_RATE", "0.015"))


def _rdl_eval_table(table):
    """At-risk filter and downsample a val partition in place, as the RAD scripts do."""
    act = _rdl_activity(table.df)
    unexp, keep, _ = _rdl_atrisk(table.df, act, thresholds=_thr)
    table.df = table.df.loc[keep].copy()
    table.df["churn"] = unexp[keep]
    m = downsample_positions(table.df["churn"].to_numpy(), _target_anom_rate, RANDOM_STATE)
    table.df = table.df.iloc[m].copy()
    return table.df["churn"].astype(_np.int64).to_numpy()


_rdl_val_labels = _rdl_eval_table(_rdl_val_table)
_rdl_eval_fp = eval_fingerprint(_rdl_val_table.df, _rdl_val_labels)
_rdl_fit_labels = _rdl_eval_table(_rdl_val_fit_table)
print(f"Supervised RDL — train: {len(_rdl_train_table.df):,} "
      f"({int(_rdl_train_table.df['churn'].sum()):,} pos) | "
      f"val fit: {len(_rdl_val_fit_table.df):,} ({int(_rdl_fit_labels.sum()):,} pos) | "
      f"val report: {len(_rdl_val_table.df):,} "
      f"({int(_rdl_val_table.df['churn'].sum()):,} pos) | "
      f"eval_fingerprint: {_rdl_eval_fp}")


def _make_rdl_loader(table, shuffle):
    ti = get_node_train_table_input(table=table, task=_rdl_task)
    return NeighborLoader(
        _rdl_data, num_neighbors=[32, 32], time_attr="time",
        input_nodes=ti.nodes, input_time=ti.time, transform=ti.transform,
        batch_size=1024, temporal_strategy="uniform",
        shuffle=shuffle, num_workers=0, persistent_workers=False,
    )


_tr_loader_rdl = _make_rdl_loader(_rdl_train_table, shuffle=True)
_vl_loader_rdl = _make_rdl_loader(_rdl_val_table,   shuffle=False)
_fit_loader_rdl = _make_rdl_loader(_rdl_val_fit_table, shuffle=False)
print(f"Loaders — train: {len(_tr_loader_rdl)} batches | val: {len(_vl_loader_rdl)} batches")


class SupervisedGNN(torch.nn.Module):
    def __init__(self, data, col_stats, channels=128, num_layers=2, aggr="sum"):
        super().__init__()
        self.encoder = HeteroEncoder(channels=channels, node_to_col_names_dict={
            n: data[n].tf.col_names_dict for n in data.node_types
        }, node_to_col_stats=col_stats)
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[n for n in data.node_types if data[n].get("time") is not None],
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
                          num_sampled_nodes_dict=getattr(batch, "num_sampled_nodes_dict", None),
                          num_sampled_edges_dict=getattr(batch, "num_sampled_edges_dict", None))
        out    = self.head(x_dict[entity_table]).squeeze(-1)
        return out[:batch[entity_table].batch_size]


_rdl_sup_model = SupervisedGNN(_rdl_data, _rdl_col_stats, channels=128, num_layers=2).to(DEVICE)
_rdl_sup_opt   = torch.optim.AdamW(_rdl_sup_model.parameters(), lr=5e-4, weight_decay=1e-4)
print(f"SupervisedGNN | params={sum(p.numel() for p in _rdl_sup_model.parameters() if p.requires_grad):,}")

RDL_EPOCHS    = int(os.environ.get("RDL_EPOCHS", "10"))
RDL_MAX_STEPS = 2000
RDL_PATIENCE  = 3
RDL_GRAD_CLIP = 1.0

_best_rdl_ap = -1.0; _best_rdl_sd = None; _rdl_patience = 0

for _epoch in range(1, RDL_EPOCHS + 1):
    _rdl_sup_model.train()
    _steps = 0; _running = 0.0
    for _batch in tqdm(_tr_loader_rdl, desc=f"  RDL epoch {_epoch}", leave=False):
        _batch  = _batch.to(DEVICE)
        _logits = _rdl_sup_model(_batch, _entity_table)
        _lbl    = _batch[_entity_table].y.float().view(-1)[:_logits.shape[0]]
        _loss   = F.binary_cross_entropy_with_logits(_logits, _lbl)
        if not torch.isfinite(_loss):
            _rdl_sup_opt.zero_grad(set_to_none=True)
            continue
        _rdl_sup_opt.zero_grad(set_to_none=True); _loss.backward()
        torch.nn.utils.clip_grad_norm_(_rdl_sup_model.parameters(), RDL_GRAD_CLIP)
        _rdl_sup_opt.step()
        _running += float(_loss.detach()); _steps += 1
        if _steps >= RDL_MAX_STEPS: break

    if _steps == 0:
        print(f"  RDL epoch {_epoch:02d} | all batches produced NaN loss — skipping")
        _rdl_patience += 1
        if _rdl_patience >= RDL_PATIENCE:
            print(f"  Early stopping at epoch {_epoch}"); break
        continue

    _rdl_sup_model.eval()
    _all_scores = []
    with torch.no_grad():
        for _vb in _fit_loader_rdl:
            _all_scores.append(torch.sigmoid(_rdl_sup_model(_vb.to(DEVICE), _entity_table)).detach().cpu().numpy())
    _vscores = _np.nan_to_num(_np.concatenate(_all_scores), nan=0.5)
    _vap  = float(average_precision_score(_rdl_fit_labels, _vscores)) if _np.unique(_rdl_fit_labels).size > 1 else float("nan")
    _vauc = float(roc_auc_score(_rdl_fit_labels, _vscores))           if _np.unique(_rdl_fit_labels).size > 1 else float("nan")
    print(f"  RDL epoch {_epoch:02d} | steps={_steps} | loss={_running/_steps:.4f} | val_auprc={_vap:.4f} | val_auroc={_vauc:.4f}")
    if _np.isfinite(_vap) and _vap > _best_rdl_ap + 1e-8:
        _best_rdl_ap = _vap
        _best_rdl_sd = {k: v.detach().cpu().clone() for k, v in _rdl_sup_model.state_dict().items()}
        _rdl_patience = 0
    else:
        _rdl_patience += 1
        if _rdl_patience >= RDL_PATIENCE:
            print(f"  Early stopping at epoch {_epoch}"); break

if _best_rdl_sd is not None:
    _rdl_sup_model.load_state_dict(_best_rdl_sd)
_rdl_sup_model.eval()
_final_rdl_scores = []
with torch.no_grad():
    for _vb in _vl_loader_rdl:
        _final_rdl_scores.append(
            torch.sigmoid(_rdl_sup_model(_vb.to(DEVICE), _entity_table)).detach().cpu().numpy())
_final_rdl_scores = _np.concatenate(_final_rdl_scores)

RESULTS["Supervised RDL"] = eval_metrics(_rdl_val_labels, _final_rdl_scores, eval_fp=_rdl_eval_fp)
print("Supervised RDL:", RESULTS["Supervised RDL"])

# ── Results Summary ───────────────────────────────────────────────────────────
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

_persist_results_csv(RESULTS)
