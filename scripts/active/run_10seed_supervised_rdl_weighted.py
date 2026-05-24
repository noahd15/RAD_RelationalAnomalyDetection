#!/usr/bin/env python3
"""Run 10-seed supervised RDL with optional class-weighted BCE for Amazon or H&M.

Usage examples:
  source rad/bin/activate
  DATASET=amazon python scripts/active/run_10seed_supervised_rdl_weighted.py
  DATASET=hm python scripts/active/run_10seed_supervised_rdl_weighted.py

Optional env vars:
  SEEDS=0,1,2,3,4,5,6,7,8,9
  START_SEED=0
  N_SEEDS=10
  WEIGHTED_LOSS=1          # 1=true, 0=false
  BASELINES_DEVICE=auto    # auto|cpu
  RDL_EPOCHS=10
  RDL_MAX_STEPS=2000
  RDL_PATIENCE=3
  RDL_GRAD_CLIP=2.0
  TARGET_ANOM_RATE=0.015
"""

from __future__ import annotations

import gc
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.tasks import get_task
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import NeighborLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.active.path_helpers import ROOT_DIR, amazon_data, hm_data  # noqa: E402


DATASET = os.environ.get("DATASET", "amazon").strip().lower()
if DATASET not in {"amazon", "hm"}:
    raise ValueError(f"Unsupported DATASET={DATASET!r}; use 'amazon' or 'hm'.")

seeds_env = os.environ.get("SEEDS", "").strip()
if seeds_env:
    SEEDS = [int(s.strip()) for s in seeds_env.split(",") if s.strip()]
else:
    start_seed = int(os.environ.get("START_SEED", "0"))
    n_seeds = int(os.environ.get("N_SEEDS", "10"))
    SEEDS = list(range(start_seed, start_seed + n_seeds))

WEIGHTED_LOSS = os.environ.get("WEIGHTED_LOSS", "1").strip().lower() in {"1", "true", "yes", "y"}

_denv = os.environ.get("BASELINES_DEVICE", "auto").strip().lower()
if _denv == "cpu":
    DEVICE = "cpu"
else:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

RDL_EPOCHS = int(os.environ.get("RDL_EPOCHS", "10"))
RDL_MAX_STEPS = int(os.environ.get("RDL_MAX_STEPS", "2000"))
RDL_PATIENCE = int(os.environ.get("RDL_PATIENCE", "3"))
RDL_GRAD_CLIP = float(os.environ.get("RDL_GRAD_CLIP", "2.0"))

ATRISK_A = float(os.environ.get("ATRISK_TAU_A", "0.05"))
ATRISK_B = float(os.environ.get("ATRISK_TAU_B", "10.0"))
ATRISK_ALPHA = float(os.environ.get("ATRISK_TAU_ALPHA", "0.02"))
TARGET_ANOM_RATE = float(os.environ.get("TARGET_ANOM_RATE", "0.015"))

OUT_DIR = ROOT_DIR / "results" / DATASET / ("rdl_weighted_10seed" if WEIGHTED_LOSS else "rdl_unweighted_10seed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_SEED_CSV = OUT_DIR / "per_seed.csv"
SUMMARY_CSV = OUT_DIR / "summary_mean_std.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    scores = scores.astype(float)
    if np.unique(y_true).size > 1:
        auroc = float(roc_auc_score(y_true, scores))
        auprc = float(average_precision_score(y_true, scores))
    else:
        auroc = float("nan")
        auprc = float("nan")
    k = max(int(y_true.sum()), 1)
    top_k = np.argsort(-scores)[:k]
    p_at_k = float(y_true[top_k].mean())
    prev = float(y_true.mean())
    lift = (auprc / prev) if (prev > 0 and np.isfinite(auprc)) else float("nan")
    return {"auroc": auroc, "auprc": auprc, "p@k": p_at_k, "lift": lift}


def compute_activity(target_df: pd.DataFrame, events_df: pd.DataFrame, event_time_col: str, lookback_days: int = 30) -> pd.DataFrame:
    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(np.int64)
    q["timestamp"] = pd.to_datetime(q["timestamp"])
    q["_row_id"] = np.arange(len(q), dtype=np.int64)
    q = q.sort_values(["timestamp", "customer_id", "_row_id"]).reset_index(drop=True)

    right = events_df[["customer_id", event_time_col, "cum_events"]]

    now_join = pd.merge_asof(
        q[["customer_id", "timestamp", "_row_id"]].sort_values(["timestamp", "customer_id"]),
        right.sort_values([event_time_col, "customer_id"]),
        left_on="timestamp",
        right_on=event_time_col,
        by="customer_id",
        direction="backward",
        allow_exact_matches=False,
    )
    q["events_before_now"] = now_join["cum_events"].fillna(0).astype(np.int64)

    q["window_start"] = q["timestamp"] - pd.Timedelta(days=lookback_days)
    start_join = pd.merge_asof(
        q[["customer_id", "window_start", "_row_id"]].sort_values(["window_start", "customer_id"]),
        right.sort_values([event_time_col, "customer_id"]),
        left_on="window_start",
        right_on=event_time_col,
        by="customer_id",
        direction="backward",
        allow_exact_matches=False,
    )
    w30 = start_join["cum_events"].fillna(0).astype(np.int64)
    q["recent_events"] = np.maximum(0, q["events_before_now"].values - w30.values)

    q["window_60"] = q["timestamp"] - pd.Timedelta(days=2 * lookback_days)
    w60_join = pd.merge_asof(
        q[["customer_id", "window_60", "_row_id"]].sort_values(["window_60", "customer_id"]),
        right.sort_values([event_time_col, "customer_id"]),
        left_on="window_60",
        right_on=event_time_col,
        by="customer_id",
        direction="backward",
        allow_exact_matches=False,
    )
    w60 = w60_join["cum_events"].fillna(0).astype(np.int64)

    q["prev_events"] = np.maximum(0, w30.values - w60.values)
    q["trend_w"] = (q["recent_events"].values - q["prev_events"].values) / (q["prev_events"].values + 1.0)
    q["avg_activity"] = 0.5 * (q["recent_events"].values + q["prev_events"].values)

    q = q.sort_values("_row_id").reset_index(drop=True)
    return q[["recent_events", "prev_events", "trend_w", "avg_activity"]]


def label_with_atrisk(target_df: pd.DataFrame, activity_df: pd.DataFrame, thresholds: Dict[str, float] | None = None):
    raw = target_df["churn"].astype(np.int64).to_numpy()
    if thresholds is None:
        thresholds = {"a": ATRISK_A, "b": ATRISK_B, "alpha": ATRISK_ALPHA}

    a = float(thresholds["a"])
    b = float(thresholds["b"])
    alpha = float(thresholds["alpha"])
    trend = activity_df["trend_w"].to_numpy(dtype=np.float64)
    avg = np.maximum(activity_df["avg_activity"].to_numpy(dtype=np.float64), 0.0)
    tau = a + (np.log(avg + 1.0) / np.log(b)) * (1.0 + alpha * avg)

    at_risk = trend <= (-tau)
    keep = ~at_risk
    labels = raw * keep.astype(np.int64)
    return labels, keep, thresholds


class SupervisedGNN(torch.nn.Module):
    def __init__(self, data, col_stats, channels: int = 128, num_layers: int = 2, aggr: str = "sum"):
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
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = torch.nn.Linear(channels, 1)

    def forward(self, batch, entity_table: str):
        x_dict = self.encoder(batch.tf_dict)
        rel_t = batch[entity_table].seed_time
        t_dict = self.temporal_encoder(rel_t, batch.time_dict, batch.batch_dict)
        x_dict = {k: (x_dict[k] + t_dict[k]) if k in t_dict else x_dict[k] for k in x_dict}
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            num_sampled_nodes_dict=getattr(batch, "num_sampled_nodes", None),
            num_sampled_edges_dict=getattr(batch, "num_sampled_edges", None),
        )
        out = self.head(x_dict[entity_table]).squeeze(-1)
        return out[: batch[entity_table].batch_size]


def make_loader(graph_data, task, table, shuffle: bool):
    ti = get_node_train_table_input(table=table, task=task)
    return NeighborLoader(
        graph_data,
        num_neighbors=[32, 32],
        time_attr="time",
        input_nodes=ti.nodes,
        input_time=ti.time,
        transform=ti.transform,
        batch_size=1024,
        temporal_strategy="uniform",
        shuffle=shuffle,
        num_workers=4,
        persistent_workers=True,
    )


def load_dataset_assets(dataset_key: str):
    if dataset_key == "amazon":
        rel_name = "rel-amazon"
        graph_cache = Path(amazon_data("relbench_rdl_graph_cache.pt"))
        stats_cache = Path(amazon_data("relbench_rdl_col_stats_cache.pt"))
        event_table_name = "review"
        event_time_col = "review_time"
    else:
        rel_name = "rel-hm"
        graph_cache = Path(hm_data("relbench_rdl_graph_cache_hm.pt"))
        stats_cache = Path(hm_data("relbench_rdl_col_stats_cache_hm.pt"))
        event_table_name = "transactions"
        event_time_col = "t_dat"

    if not graph_cache.exists() or not stats_cache.exists():
        raise FileNotFoundError(
            f"Missing graph/stat caches for {dataset_key}: {graph_cache} | {stats_cache}"
        )

    dataset = get_dataset(rel_name, download=True)
    task = get_task(rel_name, "user-churn", download=True)
    db = dataset.get_db()

    events = db.table_dict[event_table_name].df[["customer_id", event_time_col]].dropna().copy()
    events["customer_id"] = events["customer_id"].astype(np.int64)
    events[event_time_col] = pd.to_datetime(events[event_time_col])
    events = events.sort_values(["customer_id", event_time_col]).reset_index(drop=True)
    events["cum_events"] = events.groupby("customer_id").cumcount() + 1

    graph_data = torch.load(str(graph_cache), weights_only=False)
    col_stats = torch.load(str(stats_cache), weights_only=False)
    return task, graph_data, col_stats, events, event_time_col


def run_seed(seed: int, task, graph_data, col_stats, events_df, event_time_col: str) -> Dict[str, float]:
    set_seed(seed)

    train_table = task.get_table("train")
    val_table = task.get_table("val")

    act_tr = compute_activity(train_table.df, events_df, event_time_col)
    act_vl = compute_activity(val_table.df, events_df, event_time_col)

    labels_tr, keep_tr, thresholds = label_with_atrisk(train_table.df, act_tr)
    labels_vl, keep_vl, _ = label_with_atrisk(val_table.df, act_vl, thresholds=thresholds)

    train_table.df = train_table.df.loc[keep_tr].copy()
    train_table.df["churn"] = labels_tr[keep_tr]
    val_table.df = val_table.df.loc[keep_vl].copy()
    val_table.df["churn"] = labels_vl[keep_vl]

    rng = np.random.default_rng(seed)
    vp = val_table.df[val_table.df["churn"] == 1]
    vn = val_table.df[val_table.df["churn"] == 0]
    n_tgt = int(round((TARGET_ANOM_RATE / (1.0 - TARGET_ANOM_RATE)) * len(vn)))
    n_tgt = min(n_tgt, len(vp))
    if n_tgt < len(vp):
        keep_pos = rng.choice(vp.index.to_numpy(), size=n_tgt, replace=False)
        val_table.df = val_table.df.loc[np.sort(np.concatenate([vn.index.to_numpy(), keep_pos]))].copy()

    val_labels = val_table.df["churn"].astype(np.int64).to_numpy()

    tr_loader = make_loader(graph_data, task, train_table, shuffle=True)
    vl_loader = make_loader(graph_data, task, val_table, shuffle=False)

    model = SupervisedGNN(graph_data, col_stats).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_pos = int(train_table.df["churn"].sum())
    train_neg = int(len(train_table.df) - train_pos)
    pos_weight_scalar = (train_neg / max(train_pos, 1)) if WEIGHTED_LOSS else 1.0
    pos_weight = torch.tensor([pos_weight_scalar], dtype=torch.float32, device=DEVICE)

    best_ap = -1.0
    best_sd = None
    patience = 0
    entity_table = task.entity_table

    for epoch in range(1, RDL_EPOCHS + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in tr_loader:
            batch = batch.to(DEVICE)
            logits = model(batch, entity_table)
            labels = batch[entity_table].y.float().view(-1)[: logits.shape[0]]
            if WEIGHTED_LOSS:
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), RDL_GRAD_CLIP)
            opt.step()

            running += float(loss.detach())
            steps += 1
            if steps >= RDL_MAX_STEPS:
                break

        model.eval()
        scores = []
        with torch.no_grad():
            for vb in vl_loader:
                vb = vb.to(DEVICE)
                scores.append(torch.sigmoid(model(vb, entity_table)).detach().cpu().numpy())
        val_scores = np.concatenate(scores)
        val_ap = float(average_precision_score(val_labels, val_scores)) if np.unique(val_labels).size > 1 else float("nan")
        val_auc = float(roc_auc_score(val_labels, val_scores)) if np.unique(val_labels).size > 1 else float("nan")

        print(
            f"  epoch {epoch:02d} | steps={steps} | loss={running / max(steps, 1):.4f} "
            f"| val_auprc={val_ap:.4f} | val_auroc={val_auc:.4f}",
            flush=True,
        )

        if np.isfinite(val_ap) and val_ap > best_ap + 1e-8:
            best_ap = val_ap
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= RDL_PATIENCE:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    model.eval()
    final_scores = []
    with torch.no_grad():
        for vb in vl_loader:
            vb = vb.to(DEVICE)
            final_scores.append(torch.sigmoid(model(vb, entity_table)).detach().cpu().numpy())
    final_scores = np.concatenate(final_scores)

    metrics = eval_metrics(val_labels, final_scores)

    del model, opt, tr_loader, vl_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics["train_rows"] = float(len(train_table.df))
    metrics["train_pos"] = float(train_pos)
    metrics["val_rows"] = float(len(val_table.df))
    metrics["val_pos"] = float(val_labels.sum())
    metrics["pos_weight"] = float(pos_weight_scalar)
    return metrics


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["auroc", "auprc", "p@k", "lift"]
    g = df.groupby(["dataset", "weighted_loss"], as_index=False)[metric_cols].agg(["mean", "std"])
    g.columns = ["dataset", "weighted_loss"] + [f"{c}_{s}" for c, s in g.columns.tolist()[2:]]
    return g


def main() -> None:
    print("=" * 72)
    print(f"Supervised RDL 10-seed run | dataset={DATASET} | weighted_loss={WEIGHTED_LOSS}")
    print(f"device={DEVICE} | seeds={SEEDS}")
    print(f"output={PER_SEED_CSV}")
    print("=" * 72)

    task, graph_data, col_stats, events_df, event_time_col = load_dataset_assets(DATASET)

    rows: List[Dict[str, float]] = []
    done = set()

    if PER_SEED_CSV.exists():
        prev = pd.read_csv(PER_SEED_CSV)
        if not prev.empty and {"seed", "dataset", "weighted_loss", "auprc", "auroc"}.issubset(prev.columns):
            prev = prev[(prev["dataset"] == DATASET) & (prev["weighted_loss"].astype(bool) == WEIGHTED_LOSS)].copy()
            prev = prev.dropna(subset=["seed"]).copy()
            prev["seed"] = prev["seed"].astype(int)
            prev = prev.sort_values("seed").drop_duplicates(subset=["seed"], keep="last")
            rows = prev.to_dict("records")
            done = set(prev["seed"].tolist())
            if done:
                print(f"[resume] found {len(done)} completed seeds in {PER_SEED_CSV}")

    for seed in SEEDS:
        if seed in done:
            print(f"[skip] seed={seed} already completed")
            continue

        print(f"\n[seed={seed}] starting", flush=True)
        m = run_seed(seed, task, graph_data, col_stats, events_df, event_time_col)
        row = {
            "dataset": DATASET,
            "seed": int(seed),
            "weighted_loss": bool(WEIGHTED_LOSS),
            **m,
        }
        rows.append(row)
        done.add(seed)

        df = pd.DataFrame(rows).sort_values("seed").drop_duplicates(subset=["seed"], keep="last")
        df.to_csv(PER_SEED_CSV, index=False)
        print(
            f"[seed={seed}] auprc={row['auprc']:.4f} auroc={row['auroc']:.4f} p@k={row['p@k']:.4f} "
            f"lift={row['lift']:.4f} pos_weight={row['pos_weight']:.2f}",
            flush=True,
        )

    final_df = pd.DataFrame(rows).sort_values("seed").drop_duplicates(subset=["seed"], keep="last")
    final_df.to_csv(PER_SEED_CSV, index=False)
    summary_df = summarize(final_df)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\n10-SEED SUMMARY")
    print(summary_df.to_string(index=False))
    print(f"\nPer-seed: {PER_SEED_CSV}")
    print(f"Summary:  {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
