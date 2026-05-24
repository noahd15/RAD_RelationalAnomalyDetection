#!/usr/bin/env python3
"""
RAD-only latent dimension sweep: latent_dim in {2, 4, 6, 8, 10} with channels=128 bottleneck.
Loads trend-filtered data once, trains RAD for each latent_dim, reports metrics.
Skips supervised baseline (no re-runs per dimension).
"""

import copy
import gc
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP
from torch_geometric.seed import seed_everything
from torch_frame.config.text_embedder import TextEmbedderConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from path_helpers import amazon_data

ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)

seed_everything(42)

# ============================================================================
# SECTION 1: SETUP & DATA LOADING
# ============================================================================

print("="*80, flush=True)
print("RAD LATENT DIMENSION SWEEP: Amazon User-Churn at-risk-filtered dataset", flush=True)
print("="*80, flush=True)

_cuda_index = int(os.environ.get("CUDA_DEVICE_INDEX", "1"))
if torch.cuda.device_count() <= _cuda_index:
    # If CUDA_VISIBLE_DEVICES remaps to a single visible GPU, use index 0.
    _cuda_index = 0
device = torch.device(f"cuda:{_cuda_index}")
torch.cuda.set_device(_cuda_index)
print(f"Device: {torch.cuda.get_device_name(_cuda_index)} (cuda:{_cuda_index})", flush=True)

rdl_dataset = get_dataset("rel-amazon", download=True)
rdl_task = get_task("rel-amazon", "user-churn", download=True)
rdl_db = rdl_dataset.get_db()

print(f"Dataset loaded (Amazon user-churn)", flush=True)

rdl_train_table = rdl_task.get_table("train")
rdl_val_table = rdl_task.get_table("val")
rdl_test_table = rdl_task.get_table("test")

_rdl_orig_val_table = copy.deepcopy(rdl_val_table)
_rdl_orig_train_table = copy.deepcopy(rdl_train_table)

print(f"Loaded dataset: train={len(rdl_train_table.df)} | val={len(rdl_val_table.df)} | test={len(rdl_test_table.df)}")

# ============================================================================
# SECTION 2: ACTIVITY FEATURES & AT-RISK FILTERING
# ============================================================================

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
    q["last_review_time"] = now_join["review_time"]
    q["window_start"] = q["timestamp"] - pd.Timedelta(days=lookback_days)
    start_join = pd.merge_asof(
        q[["customer_id", "window_start", "_row_id"]].sort_values(["window_start", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="window_start", right_on="review_time",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    q["reviews_before_window"] = start_join["cum_reviews"].fillna(0).astype(np.int64)
    q["recent_reviews"] = q["reviews_before_now"] - q["reviews_before_window"]
    q["prev_window_start"] = q["timestamp"] - pd.Timedelta(days=2 * lookback_days)
    prev_join = pd.merge_asof(
        q[["customer_id", "prev_window_start", "_row_id"]].sort_values(["prev_window_start", "customer_id"]),
        right.sort_values(["review_time", "customer_id"]),
        left_on="prev_window_start", right_on="review_time",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    q["reviews_before_prev_window"] = prev_join["cum_reviews"].fillna(0).astype(np.int64)
    q["prev_reviews"] = q["reviews_before_window"] - q["reviews_before_prev_window"]
    q["trend_w"] = (q["recent_reviews"] - q["prev_reviews"]) / (q["prev_reviews"] + 1.0)
    q["avg_activity"] = 0.5 * (q["recent_reviews"] + q["prev_reviews"])
    delta = q["timestamp"] - q["last_review_time"]
    q["days_since_last_review"] = delta.dt.days.fillna(9999).astype(np.int64)
    q = q.sort_values("_row_id").reset_index(drop=True)
    return q[["recent_reviews", "prev_reviews", "trend_w", "avg_activity", "days_since_last_review"]]


def _identify_at_risk_users_trend(target_df, activity_df, thresholds=None):
    """Trend-based at-risk: trend_w(u) <= -tau(avg_activity(u))"""
    raw = target_df["churn"].astype(np.int64).to_numpy()
    if thresholds is None:
        thresholds = {"a": 0.05, "b": 10.0, "alpha": 0.02}
    a, b, alpha = float(thresholds["a"]), float(thresholds["b"]), float(thresholds["alpha"])
    trend = activity_df["trend_w"].to_numpy(dtype=np.float64)
    a_bar = np.maximum(activity_df["avg_activity"].to_numpy(dtype=np.float64), 0.0)
    tau = a + (np.log(a_bar + 1.0) / np.log(b)) * (1.0 + alpha * a_bar)
    at_risk = trend <= (-tau)
    keep_mask = ~at_risk
    anomaly_labels = raw * keep_mask.astype(np.int64)
    return anomaly_labels, keep_mask, thresholds


rdl_activity_train = _activity_features_for_targets(rdl_train_table.df, lookback_days=30)
rdl_activity_val = _activity_features_for_targets(rdl_val_table.df, lookback_days=30)

rdl_train_anomaly_labels, rdl_train_keep, rdl_at_risk_thresholds = _identify_at_risk_users_trend(
    rdl_train_table.df, rdl_activity_train)
rdl_val_anomaly_labels, rdl_val_keep, _ = _identify_at_risk_users_trend(
    rdl_val_table.df, rdl_activity_val, thresholds=rdl_at_risk_thresholds)

# Filter
rdl_train_table.df = rdl_train_table.df.loc[rdl_train_keep].copy()
rdl_train_table.df["churn"] = rdl_train_anomaly_labels[rdl_train_keep]
rdl_val_table.df = rdl_val_table.df.loc[rdl_val_keep].copy()
rdl_val_table.df["churn"] = rdl_val_anomaly_labels[rdl_val_keep]

# Downsample to 1.5% anomaly rate
_target_anom_rate = 0.015
_pos_df = rdl_train_table.df[rdl_train_table.df["churn"] == 1]
_neg_df = rdl_train_table.df[rdl_train_table.df["churn"] == 0]
if len(_pos_df) > 0 and len(_neg_df) > 0:
    _n_pos = int(round((_target_anom_rate / (1.0 - _target_anom_rate)) * len(_neg_df)))
    _n_pos = min(_n_pos, len(_pos_df))
    if _n_pos < len(_pos_df):
        _rng = np.random.default_rng(42)
        _keep = np.sort(np.concatenate([
            _neg_df.index.to_numpy(),
            _rng.choice(_pos_df.index.to_numpy(), size=_n_pos, replace=False)
        ]))
        rdl_train_table.df = rdl_train_table.df.loc[_keep].copy()

# Match val
_train_rate = float(rdl_train_table.df["churn"].mean())
_val_pos = rdl_val_table.df[rdl_val_table.df["churn"] == 1]
_val_neg = rdl_val_table.df[rdl_val_table.df["churn"] == 0]
if len(_val_pos) > 0 and len(_val_neg) > 0:
    _n_pos_val = int(round((_train_rate / (1.0 - _train_rate)) * len(_val_neg)))
    _n_pos_val = min(_n_pos_val, len(_val_pos))
    if _n_pos_val < len(_val_pos):
        _rng_val = np.random.default_rng(42)
        _keep_val = np.sort(np.concatenate([
            _val_neg.index.to_numpy(),
            _rng_val.choice(_val_pos.index.to_numpy(), size=_n_pos_val, replace=False)
        ]))
        rdl_val_table.df = rdl_val_table.df.loc[_keep_val].copy()

print(f"At-risk filtered: train={len(rdl_train_table.df)} ({rdl_train_table.df['churn'].mean():.4f}) | "
      f"val={len(rdl_val_table.df)} ({rdl_val_table.df['churn'].mean():.4f})", flush=True)

# ============================================================================
# SECTION 3: GRAPH & RULES SETUP
# ============================================================================

rdl_entity_table = rdl_task.entity_table
rdl_tune_metric = os.environ.get("TUNE_METRIC", "roc_auc").strip().lower()
if rdl_tune_metric in {"auprc", "average_precision", "ap"}:
    rdl_tune_metric = "auprc"
elif rdl_tune_metric in {"roc_auc", "auroc"}:
    rdl_tune_metric = "roc_auc"
else:
    raise ValueError(f"Unsupported TUNE_METRIC={rdl_tune_metric}. Use 'roc_auc' or 'auprc'.")
print(f"Checkpoint tuning metric: {rdl_tune_metric}", flush=True)


class GloveTextEmbedding:
    def __init__(self, device="cpu", encode_batch_size=64):
        self.encode_batch_size = int(encode_batch_size)
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", device=device)

    def __call__(self, sentences):
        return torch.from_numpy(self.model.encode(
            sentences, batch_size=self.encode_batch_size, show_progress_bar=False, convert_to_numpy=True))


rdl_col_to_stype = get_stype_proposal(rdl_db)
rdl_text_embedder_cfg = TextEmbedderConfig(
    text_embedder=GloveTextEmbedding(device="cpu", encode_batch_size=64), batch_size=64)

print("Using pre-computed rules from cache (skip re-mining).")

rdl_graph_cache = amazon_data("relbench_rdl_graph_cache.pt")
rdl_stats_cache = amazon_data("relbench_rdl_col_stats_cache.pt")

if os.path.exists(rdl_graph_cache) and os.path.exists(rdl_stats_cache):
    rdl_data = torch.load(rdl_graph_cache, weights_only=False)
    rdl_col_stats_dict = torch.load(rdl_stats_cache, weights_only=False)
    print("Graph cache loaded.")
else:
    rdl_data, rdl_col_stats_dict = make_pkey_fkey_graph(
        rdl_db, col_to_stype_dict=rdl_col_to_stype,
        text_embedder_cfg=rdl_text_embedder_cfg, cache_dir=None)
    torch.save(rdl_data, rdl_graph_cache)
    torch.save(rdl_col_stats_dict, rdl_stats_cache)
    print("Graph cache created.")

gc.collect()
torch.cuda.empty_cache()

# Load PRE-COMPUTED rule table (skip re-mining entirely)
RULE_TABLE_CACHE = amazon_data("amazon_customer_rule_table.pt")
if os.path.exists(RULE_TABLE_CACHE):
    _rt = torch.load(RULE_TABLE_CACHE, weights_only=False)
    rule_table = _rt["rule_table"]
    n_rules = _rt["n_rules"]
    print(f"Rule table loaded from cache: shape={tuple(rule_table.shape)}, n_rules={n_rules}", flush=True)
else:
    raise FileNotFoundError(f"Pre-computed rule table not found at {RULE_TABLE_CACHE}. "
                           "Run rad_vs_supervised_rdl.py first to generate rules.")

# ============================================================================
# CREATE LOADERS ONCE (before sweep loop) to avoid repeated rebuilding
# ============================================================================
print("Creating data loaders (once before sweep)...", flush=True)
_val_loaders = {}  # Store original loaders for evaluation
for _split, _table in [("val", rdl_val_table), ("test", rdl_test_table)]:
    _ti = get_node_train_table_input(table=_table, task=rdl_task)
    _val_loaders[_split] = NeighborLoader(
        rdl_data, num_neighbors=[32, 32], time_attr="time",
        input_nodes=_ti.nodes, input_time=_ti.time, transform=_ti.transform,
        batch_size=1024, temporal_strategy="uniform",
        shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True,
    )

# Also create train loader
_train_ti = get_node_train_table_input(table=rdl_train_table, task=rdl_task)
_train_loader = NeighborLoader(
    rdl_data, num_neighbors=[32, 32], time_attr="time",
    input_nodes=_train_ti.nodes, input_time=_train_ti.time, transform=_train_ti.transform,
    batch_size=1024, temporal_strategy="uniform",
    shuffle=True, num_workers=0, persistent_workers=False, pin_memory=True,
)
print(f"Loaders created: train={len(_train_loader)} batches, "
      f"val={len(_val_loaders['val'])} batches", flush=True)

# ============================================================================
# SECTION 4: RAD MODEL DEFINITION
# ============================================================================


class RADAutoencoder(torch.nn.Module):
    def __init__(self, data, col_stats_dict, num_layers, channels, aggr, norm,
                 rule_dim=0, latent_dim=48, decoder_num_layers=None, decoder_head_layers=2):
        super().__init__()
        if decoder_num_layers is None:
            decoder_num_layers = num_layers
        if latent_dim >= channels:
            raise ValueError(f"latent_dim ({latent_dim}) must be < channels ({channels})")

        self.channels = channels
        self.latent_dim = latent_dim
        self.rule_dim = rule_dim
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]], channels=channels)
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types, edge_types=data.edge_types,
            channels=channels, aggr=aggr, num_layers=num_layers,
        )
        self.to_latent = nn.Linear(channels, latent_dim)
        self.decoder_gnn = HeteroGraphSAGE(
            node_types=data.node_types, edge_types=data.edge_types,
            channels=latent_dim, aggr=aggr, num_layers=decoder_num_layers,
        )
        self.from_latent = nn.Linear(latent_dim, channels)
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
        torch.nn.init.xavier_uniform_(self.to_latent.weight)
        torch.nn.init.zeros_(self.to_latent.bias)
        torch.nn.init.xavier_uniform_(self.from_latent.weight)
        torch.nn.init.zeros_(self.from_latent.bias)
        for p in self.rel_diag.values():
            torch.nn.init.ones_(p)
        if self.rule_proj is not None:
            torch.nn.init.xavier_uniform_(self.rule_proj.weight)

    @staticmethod
    def _ek(et):
        return f"{et[0]}__{et[1]}__{et[2]}"

    def _encode(self, batch, entity_table):
        seed_time = batch[entity_table].seed_time
        x = self.encoder(batch.tf_dict)
        for nt, rt in self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict).items():
            x[nt] = x[nt] + rt
        z = self.gnn(x, batch.edge_index_dict, batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict)
        z_latent = {k: self.to_latent(v) for k, v in z.items()}
        pre = {k: v.detach() for k, v in z.items()}  # Store pre-bottleneck embeddings for reconstruction
        return pre, z_latent, seed_time.size(0)

    def forward(self, batch, entity_table, rule_table=None):
        pre, z_latent, n = self._encode(batch, entity_table)
        z_dec_in = dict(z_latent)
        if rule_table is not None and self.rule_proj is not None:
            node_ids = batch[entity_table].n_id
            z_dec_in[entity_table] = z_dec_in[entity_table] + self.rule_proj(
                rule_table[node_ids].to(z_dec_in[entity_table].device)
            )
        z_dec = self.decoder_gnn(z_dec_in, batch.edge_index_dict,
                                 batch.num_sampled_nodes_dict, batch.num_sampled_edges_dict)
        seed_dec = z_dec[entity_table][:n]
        seed_dec = self.from_latent(seed_dec)
        return self.feat_decoder(seed_dec), pre[entity_table][:n], z_dec, n

    def structure_bce_loss(self, z, batch, neg_ratio=1.0):
        losses = []
        for et in batch.edge_types:
            ei = batch[et].edge_index
            if ei.numel() == 0 or self._ek(et) not in self.rel_diag:
                continue
            ns, nd = int(z[et[0]].size(0)), int(z[et[2]].size(0))
            if ns == 0 or nd == 0:
                continue
            n_neg = max(1, int(neg_ratio * ei.size(1)))
            neg_ei = torch.stack([
                torch.randint(0, ns, (n_neg,), device=ei.device),
                torch.randint(0, nd, (n_neg,), device=ei.device),
            ])
            losses.append(0.5 * (
                F.binary_cross_entropy_with_logits(
                    self.score_edges(z, et, ei),
                    torch.ones_like(self.score_edges(z, et, ei)),
                )
                + F.binary_cross_entropy_with_logits(
                    self.score_edges(z, et, neg_ei),
                    torch.zeros_like(self.score_edges(z, et, neg_ei)),
                )
            ))
        return torch.stack(losses).mean() if losses else next(iter(z.values())).new_zeros(())

    def score_edges(self, z, et, ei):
        k = self._ek(et)
        if k not in self.rel_diag or ei.numel() == 0:
            return ei.new_zeros(ei.size(1), dtype=torch.float32)
        d = self.rel_diag[k]
        return (z[et[0]][ei[0]] * d * z[et[2]][ei[1]]).sum(dim=1)

    def seed_struct_error(self, z, batch, n_seed, entity_table):
        dev = z[entity_table].device
        es = torch.zeros(n_seed, device=dev)
        ec = torch.zeros(n_seed, device=dev)
        for et in batch.edge_types:
            ei = batch[et].edge_index
            if ei.numel() == 0 or self._ek(et) not in self.rel_diag:
                continue
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
    n = min(int(pos.numel()) * int(neg.numel()), max_pairs)
    pi = torch.randint(0, int(pos.numel()), (n,), device=scores.device)
    ni = torch.randint(0, int(neg.numel()), (n,), device=scores.device)
    return F.softplus(-(pos[pi] - neg[ni])).mean(), n


# ============================================================================
# SECTION 5: TRAINING & EVALUATION FUNCTIONS
# ============================================================================

_val_labels_filtered = rdl_val_table.df["churn"].astype(np.int64).to_numpy()
_val_labels_original = _rdl_orig_val_table.df["churn"].astype(np.int64).to_numpy()
_rt_gpu = rule_table.to(device) if rule_table is not None else None


def train_rad(model, opt, loader, entity_table, rule_table_gpu, hp):
    """Train one epoch."""
    model.train()
    S = dict(loss=0., feat=0., struct=0., rank=0., n=0, pairs=0, steps=0)
    feat_buf, struct_buf = [], []

    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)

        recon, z_seed, z_dec, n = model(batch, entity_table, rule_table_gpu)
        lbl = batch[entity_table].y.float().view(-1)
        norm = lbl < 0.5
        if not norm.any():
            continue

        feat_loss = F.mse_loss(recon[norm], z_seed[norm])
        struct_loss = model.structure_bce_loss(z_dec, batch, hp["neg_ratio"])
        feat_err = ((recon - z_seed) ** 2).mean(dim=1)
        struct_err = model.seed_struct_error(z_dec, batch, n, entity_table)
        combined = hp["alpha_feat"] * feat_err + hp["alpha_struct"] * struct_err
        rank_loss, np_ = pairwise_rank_loss(combined, lbl, hp["rank_max_pairs"])

        loss = feat_loss + hp["lambda_struct"] * struct_loss + hp["lambda_rank"] * rank_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
        opt.step()

        cnt = int(norm.sum())
        S["loss"] += float(loss.detach()) * cnt
        S["feat"] += float(feat_loss.detach()) * cnt
        S["struct"] += float(struct_loss.detach()) * cnt
        S["rank"] += float(rank_loss.detach()) * cnt
        S["n"] += cnt
        S["pairs"] += np_
        S["steps"] += 1

        feat_buf.append(feat_err[norm].detach().cpu())
        struct_buf.append(struct_err[norm].detach().cpu())

        if hp["max_steps_train"] and S["steps"] >= hp["max_steps_train"]:
            break

    if S["n"] == 0:
        raise RuntimeError("No normal training samples.")

    all_f = torch.cat(feat_buf).numpy()
    all_s = torch.cat(struct_buf).numpy()
    norm_stats = dict(
        feat_mu=float(np.mean(all_f)),
        feat_sd=float(np.std(all_f)),
        struct_mu=float(np.mean(all_s)),
        struct_sd=float(np.std(all_s)),
    )
    return {k: (v / S["n"] if k in ("loss", "feat", "struct", "rank") else v) for k, v in S.items()}, norm_stats


@torch.no_grad()
def score_rad(model, loader, entity_table, rule_table_gpu, max_batches=None):
    """Generate scores for a loader."""
    model.eval()
    feats, structs = [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = batch.to(device)
        recon, z_seed, z_dec, n = model(batch, entity_table, rule_table_gpu)
        feats.append(((recon - z_seed) ** 2).mean(dim=1).cpu())
        structs.append(model.seed_struct_error(z_dec, batch, n, entity_table).cpu())
    return torch.cat(feats).numpy(), torch.cat(structs).numpy()


def combine_scores(feat_err, struct_err, norm_stats, hp):
    """Combine feature + structure errors using normalization stats."""
    fz = (feat_err - norm_stats["feat_mu"]) / max(norm_stats["feat_sd"], 1e-8)
    sz = (struct_err - norm_stats["struct_mu"]) / max(norm_stats["struct_sd"], 1e-8)
    return hp["alpha_feat"] * fz + hp["alpha_struct"] * sz


# ============================================================================
# SECTION 6: LATENT DIMENSION SWEEP
# ============================================================================

_latent_dims_env = os.environ.get("LATENT_DIMS", "8,16,32,48,64,96")
LATENT_DIMS = [int(x.strip()) for x in _latent_dims_env.split(",") if x.strip()]
CHANNELS = 128

hp_base = dict(
    channels=CHANNELS,
    num_layers=2,
    decoder_num_layers=2,
    decoder_head_layers=2,
    aggr="sum",
    norm="batch_norm",
    lr=1e-3,
    weight_decay=1e-4,
    epochs=int(os.environ.get("EPOCHS", "10")),
    patience=3,
    max_steps_train=2000,
    grad_clip=2.0,
    lambda_struct=float(os.environ.get("LAMBDA_STRUCT", "0.5")),
    lambda_rank=float(os.environ.get("LAMBDA_RANK", "1.0")),
    rank_max_pairs=32768,
    neg_ratio=1.0,
    alpha_feat=float(os.environ.get("ALPHA_FEAT", "1.0")),
    alpha_struct=float(os.environ.get("ALPHA_STRUCT", "1.0")),
)

print(
    "Ablation config: "
    f"latent_dims={LATENT_DIMS}, "
    f"lambda_struct={hp_base['lambda_struct']}, lambda_rank={hp_base['lambda_rank']}, "
    f"alpha_feat={hp_base['alpha_feat']}, alpha_struct={hp_base['alpha_struct']}, "
    f"epochs={hp_base['epochs']}",
    flush=True,
)

results = []

for latent_dim in LATENT_DIMS:
    print(f"\n{'='*80}", flush=True)
    print(f"LATENT_DIM = {latent_dim} (channels={CHANNELS})", flush=True)
    print(f"{'='*80}", flush=True)

    # Create model (reuse pre-created loaders)
    hp = dict(hp_base)
    hp["latent_dim"] = latent_dim

    seed_everything(42)
    rad_model = RADAutoencoder(
        rdl_data, rdl_col_stats_dict,
        num_layers=hp["num_layers"],
        channels=hp["channels"],
        aggr=hp["aggr"],
        norm=hp["norm"],
        rule_dim=n_rules,
        latent_dim=hp["latent_dim"],
        decoder_num_layers=hp["decoder_num_layers"],
        decoder_head_layers=hp["decoder_head_layers"],
    ).to(device)

    rad_opt = torch.optim.AdamW(
        rad_model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    print(f"Model: params={sum(p.numel() for p in rad_model.parameters() if p.requires_grad):,}", flush=True)

    # Train
    best_metric, best_state, best_norm, bad = -math.inf, None, None, 0

    for epoch in range(1, hp["epochs"] + 1):
        tr, norm_stats = train_rad(rad_model, rad_opt, _train_loader, rdl_entity_table, _rt_gpu, hp)

        vf, vs = score_rad(rad_model, _val_loaders["val"], rdl_entity_table, _rt_gpu)
        val_pred = combine_scores(vf, vs, norm_stats, hp)

        val_aupr = average_precision_score(_val_labels_filtered, val_pred)
        val_auroc = roc_auc_score(_val_labels_filtered, val_pred) if np.unique(_val_labels_filtered).size > 1 else float("nan")

        print(f"  Epoch {epoch:02d}/{hp['epochs']} | "
              f"loss={tr['loss']:.4f} | feat={tr['feat']:.4f} | struct={tr['struct']:.4f} | "
              f"auprc={val_aupr:.4f} | auroc={val_auroc:.4f}", flush=True)

        val_target = val_aupr if rdl_tune_metric == "auprc" else val_auroc

        if val_target > best_metric:
            best_metric = val_target
            best_state = {k: v.detach().cpu().clone() for k, v in rad_model.state_dict().items()}
            best_norm = dict(norm_stats)
            bad = 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                print(f"  Early stopping at epoch {epoch}.", flush=True)
                break

    # Load best and evaluate
    rad_model.load_state_dict(best_state)
    ns = best_norm

    # Filtered labels (used for training)
    vf, vs = score_rad(rad_model, _val_loaders["val"], rdl_entity_table, _rt_gpu)
    val_pred = combine_scores(vf, vs, ns, hp)
    val_aupr_filt = average_precision_score(_val_labels_filtered, val_pred)
    val_auroc_filt = roc_auc_score(_val_labels_filtered, val_pred) if np.unique(_val_labels_filtered).size > 1 else float("nan")

    # Original labels (full dataset) — create loader once per latent_dim
    if latent_dim == LATENT_DIMS[0]:  # Only create once
        orig_ti = get_node_train_table_input(table=_rdl_orig_val_table, task=rdl_task)
        _orig_loader = NeighborLoader(
            rdl_data, num_neighbors=[32, 32], time_attr="time",
            input_nodes=orig_ti.nodes, input_time=orig_ti.time, transform=orig_ti.transform,
            batch_size=1024, temporal_strategy="uniform",
            shuffle=False, num_workers=0, persistent_workers=False, pin_memory=True,
        )
    orig_vf, orig_vs = score_rad(rad_model, _orig_loader, rdl_entity_table, _rt_gpu)
    orig_pred = combine_scores(orig_vf, orig_vs, ns, hp)
    orig_aupr = average_precision_score(_val_labels_original, orig_pred)
    orig_auroc = roc_auc_score(_val_labels_original, orig_pred) if np.unique(_val_labels_original).size > 1 else float("nan")

    result = {
        "latent_dim": latent_dim,
        "tune_metric": rdl_tune_metric,
        "lambda_struct": hp["lambda_struct"],
        "lambda_rank": hp["lambda_rank"],
        "alpha_feat": hp["alpha_feat"],
        "alpha_struct": hp["alpha_struct"],
        "auprc_filtered": val_aupr_filt,
        "auroc_filtered": val_auroc_filt,
        "auprc_original": orig_aupr,
        "auroc_original": orig_auroc,
    }
    results.append(result)

    print(f"  → Filtered:  auprc={val_aupr_filt:.4f}  auroc={val_auroc_filt:.4f}", flush=True)
    print(f"  → Original:  auprc={orig_aupr:.4f}  auroc={orig_auroc:.4f}", flush=True)

    # Cleanup
    del rad_model, rad_opt
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# SECTION 7: RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}", flush=True)
print("LATENT DIMENSION ABLATION RESULTS", flush=True)
print(f"{'='*80}", flush=True)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False), flush=True)

# Save CSV
_out_csv = os.environ.get("OUT_CSV", "rad_latent_sweep_results.csv")
results_df.to_csv(_out_csv, index=False)
print(f"\nSaved → {_out_csv}", flush=True)

print(f"\n{'='*80}", flush=True)
print("SUMMARY:", flush=True)
print(f"{'='*80}", flush=True)
for col in ["auprc_filtered", "auroc_filtered", "auprc_original", "auroc_original"]:
    best_row = results_df.loc[results_df[col].idxmax()]
    print(f"{col:20s}: best @ latent_dim={int(best_row['latent_dim']):2d}  value={best_row[col]:.4f}", flush=True)
