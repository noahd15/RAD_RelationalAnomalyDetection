#!/usr/bin/env python3
"""
Single-process Optuna tuning for RAD on H&M user-churn.

Mirrors scripts/optuna_rad32_tune.py (Amazon), adapted for H&M's transaction-based
activity features and rule/graph cache paths. Loads the dataset, graph cache, rule
cache, and data loaders once, then runs many RAD trials in-process with pruning.
"""

import copy
import gc
import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from path_helpers import hm_data

ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)


seed_everything(42)

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results/hm/optuna_hm48"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STUDY_DB = RESULTS_DIR / "study.db"
TRIALS_CSV = RESULTS_DIR / "trials.csv"
BEST_JSON = RESULTS_DIR / "best_trial.txt"

LATENT_DIM = int(os.environ.get("LATENT_DIM", "48"))
CHANNELS = int(os.environ.get("CHANNELS", "128"))
EPOCHS = int(os.environ.get("EPOCHS", "10"))
PATIENCE = int(os.environ.get("PATIENCE", "3"))
N_TRIALS = int(os.environ.get("N_TRIALS", "30"))
MAX_STEPS_TRAIN = int(os.environ.get("MAX_STEPS_TRAIN", "2000"))
TUNE_METRIC = os.environ.get("TUNE_METRIC", "auprc").strip().lower()
if TUNE_METRIC in {"ap", "average_precision"}:
    TUNE_METRIC = "auprc"
elif TUNE_METRIC in {"auroc", "roc_auc"}:
    TUNE_METRIC = "roc_auc"
elif TUNE_METRIC != "auprc":
    raise ValueError(f"Unsupported TUNE_METRIC={TUNE_METRIC}")

print("=" * 80, flush=True)
print("OPTUNA RAD TUNING: H&M User-Churn at-risk-filtered dataset", flush=True)
print("=" * 80, flush=True)

_cuda_index = int(os.environ.get("CUDA_DEVICE_INDEX", "0"))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this tuner.")
if torch.cuda.device_count() <= _cuda_index:
    _cuda_index = 0
device = torch.device(f"cuda:{_cuda_index}")
torch.cuda.set_device(_cuda_index)
print(f"Device: {torch.cuda.get_device_name(_cuda_index)} (cuda:{_cuda_index})", flush=True)
print(
    f"Tuning config: latent_dim={LATENT_DIM}, metric={TUNE_METRIC}, trials={N_TRIALS}, epochs={EPOCHS}",
    flush=True,
)


def _sanitize_graph(data):
    for ntype in data.node_types:
        node = data[ntype]
        if hasattr(node, "tf") and node.tf is not None and hasattr(node.tf, "feat_dict"):
            for key, t in node.tf.feat_dict.items():
                if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                    node.tf.feat_dict[key] = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return data


class GloveTextEmbedding:
    def __init__(self, device: str = "cpu", encode_batch_size: int = 64):
        self.encode_batch_size = int(encode_batch_size)
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", device=device)

    def __call__(self, sentences):
        return torch.from_numpy(self.model.encode(
            sentences,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ))


class RADAutoencoder(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict,
        num_layers: int,
        channels: int,
        aggr: str,
        norm: str,
        rule_dim: int = 0,
        latent_dim: int = 48,
        decoder_num_layers: int | None = None,
        decoder_head_layers: int = 2,
    ):
        super().__init__()
        if decoder_num_layers is None:
            decoder_num_layers = num_layers
        if latent_dim >= channels:
            raise ValueError(f"latent_dim ({latent_dim}) must be < channels ({channels})")

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]], channels=channels)
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.to_latent = nn.Linear(channels, latent_dim)
        self.decoder_gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=latent_dim,
            aggr=aggr,
            num_layers=decoder_num_layers,
        )
        self.from_latent = nn.Linear(latent_dim, channels)
        self.feat_decoder = MLP(
            channels,
            hidden_channels=channels,
            out_channels=channels,
            norm=norm,
            num_layers=decoder_head_layers,
        )
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
        pre = {k: v.detach() for k, v in x.items()}
        z = self.gnn(
            x,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        z_latent = {k: self.to_latent(v) for k, v in z.items()}
        return pre, z_latent, seed_time.size(0)

    def forward(self, batch, entity_table, rule_table=None):
        pre, z_latent, n = self._encode(batch, entity_table)
        z_dec_in = {k: v for k, v in z_latent.items()}
        if rule_table is not None and self.rule_proj is not None:
            node_ids = batch[entity_table].n_id
            z_dec_in[entity_table] = z_dec_in[entity_table] + self.rule_proj(
                rule_table[node_ids].to(z_dec_in[entity_table].device)
            )
        z_dec = self.decoder_gnn(
            z_dec_in,
            batch.edge_index_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
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
            pos_scores = self.score_edges(z, et, ei)
            neg_scores = self.score_edges(z, et, neg_ei)
            losses.append(0.5 * (
                F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
                + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            ))
        return torch.stack(losses).mean() if losses else next(iter(z.values())).new_zeros(())

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


def _activity_features_for_targets(target_df: pd.DataFrame, txn_events: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    q = target_df[["customer_id", "timestamp"]].copy()
    q["customer_id"] = q["customer_id"].astype(np.int64)
    q["timestamp"] = pd.to_datetime(q["timestamp"])
    q["_row_id"] = np.arange(len(q), dtype=np.int64)
    q = q.sort_values(["timestamp", "customer_id", "_row_id"]).reset_index(drop=True)
    right = txn_events[["customer_id", "t_dat", "cum_txns"]]
    now_join = pd.merge_asof(
        q[["customer_id", "timestamp", "_row_id"]].sort_values(["timestamp", "customer_id"]),
        right.sort_values(["t_dat", "customer_id"]),
        left_on="timestamp", right_on="t_dat",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    q["txns_before_now"] = now_join["cum_txns"].fillna(0).astype(np.int64)
    q["window_start"] = q["timestamp"] - pd.Timedelta(days=lookback_days)
    start_join = pd.merge_asof(
        q[["customer_id", "window_start", "_row_id"]].sort_values(["window_start", "customer_id"]),
        right.sort_values(["t_dat", "customer_id"]),
        left_on="window_start", right_on="t_dat",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    _w30 = start_join["cum_txns"].fillna(0).astype(np.int64)
    q["recent_txns"] = np.maximum(0, q["txns_before_now"].values - _w30.values)
    q["window_60"] = q["timestamp"] - pd.Timedelta(days=2 * lookback_days)
    w60_join = pd.merge_asof(
        q[["customer_id", "window_60", "_row_id"]].sort_values(["window_60", "customer_id"]),
        right.sort_values(["t_dat", "customer_id"]),
        left_on="window_60", right_on="t_dat",
        by="customer_id", direction="backward", allow_exact_matches=False,
    )
    _w60 = w60_join["cum_txns"].fillna(0).astype(np.int64)
    q["prev_txns_30d"] = np.maximum(0, _w30.values - _w60.values)
    q["trend_w"] = (q["recent_txns"].values - q["prev_txns_30d"].values) / (q["prev_txns_30d"].values + 1.0)
    q["avg_activity"] = 0.5 * (q["recent_txns"].values + q["prev_txns_30d"].values)
    q = q.sort_values("_row_id").reset_index(drop=True)
    return q[["recent_txns", "prev_txns_30d", "trend_w", "avg_activity"]]


def _identify_at_risk_users_trend(target_df, activity_df, thresholds=None):
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


def train_rad(model, optimizer, loader, entity_table, rule_table_gpu, hp):
    model.train()
    stats = dict(loss=0.0, feat=0.0, struct=0.0, rank=0.0, n=0, pairs=0, steps=0)
    feat_buf, struct_buf = [], []
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
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
        rank_loss, num_pairs = pairwise_rank_loss(combined, lbl, hp["rank_max_pairs"])

        loss = feat_loss + hp["lambda_struct"] * struct_loss + hp["lambda_rank"] * rank_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
        optimizer.step()

        cnt = int(norm.sum())
        stats["loss"] += float(loss.detach()) * cnt
        stats["feat"] += float(feat_loss.detach()) * cnt
        stats["struct"] += float(struct_loss.detach()) * cnt
        stats["rank"] += float(rank_loss.detach()) * cnt
        stats["n"] += cnt
        stats["pairs"] += num_pairs
        stats["steps"] += 1
        feat_buf.append(feat_err[norm].detach().cpu())
        struct_buf.append(struct_err[norm].detach().cpu())
        if hp["max_steps_train"] and stats["steps"] >= hp["max_steps_train"]:
            break

    if stats["n"] == 0:
        raise RuntimeError("No normal training samples.")

    all_f = torch.cat(feat_buf).numpy()
    all_s = torch.cat(struct_buf).numpy()
    norm_stats = {
        "feat_mu": float(np.mean(all_f)),
        "feat_sd": float(np.std(all_f)),
        "struct_mu": float(np.mean(all_s)),
        "struct_sd": float(np.std(all_s)),
    }
    reduced = {k: (v / stats["n"] if k in {"loss", "feat", "struct", "rank"} else v) for k, v in stats.items()}
    return reduced, norm_stats


@torch.no_grad()
def score_rad(model, loader, entity_table, rule_table_gpu):
    model.eval()
    feats, structs = [], []
    for batch in loader:
        batch = batch.to(device)
        recon, z_seed, z_dec, n = model(batch, entity_table, rule_table_gpu)
        feats.append(((recon - z_seed) ** 2).mean(dim=1).cpu())
        structs.append(model.seed_struct_error(z_dec, batch, n, entity_table).cpu())
    return torch.cat(feats).numpy(), torch.cat(structs).numpy()


def combine_scores(feat_err, struct_err, norm_stats, hp):
    fz = (feat_err - norm_stats["feat_mu"]) / max(norm_stats["feat_sd"], 1e-8)
    sz = (struct_err - norm_stats["struct_mu"]) / max(norm_stats["struct_sd"], 1e-8)
    return hp["alpha_feat"] * fz + hp["alpha_struct"] * sz


print("Loading dataset and caches once...", flush=True)
rdl_dataset = get_dataset("rel-hm", download=True)
rdl_task = get_task("rel-hm", "user-churn", download=True)
rdl_db = rdl_dataset.get_db()
rdl_train_table = rdl_task.get_table("train")
rdl_val_table = rdl_task.get_table("val")
rdl_test_table = rdl_task.get_table("test")
_rdl_orig_val_table = copy.deepcopy(rdl_val_table)

txn_events = rdl_db.table_dict["transactions"].df[["customer_id", "t_dat"]].dropna().copy()
txn_events["customer_id"] = txn_events["customer_id"].astype(np.int64)
txn_events["t_dat"] = pd.to_datetime(txn_events["t_dat"])
txn_events = txn_events.sort_values(["customer_id", "t_dat"]).reset_index(drop=True)
txn_events["cum_txns"] = txn_events.groupby("customer_id").cumcount() + 1

rdl_activity_train = _activity_features_for_targets(rdl_train_table.df, txn_events, lookback_days=30)
rdl_activity_val = _activity_features_for_targets(rdl_val_table.df, txn_events, lookback_days=30)
rdl_train_anomaly_labels, rdl_train_keep, rdl_at_risk_thresholds = _identify_at_risk_users_trend(
    rdl_train_table.df, rdl_activity_train)
rdl_val_anomaly_labels, rdl_val_keep, _ = _identify_at_risk_users_trend(
    rdl_val_table.df, rdl_activity_val, thresholds=rdl_at_risk_thresholds)

rdl_train_table.df = rdl_train_table.df.loc[rdl_train_keep].copy()
rdl_train_table.df["churn"] = rdl_train_anomaly_labels[rdl_train_keep]
rdl_val_table.df = rdl_val_table.df.loc[rdl_val_keep].copy()
rdl_val_table.df["churn"] = rdl_val_anomaly_labels[rdl_val_keep]

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
            _rng.choice(_pos_df.index.to_numpy(), size=_n_pos, replace=False),
        ]))
        rdl_train_table.df = rdl_train_table.df.loc[_keep].copy()

_val_pos = rdl_val_table.df[rdl_val_table.df["churn"] == 1]
_val_neg = rdl_val_table.df[rdl_val_table.df["churn"] == 0]
if len(_val_pos) > 0 and len(_val_neg) > 0:
    _n_pos_val = int(round((_target_anom_rate / (1.0 - _target_anom_rate)) * len(_val_neg)))
    _n_pos_val = min(_n_pos_val, len(_val_pos))
    if _n_pos_val < len(_val_pos):
        _rng_val = np.random.default_rng(42)
        _keep_val = np.sort(np.concatenate([
            _val_neg.index.to_numpy(),
            _rng_val.choice(_val_pos.index.to_numpy(), size=_n_pos_val, replace=False),
        ]))
        rdl_val_table.df = rdl_val_table.df.loc[_keep_val].copy()

print(
    f"At-risk filtered: train={len(rdl_train_table.df)} ({rdl_train_table.df['churn'].mean():.4f}) | "
    f"val={len(rdl_val_table.df)} ({rdl_val_table.df['churn'].mean():.4f})",
    flush=True,
)

rdl_entity_table = rdl_task.entity_table
rdl_col_to_stype = get_stype_proposal(rdl_db)
rdl_text_embedder_cfg = TextEmbedderConfig(
    text_embedder=GloveTextEmbedding(device="cpu", encode_batch_size=64), batch_size=64)
rdl_graph_cache = hm_data("relbench_rdl_graph_cache_hm.pt")
rdl_stats_cache = hm_data("relbench_rdl_col_stats_cache_hm.pt")
if os.path.exists(rdl_graph_cache) and os.path.exists(rdl_stats_cache):
    rdl_data = _sanitize_graph(torch.load(rdl_graph_cache, weights_only=False))
    rdl_col_stats_dict = torch.load(rdl_stats_cache, weights_only=False)
    print("Graph cache loaded.", flush=True)
else:
    rdl_data, rdl_col_stats_dict = make_pkey_fkey_graph(
        rdl_db,
        col_to_stype_dict=rdl_col_to_stype,
        text_embedder_cfg=rdl_text_embedder_cfg,
        cache_dir=None,
    )
    torch.save(rdl_data, rdl_graph_cache)
    torch.save(rdl_col_stats_dict, rdl_stats_cache)
    print("Graph cache created.", flush=True)

rule_cache = hm_data("hm_customer_rule_table.pt")
if not os.path.exists(rule_cache):
    raise FileNotFoundError(
        f"Pre-computed rule table not found at {rule_cache}. Run hm_rad_vs_supervised_rdl.py first."
    )
_rt = torch.load(rule_cache, weights_only=False)
rule_table = _rt["rule_table"]
n_rules = _rt["n_rules"]
_rt_gpu = rule_table.to(device)
print(f"Rule table loaded from cache: shape={tuple(rule_table.shape)}, n_rules={n_rules}", flush=True)

print("Creating loaders once before Optuna trials...", flush=True)
loader_train_input = get_node_train_table_input(table=rdl_train_table, task=rdl_task)
loader_train = NeighborLoader(
    rdl_data,
    num_neighbors=[32, 32],
    time_attr="time",
    input_nodes=loader_train_input.nodes,
    input_time=loader_train_input.time,
    transform=loader_train_input.transform,
    batch_size=1024,
    temporal_strategy="uniform",
    shuffle=True,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
)
loader_val = NeighborLoader(
    rdl_data,
    num_neighbors=[32, 32],
    time_attr="time",
    input_nodes=get_node_train_table_input(table=rdl_val_table, task=rdl_task).nodes,
    input_time=get_node_train_table_input(table=rdl_val_table, task=rdl_task).time,
    transform=get_node_train_table_input(table=rdl_val_table, task=rdl_task).transform,
    batch_size=1024,
    temporal_strategy="uniform",
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
)
orig_input = get_node_train_table_input(table=_rdl_orig_val_table, task=rdl_task)
loader_orig_val = NeighborLoader(
    rdl_data,
    num_neighbors=[32, 32],
    time_attr="time",
    input_nodes=orig_input.nodes,
    input_time=orig_input.time,
    transform=orig_input.transform,
    batch_size=1024,
    temporal_strategy="uniform",
    shuffle=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
)
print(f"Loaders created: train={len(loader_train)} batches, val={len(loader_val)} batches", flush=True)

_val_labels_filtered = rdl_val_table.df["churn"].astype(np.int64).to_numpy()
_val_labels_original = _rdl_orig_val_table.df["churn"].astype(np.int64).to_numpy()


def save_study_snapshot(study: optuna.Study):
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df.to_csv(TRIALS_CSV, index=False)
    if study.best_trial is not None:
        with BEST_JSON.open("w", encoding="ascii") as f:
            f.write(f"best_value={study.best_value}\n")
            f.write(f"best_trial={study.best_trial.number}\n")
            f.write(f"params={study.best_trial.params}\n")
            f.write(f"user_attrs={study.best_trial.user_attrs}\n")


def objective(trial: optuna.Trial) -> float:
    hp = dict(
        channels=CHANNELS,
        latent_dim=LATENT_DIM,
        num_layers=2,
        decoder_num_layers=2,
        decoder_head_layers=2,
        aggr="sum",
        norm="batch_norm",
        lr=1e-3,
        weight_decay=1e-4,
        epochs=EPOCHS,
        patience=PATIENCE,
        max_steps_train=MAX_STEPS_TRAIN,
        grad_clip=2.0,
        lambda_struct=trial.suggest_categorical("lambda_struct", [0.1, 0.2, 0.5, 1.0, 2.0]),
        lambda_rank=trial.suggest_categorical("lambda_rank", [0.25, 0.5, 1.0, 2.0, 4.0]),
        rank_max_pairs=32768,
        neg_ratio=1.0,
        alpha_feat=trial.suggest_categorical("alpha_feat", [0.5, 1.0, 2.0]),
        alpha_struct=trial.suggest_categorical("alpha_struct", [0.5, 1.0, 2.0]),
    )

    print(
        f"\nTRIAL {trial.number:03d} | "
        f"lambda_struct={hp['lambda_struct']} lambda_rank={hp['lambda_rank']} "
        f"alpha_feat={hp['alpha_feat']} alpha_struct={hp['alpha_struct']}",
        flush=True,
    )

    seed_everything(42 + trial.number)
    model = RADAutoencoder(
        rdl_data,
        rdl_col_stats_dict,
        num_layers=hp["num_layers"],
        channels=hp["channels"],
        aggr=hp["aggr"],
        norm=hp["norm"],
        rule_dim=n_rules,
        latent_dim=hp["latent_dim"],
        decoder_num_layers=hp["decoder_num_layers"],
        decoder_head_layers=hp["decoder_head_layers"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    best_metric = -math.inf
    best_state = None
    best_norm = None
    bad = 0

    try:
        for epoch in range(1, hp["epochs"] + 1):
            tr, norm_stats = train_rad(model, optimizer, loader_train, rdl_entity_table, _rt_gpu, hp)
            vf, vs = score_rad(model, loader_val, rdl_entity_table, _rt_gpu)
            val_pred = combine_scores(vf, vs, norm_stats, hp)
            val_aupr = float(average_precision_score(_val_labels_filtered, val_pred))
            val_auroc = float(roc_auc_score(_val_labels_filtered, val_pred))
            val_target = val_aupr if TUNE_METRIC == "auprc" else val_auroc
            print(
                f"  Epoch {epoch:02d}/{hp['epochs']} | loss={tr['loss']:.4f} | feat={tr['feat']:.4f} | "
                f"struct={tr['struct']:.4f} | auprc={val_aupr:.4f} | auroc={val_auroc:.4f}",
                flush=True,
            )

            trial.report(val_target, step=epoch)
            if val_target > best_metric:
                best_metric = val_target
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_norm = dict(norm_stats)
                bad = 0
            else:
                bad += 1

            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch {epoch} with {TUNE_METRIC}={val_target:.4f}")
            if bad >= hp["patience"]:
                break

        if best_state is None or best_norm is None:
            raise RuntimeError("Trial ended without a best checkpoint.")

        model.load_state_dict(best_state)
        vf, vs = score_rad(model, loader_val, rdl_entity_table, _rt_gpu)
        val_pred = combine_scores(vf, vs, best_norm, hp)
        orig_vf, orig_vs = score_rad(model, loader_orig_val, rdl_entity_table, _rt_gpu)
        orig_pred = combine_scores(orig_vf, orig_vs, best_norm, hp)

        val_aupr_filt = float(average_precision_score(_val_labels_filtered, val_pred))
        val_auroc_filt = float(roc_auc_score(_val_labels_filtered, val_pred))
        orig_aupr = float(average_precision_score(_val_labels_original, orig_pred))
        orig_auroc = float(roc_auc_score(_val_labels_original, orig_pred))

        trial.set_user_attr("auprc_filtered", val_aupr_filt)
        trial.set_user_attr("auroc_filtered", val_auroc_filt)
        trial.set_user_attr("auprc_original", orig_aupr)
        trial.set_user_attr("auroc_original", orig_auroc)
        trial.set_user_attr("latent_dim", LATENT_DIM)

        print(
            f"  Final | filtered auprc={val_aupr_filt:.4f} auroc={val_auroc_filt:.4f} | "
            f"original auprc={orig_aupr:.4f} auroc={orig_auroc:.4f}",
            flush=True,
        )
        return best_metric
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def after_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    save_study_snapshot(study)
    print(
        f"TRIAL {trial.number:03d} DONE | state={trial.state.name} | value={trial.value}",
        flush=True,
    )
    if study.best_trial is not None:
        print(
            f"BEST SO FAR | trial={study.best_trial.number} | value={study.best_value:.4f} | "
            f"params={study.best_trial.params}",
            flush=True,
        )


study = optuna.create_study(
    study_name=os.environ.get("STUDY_NAME", "hm48_optuna"),
    storage=f"sqlite:///{STUDY_DB}",
    direction="maximize",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2, interval_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42),
)

study.optimize(objective, n_trials=N_TRIALS, callbacks=[after_trial], gc_after_trial=True)
save_study_snapshot(study)

print("=" * 80, flush=True)
print("OPTUNA COMPLETE", flush=True)
print("=" * 80, flush=True)
print(f"Best trial: {study.best_trial.number}", flush=True)
print(f"Best value ({TUNE_METRIC}): {study.best_value:.4f}", flush=True)
print(f"Best params: {study.best_trial.params}", flush=True)
print(f"Trials CSV: {TRIALS_CSV}", flush=True)
print(f"Study DB: {STUDY_DB}", flush=True)
