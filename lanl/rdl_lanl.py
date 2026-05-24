#!/usr/bin/env python3
"""
rdl_lanl.py
-----------
Supervised RDL (Relational Deep Learning) baseline for LANL red-team detection.

Architecture:
  - Linear projection per node type  →  shared channels
  - HeteroGraphSAGE (relbench backbone, 2 layers)
  - MLP classification head on auth nodes
  - BCE loss (supervised, with pos_weight for class imbalance)

Causal constraint:
  Uses build_causal_edge_index_dict (identical to the existing GAE models) so
  message passing is restricted to the same causal edge set.  auth.time is
  currently all-zero in the stored graph, which effectively drops
  proc/flows/dns→auth edges and keeps only static user/computer→auth edges —
  exactly what the GAE models see.

Entity IDs masked:
  source_user_id, destination_user_id, source_computer_id,
  destination_computer_id are stripped from auth.x so the model cannot
  memorise actor identity from training positives.
"""

import copy
import os
import random
import json
import time as time_module
from pathlib import Path
from typing import Dict, Optional

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from relbench.modeling.nn import HeteroGraphSAGE

# ── Config ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
GRAPH_PATH   = str(_HERE / "data" / "sampled_hetero_graph.pt")
CHANNELS     = 128
NUM_LAYERS   = 2
EPOCHS       = 150
LR           = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE     = 20


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


_seeds_env = os.environ.get("SEEDS", "").strip()
if _seeds_env:
    SEEDS = [int(s.strip()) for s in _seeds_env.split(",") if s.strip()]
else:
    _start_seed = int(os.environ.get("START_SEED", "0"))
    _n_seeds = int(os.environ.get("N_SEEDS", "10"))
    SEEDS = list(range(_start_seed, _start_seed + _n_seeds))

USE_POS_WEIGHT = _parse_bool_env("RDL_USE_POS_WEIGHT", True)
RESULTS_DIR = os.environ.get("RDL_RESULTS_DIR", "results")
RAW_RESULTS_DIR = os.environ.get("RDL_RAW_RESULTS_DIR", os.path.join(RESULTS_DIR, "lanl_rdl_raw"))
PARTIAL_AUROC_MAX_FPR = float(os.environ.get("RDL_PARTIAL_AUROC_MAX_FPR", "0.1"))
LOSS_TAG = "weighted" if USE_POS_WEIGHT else "unweighted"
BY_SEED_CSV = os.path.join(RESULTS_DIR, f"lanl_rdl_{LOSS_TAG}_by_seed.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, f"lanl_rdl_{LOSS_TAG}_summary_mean_std.csv")

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Fix graph time tensors from duckdb ────────────────────────────────────────
DB_PATH = str(_HERE / "data" / "sample.duckdb")

def fix_graph_times(g, db_path: str):
    """
    auth.time and event-node times were not populated correctly when the graph
    was saved (auth.time = all-zero; proc/flows/dns.time = large wrong values).
    Reconstruct them from duckdb so build_causal_edge_index_dict works properly.
    All times are in the same relative-seconds unit as auth.x[:,0].
    """
    # auth: time is already in auth.x[:,0] (the 'time' feature column)
    fc = list(g["auth"].feature_cols)
    g["auth"].time = g["auth"].x[:, fc.index("time")].long()
    print(f"[rdl] auth.time fixed: min={g['auth'].time.min().item()}, max={g['auth'].time.max().item()}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        for nt, id_col, tbl in [
            ("proc",  "proc_id",  "proc"),
            ("flows", "flow_id",  "flows"),
            ("dns",   "dns_id",   "dns"),
        ]:
            node = g[nt]
            id_tensor = getattr(node, id_col)
            rows = con.execute(
                f"SELECT {id_col}, TRY_CAST(time AS BIGINT) FROM {tbl}"
            ).fetchall()
            id_to_time = {row[0]: (row[1] if row[1] is not None else -1) for row in rows}
            times = torch.tensor(
                [id_to_time.get(int(i), -1) for i in id_tensor.tolist()],
                dtype=torch.int64,
            )
            node.time = times
            valid = (times >= 0).sum().item()
            print(f"[rdl] {nt}.time fixed: {valid}/{times.size(0)} valid, "
                  f"range=[{times[times>=0].min().item() if valid else 'N/A'}, "
                  f"{times[times>=0].max().item() if valid else 'N/A'}]")
    finally:
        con.close()
    return g


# ── Causal edge builder (copied verbatim from run_all_experiments.py) ─────────
_DISABLED_TEMPORAL_REV = {
    ("auth","to_proc","proc"), ("auth","to_flows","flows"), ("auth","to_dns","dns"),
    ("auth","to_red_team","red_team"),
    ("auth","to_source_user","user"), ("auth","to_destination_user","user"),
    ("auth","to_source_computer","computer"), ("auth","to_destination_computer","computer"),
    ("proc","to_user","user"), ("proc","to_computer","computer"),
    ("flows","to_source_computer","computer"), ("flows","to_destination_computer","computer"),
    ("dns","to_source_computer","computer"),
    ("red_team","to_user","user"), ("red_team","to_source_computer","computer"),
    ("red_team","to_destination_computer","computer"),
}
_CAUSAL_INTO_AUTH = {
    ("proc","rev_to_proc","auth"):     ("proc","auth"),
    ("flows","rev_to_flows","auth"):   ("flows","auth"),
    ("dns","rev_to_dns","auth"):       ("dns","auth"),
    ("red_team","rev_to_red_team","auth"): ("red_team","auth"),
}
_STATIC_TO_EVENT = {
    ("user","rev_to_source_user","auth"), ("user","rev_to_destination_user","auth"),
    ("computer","rev_to_source_computer","auth"), ("computer","rev_to_destination_computer","auth"),
    ("user","rev_to_user","proc"), ("computer","rev_to_computer","proc"),
    ("computer","rev_to_source_computer","flows"), ("computer","rev_to_destination_computer","flows"),
    ("computer","rev_to_source_computer","dns"),
    ("user","rev_to_user","red_team"), ("computer","rev_to_source_computer","red_team"),
    ("computer","rev_to_destination_computer","red_team"),
}


def build_causal_edge_index_dict(data, target_auth_mask=None, drop_red_team=True):
    out = {}
    for et in data.edge_types:
        src_t, _, dst_t = et
        ei = data[et].edge_index
        if drop_red_team and "red_team" in (src_t, dst_t):
            out[et] = ei[:, :0]; continue
        if et in _DISABLED_TEMPORAL_REV:
            out[et] = ei[:, :0]; continue
        if et in _CAUSAL_INTO_AUTH:
            src_name, dst_name = _CAUSAL_INTO_AUTH[et]
            src_time = data[src_name].time.to(ei.device)
            dst_time = data[dst_name].time.to(ei.device)
            keep = (src_time[ei[0]] >= 0) & (dst_time[ei[1]] >= 0) & (src_time[ei[0]] <= dst_time[ei[1]])
            if target_auth_mask is not None and dst_t == "auth":
                keep &= target_auth_mask.to(ei.device)[ei[1]]
            out[et] = ei[:, keep]; continue
        if et in _STATIC_TO_EVENT:
            keep = torch.ones(ei.size(1), dtype=torch.bool, device=ei.device)
            if target_auth_mask is not None and dst_t == "auth":
                keep &= target_auth_mask.to(ei.device)[ei[1]]
            out[et] = ei[:, keep]; continue
        out[et] = ei[:, :0]
    return out


# ── Model ─────────────────────────────────────────────────────────────────────
class LinearNodeEncoder(nn.Module):
    def __init__(self, in_dims: dict, channels: int):
        super().__init__()
        self.proj = nn.ModuleDict({
            nt: nn.Linear(d, channels) for nt, d in in_dims.items()
        })

    def forward(self, x_dict: dict) -> dict:
        return {nt: self.proj[nt](x) for nt, x in x_dict.items() if nt in self.proj}


class RDLModel(nn.Module):
    def __init__(self, in_dims: dict, node_types, edge_types,
                 channels: int = 128, num_layers: int = 2):
        super().__init__()
        self.encoder = LinearNodeEncoder(in_dims, channels)
        self.gnn = HeteroGraphSAGE(
            node_types=list(node_types),
            edge_types=list(edge_types),
            channels=channels,
            aggr="mean",
            num_layers=num_layers,
        )
        self.head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, g, edge_index_dict: dict) -> torch.Tensor:
        x_dict = {
            nt: g[nt].x.float()
            for nt in g.node_types
            if hasattr(g[nt], "x") and g[nt].x is not None
        }
        x_dict = self.encoder(x_dict)
        x_dict = self.gnn(x_dict, edge_index_dict)
        return self.head(x_dict["auth"]).view(-1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 127) -> float:
    top_idx = np.argsort(-y_score)[:k]
    return float(y_true[top_idx].mean())


def partial_roc_auc(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float) -> float:
    y_true = np.asarray(y_true).astype(np.int8).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    if not (0.0 < max_fpr < 1.0):
        return float("nan")
    if np.unique(y_true).size <= 1:
        return float("nan")
    return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))


# ── Per-seed training ─────────────────────────────────────────────────────────
def run_one_seed(g, in_dims: dict, seed: int) -> dict:
    set_seed(seed)
    dev = DEVICE

    g_dev = g.to(dev)
    auth  = g_dev["auth"]
    y_all       = auth.y.view(-1).float()
    train_mask  = auth.train_mask.view(-1).bool()
    val_mask    = auth.val_mask.view(-1).bool()
    test_mask   = auth.test_mask.view(-1).bool()
    normal_train = train_mask & (y_all < 0.5)

    y_val  = y_all[val_mask].cpu().numpy().astype(np.int8)
    y_test = y_all[test_mask].cpu().numpy().astype(np.int8)
    val_has_both = (y_val.sum() > 0) and (y_val.sum() < y_val.size)

    # Pre-compute causal edge indices (same as GAE pipeline)
    tr_ei = {k: v.to(dev) for k, v in
             build_causal_edge_index_dict(g, target_auth_mask=train_mask.cpu(), drop_red_team=True).items()}
    va_ei = {k: v.to(dev) for k, v in
             build_causal_edge_index_dict(g, target_auth_mask=val_mask.cpu(),   drop_red_team=True).items()}
    te_ei = {k: v.to(dev) for k, v in
             build_causal_edge_index_dict(g, target_auth_mask=test_mask.cpu(),  drop_red_team=True).items()}

    model = RDLModel(in_dims, g.node_types, g.edge_types, CHANNELS, NUM_LAYERS).to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_pos = float(y_all[train_mask].sum().item())
    n_neg = float(train_mask.sum().item()) - n_pos
    pos_weight_scalar = (n_neg / max(n_pos, 1.0)) if USE_POS_WEIGHT else 1.0
    pos_weight = torch.tensor([pos_weight_scalar], device=dev)

    best_val_ap  = -1.0
    best_state   = None
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        logits = model(g_dev, tr_ei)              # all auth nodes
        if USE_POS_WEIGHT:
            loss = F.binary_cross_entropy_with_logits(
                logits[train_mask], y_all[train_mask], pos_weight=pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits[train_mask], y_all[train_mask]
            )
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits  = model(g_dev, va_ei)
            val_scores  = torch.sigmoid(val_logits[val_mask]).cpu().numpy()

        val_ap = (float(average_precision_score(y_val, val_scores))
                  if val_has_both else float("nan"))

        if np.isfinite(val_ap) and val_ap > best_val_ap + 1e-8:
            best_val_ap  = val_ap
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 20 == 0:
            print(f"    epoch={epoch:4d}  loss={loss.item():.4f}  "
                  f"val_AP={val_ap:.4f}  best_AP={best_val_ap:.4f}", flush=True)

        if patience_cnt >= PATIENCE:
            print(f"    Early stop at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    model.eval()

    with torch.no_grad():
        test_scores = torch.sigmoid(model(g_dev, te_ei)[test_mask]).cpu().numpy()

    auroc = float(roc_auc_score(y_test, test_scores))
    auprc = float(average_precision_score(y_test, test_scores))
    pauc = partial_roc_auc(y_test, test_scores, PARTIAL_AUROC_MAX_FPR)
    p127  = precision_at_k(y_test, test_scores, 127)

    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_RESULTS_DIR, f"seed_{seed:03d}.npz")
    np.savez_compressed(
        raw_path,
        seed=np.array(seed, dtype=np.int64),
        y_true=y_test.astype(np.int8),
        y_score=test_scores.astype(np.float32),
        train_pos_weight=np.array(pos_weight_scalar, dtype=np.float32),
        partial_auroc_max_fpr=np.array(PARTIAL_AUROC_MAX_FPR, dtype=np.float32),
    )
    return {
        "auroc": auroc,
        "auprc": auprc,
        "pauroc": pauc,
        "p@127": p127,
        "train_pos_weight": float(pos_weight_scalar),
        "raw_path": raw_path,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[rdl] Device: {DEVICE}")
    print(f"[rdl] weighted_loss={USE_POS_WEIGHT}  seeds={SEEDS}")
    print(f"[rdl] partial_auroc_max_fpr={PARTIAL_AUROC_MAX_FPR}")
    print(f"[rdl] by-seed csv: {BY_SEED_CSV}")
    print(f"[rdl] summary csv: {SUMMARY_CSV}")
    print(f"[rdl] raw outputs: {RAW_RESULTS_DIR}")
    t0 = time_module.time()

    print(f"[rdl] Loading graph from {GRAPH_PATH} ...")
    g = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)

    # Fix time tensors (auth.time was all-zero, event times were wrong units)
    g = fix_graph_times(g, DB_PATH)

    # Mask entity ID columns — prevents actor-identity memorisation
    if hasattr(g["auth"], "feature_cols"):
        fc   = list(g["auth"].feature_cols)
        drop = [
            i for i, c in enumerate(fc)
            if str(c).strip().lower() in ("auth_id",)
            or "user_id"     in str(c).lower()
            or "computer_id" in str(c).lower()
        ]
        keep = [i for i in range(len(fc)) if i not in set(drop)]
        if drop:
            g["auth"].x = g["auth"].x[:, keep]
            g["auth"].feature_cols = [fc[i] for i in keep]
            print(f"[rdl] Masked entity ID cols: {[fc[i] for i in drop]}")
            print(f"[rdl] Kept behavioral cols:  {[fc[i] for i in keep]}")

    auth = g["auth"]
    print(f"[rdl] auth nodes={auth.num_nodes}  x_dim={auth.x.size(1)}")
    print(f"[rdl] train={auth.train_mask.sum().item()}  "
          f"val={auth.val_mask.sum().item()}  "
          f"test={auth.test_mask.sum().item()}")
    print(f"[rdl] positives: train={int(auth.y[auth.train_mask].sum())}  "
          f"val={int(auth.y[auth.val_mask].sum())}  "
          f"test={int(auth.y[auth.test_mask].sum())}")

    in_dims = {
        nt: int(g[nt].x.size(1))
        for nt in g.node_types
        if hasattr(g[nt], "x") and g[nt].x is not None
    }
    print(f"[rdl] Node feature dims: {in_dims}")
    print(f"[rdl] Running {len(SEEDS)} seeds ...\n")

    all_results = []
    done_seeds = set()

    if os.path.exists(BY_SEED_CSV):
        try:
            prev = pd.read_csv(BY_SEED_CSV)
            need_cols = {"seed", "auroc", "auprc", "pauroc", "p@127"}
            if need_cols.issubset(set(prev.columns)):
                prev = prev.dropna(subset=["seed"]).copy()
                prev["seed"] = prev["seed"].astype(int)
                prev = prev.sort_values(["seed"]).drop_duplicates(subset=["seed"], keep="last")
                all_results = prev.to_dict("records")
                done_seeds = set(prev["seed"].astype(int).tolist())
                if done_seeds:
                    print(f"[resume] loaded {len(done_seeds)} completed seeds from {BY_SEED_CSV}")
        except Exception as e:
            print(f"[resume] failed to read {BY_SEED_CSV}: {e}")

    for seed in SEEDS:
        if seed in done_seeds:
            print(f"[skip] seed={seed} already completed", flush=True)
            continue
        print(f"[rdl] ── Seed {seed} ──────────────────────────────────", flush=True)
        res = run_one_seed(g, in_dims, seed)
        all_results.append({"seed": int(seed), **res, "weighted_loss": bool(USE_POS_WEIGHT)})
        print(f"  → AUROC={res['auroc']:.4f}  AUPRC={res['auprc']:.4f}  P@127={res['p@127']:.4f}\n",
              flush=True)

        by_seed_df = pd.DataFrame(all_results).sort_values(["seed"]).drop_duplicates(subset=["seed"], keep="last")
        by_seed_df.to_csv(BY_SEED_CSV, index=False)
        print(f"[seed {seed}] checkpoint saved -> {BY_SEED_CSV}", flush=True)

    by_seed_df = pd.DataFrame(all_results).sort_values(["seed"]).drop_duplicates(subset=["seed"], keep="last")
    by_seed_df.to_csv(BY_SEED_CSV, index=False)

    aurocs = by_seed_df["auroc"].astype(float).tolist()
    auprcs = by_seed_df["auprc"].astype(float).tolist()
    paurocs = by_seed_df["pauroc"].astype(float).tolist()
    p127s = by_seed_df["p@127"].astype(float).tolist()

    summary_df = pd.DataFrame(
        [
            {
                "weighted_loss": bool(USE_POS_WEIGHT),
                "num_seeds": int(len(by_seed_df)),
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs)),
                "auprc_mean": float(np.mean(auprcs)),
                "auprc_std": float(np.std(auprcs)),
                "pauroc_mean": float(np.mean(paurocs)),
                "pauroc_std": float(np.std(paurocs)),
                "p@127_mean": float(np.mean(p127s)),
                "p@127_std": float(np.std(p127s)),
            }
        ]
    )
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("=" * 60)
    print("RDL LANL Results (entity IDs masked, causal edges)")
    print("=" * 60)
    print(f"  weighted_loss: {USE_POS_WEIGHT}")
    print(f"  seeds_ran:      {len(by_seed_df)}")
    print(f"  AUROC:         {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  AUPRC:         {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print(f"  pAUROC@{PARTIAL_AUROC_MAX_FPR:.2f}: {np.mean(paurocs):.4f} ± {np.std(paurocs):.4f}")
    print(f"  Precision@127: {np.mean(p127s):.4f} ± {np.std(p127s):.4f}")
    print(f"\nPer-seed breakdown:")
    for _, r in by_seed_df.sort_values("seed").iterrows():
        print(f"  seed={int(r['seed'])}  AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  P@127={r['p@127']:.4f}")
    print(f"\n[done] by-seed: {BY_SEED_CSV}")
    print(f"[done] summary: {SUMMARY_CSV}")
    print(f"\nTotal wall time: {(time_module.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
