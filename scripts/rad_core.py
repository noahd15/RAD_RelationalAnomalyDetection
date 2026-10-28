"""RAD model and training loop, shared by the H&M and Amazon RAD scripts.

The RAD scripts (`hm_rad_vs_supervised_rdl.py`, `rad_vs_supervised_rdl.py`) and their per-seed
tuning mode (`rad_tune.py`) all train RAD through this module, so tuning and reporting use one
copy of the model, loss, scoring and early-stopping code.
"""
import hashlib
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import MLP
from tqdm import tqdm

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder


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


def build_model(data, col_stats_dict, hp, n_rules, device):
    return RADAutoencoder(
        data, col_stats_dict,
        num_layers=hp["num_layers"], channels=hp["channels"],
        aggr=hp["aggr"], norm=hp["norm"],
        rule_dim=n_rules, latent_dim=hp["latent_dim"],
        decoder_num_layers=hp["decoder_num_layers"],
        decoder_head_layers=hp["decoder_head_layers"],
    ).to(device)


def make_optimizer(model, hp):
    return torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])


def train_epoch(model, opt, loader, entity_table, rule_table_gpu, hp, device,
                lambda_struct: float, lambda_rank: float):
    model.train()
    S = dict(loss=0., feat=0., struct=0., rank=0., n=0, pairs=0, steps=0)
    feat_buf, struct_buf = [], []
    for batch in tqdm(loader, desc="  RAD train", leave=False):
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        recon, target, z, n = model(batch, entity_table, rule_table_gpu)
        lbl    = batch[entity_table].y.float().view(-1)
        norm   = lbl < 0.5
        if not norm.any(): continue
        feat_loss   = F.mse_loss(recon[norm], target[norm])
        struct_loss = model.structure_bce_loss(z, batch, hp["neg_ratio"])
        feat_err    = ((recon - target) ** 2).mean(dim=1)
        struct_err  = model.seed_struct_error(z, batch, n, entity_table)
        feat_norm = feat_err[norm].detach()
        struct_norm = struct_err[norm].detach()
        feat_mu = feat_norm.mean()
        feat_sd = torch.clamp(feat_norm.std(unbiased=False), min=1e-8)
        struct_mu = struct_norm.mean()
        struct_sd = torch.clamp(struct_norm.std(unbiased=False), min=1e-8)
        combined = (
            hp["alpha_feat"] * ((feat_err - feat_mu) / feat_sd)
            + hp["alpha_struct"] * ((struct_err - struct_mu) / struct_sd)
        )
        rank_loss, np_ = pairwise_rank_loss(combined, lbl, hp["rank_max_pairs"])
        loss = feat_loss + lambda_struct * struct_loss + lambda_rank * rank_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
        opt.step()
        cnt = int(norm.sum())
        S["loss"]   += float(loss.detach())        * cnt
        S["feat"]   += float(feat_loss.detach())   * cnt
        S["struct"] += float(struct_loss.detach()) * cnt
        S["rank"]   += float(rank_loss.detach())   * cnt
        S["n"] += cnt; S["pairs"] += np_; S["steps"] += 1
        feat_buf.append(feat_err[norm].detach().cpu())
        struct_buf.append(struct_err[norm].detach().cpu())
        if hp["max_steps_train"] and S["steps"] >= hp["max_steps_train"]: break
    if S["n"] == 0: raise RuntimeError("No normal training samples.")
    all_f = torch.cat(feat_buf).numpy(); all_s = torch.cat(struct_buf).numpy()
    ns_stats = dict(feat_mu=float(np.mean(all_f)), feat_sd=float(np.std(all_f)),
                    struct_mu=float(np.mean(all_s)), struct_sd=float(np.std(all_s)))
    return {k: (v / S["n"] if k in ("loss", "feat", "struct", "rank") else v) for k, v in S.items()}, ns_stats


@torch.no_grad()
def score(model, loader, entity_table, rule_table_gpu, device, max_batches=None):
    model.eval()
    feats, structs = [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches: break
        batch = batch.to(device)
        recon, target, z, n = model(batch, entity_table, rule_table_gpu)
        feats.append(((recon - target) ** 2).mean(dim=1).cpu())
        structs.append(model.seed_struct_error(z, batch, n, entity_table).cpu())
    return torch.cat(feats).numpy(), torch.cat(structs).numpy()


def combine(fe, se, ns, hp):
    fz = (fe - ns["feat_mu"]) / max(ns["feat_sd"],   1e-8)
    sz = (se - ns["struct_mu"]) / max(ns["struct_sd"], 1e-8)
    return hp["alpha_feat"] * fz + hp["alpha_struct"] * sz


def fit(model, opt, hp, loader_train, loader_val, val_labels, entity_table, rule_table_gpu, device,
        tune_metric, val_roc_auc=None, on_epoch=None):
    """Train RAD with early stopping on `val_labels` (the fit partition).

    `val_roc_auc(val_pred)`, if given, replaces sklearn's AUROC as the roc_auc target (the Amazon
    script uses rdl_task.evaluate). `on_epoch(epoch, val_target)` is called once per epoch; the
    tuning mode uses it to report to Optuna and prune.
    Returns (best_metric, best_state, best_norm).
    """
    best_metric, best_state, bad = -math.inf, None, 0
    best_norm = None
    norm_stats = dict(feat_mu=0., feat_sd=1., struct_mu=0., struct_sd=1.)

    for epoch in range(1, hp["epochs"] + 1):
        in_warmup         = epoch <= hp["warmup_epochs"]
        cur_lambda_struct = hp["lambda_struct"] * (hp["warmup_struct_scale"] if in_warmup else 1.0)
        cur_lambda_rank   = hp["lambda_rank"]   * (hp["warmup_rank_scale"]   if in_warmup else 1.0)

        tr, norm_stats = train_epoch(model, opt, loader_train, entity_table, rule_table_gpu, hp, device,
                                     cur_lambda_struct, cur_lambda_rank)
        vf, vs   = score(model, loader_val, entity_table, rule_table_gpu, device)
        val_pred = combine(vf, vs, norm_stats, hp)
        val_roc  = val_roc_auc(val_pred) if val_roc_auc is not None else None
        val_aupr = average_precision_score(val_labels, val_pred)
        val_auroc_ep = roc_auc_score(val_labels, val_pred) if np.unique(val_labels).size > 1 else float("nan")
        if tune_metric == "auprc":
            val_target = val_aupr
        else:
            val_target = val_auroc_ep if val_roc is None else val_roc
        stage = "warmup" if in_warmup else "main"
        print(f"  RAD epoch {epoch:02d}/{hp['epochs']} [{stage}] | steps={tr['steps']} | "
              f"loss={tr['loss']:.4f} | feat={tr['feat']:.4f} | struct={tr['struct']:.4f} | "
              f"rank={tr['rank']:.4f} | ls={cur_lambda_struct:.3f} | lrk={cur_lambda_rank:.3f} | "
              f"val_auprc={val_aupr:.4f} | val_auroc={val_auroc_ep:.4f}")
        if on_epoch is not None:
            on_epoch(epoch, val_target)
        if val_target > best_metric:
            best_metric = val_target
            best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_norm   = dict(norm_stats)
            bad = 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                print(f"  Early stopping at epoch {epoch}."); break

    return best_metric, best_state, best_norm


def inputs_fingerprint(df: pd.DataFrame, label_col: str = "churn") -> str:
    """Row count plus an order-independent hash of the (customer_id, timestamp, label) rows.

    Vectorized (unlike eval_split.eval_fingerprint) because it also hashes the multi-million-row
    train table. The tuning mode records it for the train and fit rows; the reported run prints it,
    so the runner can check that both trained and early-stopped on the same rows.
    """
    a = np.stack([
        df["customer_id"].astype(np.int64).to_numpy(),
        pd.to_datetime(df["timestamp"]).astype("int64").to_numpy(),
        df[label_col].astype(np.int64).to_numpy(),
    ], axis=1)
    a = np.ascontiguousarray(a[np.lexsort(a.T[::-1])])
    return f"{len(a)}:{hashlib.sha1(a.tobytes()).hexdigest()[:12]}"


def check_same_rule_table(built_path, expected_path) -> None:
    """Raise unless the rule table at `built_path` equals the one at `expected_path`."""
    built = torch.load(built_path, weights_only=False)
    expected = torch.load(expected_path, weights_only=False)
    problems = []
    if built["n_rules"] != expected["n_rules"]:
        problems.append(f"n_rules {built['n_rules']} vs {expected['n_rules']}")
    if list(built["rule_cols"]) != list(expected["rule_cols"]):
        problems.append(f"rule_cols {list(built['rule_cols'])} vs {list(expected['rule_cols'])}")
    if tuple(built["rule_table"].shape) != tuple(expected["rule_table"].shape):
        problems.append(f"shape {tuple(built['rule_table'].shape)} vs {tuple(expected['rule_table'].shape)}")
    elif not torch.equal(built["rule_table"], expected["rule_table"]):
        n_diff = int((built["rule_table"] != expected["rule_table"]).any(dim=1).sum())
        problems.append(f"values differ on {n_diff} rows")
    if problems:
        raise RuntimeError(f"Rule table {built_path} differs from {expected_path}: " + "; ".join(problems))
    print(f"[rules] rule table matches {expected_path} ({built['n_rules']} rules)")
