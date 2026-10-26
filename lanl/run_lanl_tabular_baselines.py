#!/usr/bin/env python3
from __future__ import annotations

import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT_DIR / "lanl" / "data" / "sampled_hetero_graph.pt"
RESULTS_DIR = Path(os.environ.get("LANL_BASELINES_RESULTS_DIR", str(ROOT_DIR / "lanl" / "results")))
BY_SEED_CSV = RESULTS_DIR / "lanl_tabular_baselines_by_seed.csv"
SUMMARY_CSV = RESULTS_DIR / "lanl_tabular_baselines_summary_mean_std.csv"

if str(ROOT_DIR / "DRL") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT_DIR / "DRL"))

from PrototypeNet.Model import PN

from eval_fingerprint import auth_eval_fingerprint


def _device() -> torch.device:
    raw = os.environ.get("LANL_BASELINE_DEVICE", "").strip().lower()
    if raw:
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = _device()


def _parse_seeds() -> list[int]:
    seeds_env = os.environ.get("SEEDS", "").strip()
    if seeds_env:
        return [int(part.strip()) for part in seeds_env.split(",") if part.strip()]
    start_seed = int(os.environ.get("START_SEED", "0"))
    n_seeds = int(os.environ.get("N_SEEDS", "10"))
    return list(range(start_seed, start_seed + n_seeds))


SEEDS = _parse_seeds()
DRL_EPOCHS = int(os.environ.get("DRL_EPOCHS", "10"))
SVDD_EPOCHS = int(os.environ.get("SVDD_EPOCHS", "100"))
AE_EPOCHS = int(os.environ.get("AE_EPOCHS", "10"))
AE_RANK_POOL_MODE = os.environ.get("AE_RANK_POOL_MODE", "labeled").strip().lower()
P_AT_K = int(os.environ.get("P_AT_K", "127"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = P_AT_K) -> float:
    top_idx = np.argsort(-scores)[:k]
    return float(y_true[top_idx].mean())


def eval_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int)
    scores = scores.astype(float)
    return {
        "auroc": float(roc_auc_score(y_true, scores)),
        "auprc": float(average_precision_score(y_true, scores)),
        "p@127": precision_at_k(y_true, scores, P_AT_K),
        "test_rows": int(y_true.size),
        "test_pos": int(y_true.sum()),
    }


def load_split_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    graph = torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    auth = graph["auth"]

    x = auth.x.detach().cpu().float()
    if hasattr(auth, "feature_cols"):
        fc = list(auth.feature_cols)
        keep = [
            i for i, col in enumerate(fc)
            if str(col).strip().lower() not in {"auth_id"}
            and "user_id" not in str(col).lower()
            and "computer_id" not in str(col).lower()
        ]
        x = x[:, keep]

    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).numpy().astype(np.float32, copy=False)
    y = auth.y.view(-1).detach().cpu().numpy().astype(np.int64)
    train_mask = auth.train_mask.view(-1).detach().cpu().numpy().astype(bool)
    val_mask = auth.val_mask.view(-1).detach().cpu().numpy().astype(bool)
    test_mask = auth.test_mask.view(-1).detach().cpu().numpy().astype(bool)
    print(f"[split-check] train&val={int((train_mask & val_mask).sum())} train&test={int((train_mask & test_mask).sum())} "
          f"val&test={int((val_mask & test_mask).sum())}", flush=True)
    if (train_mask & val_mask).any() or (train_mask & test_mask).any() or (val_mask & test_mask).any():
        raise RuntimeError("LANL train/val/test masks overlap")
    global EVAL_SET
    EVAL_SET = auth_eval_fingerprint(auth)

    x_train = x[train_mask]
    x_val = x[val_mask]
    x_test = x[test_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    train_normal_mask = y_train == 0
    x_mu = x_train[train_normal_mask].mean(axis=0, keepdims=True)
    x_sigma = x_train[train_normal_mask].std(axis=0, keepdims=True)
    x_train_n = (x_train - x_mu) / (x_sigma + 1e-6)
    x_val_n = (x_val - x_mu) / (x_sigma + 1e-6)
    x_test_n = (x_test - x_mu) / (x_sigma + 1e-6)
    x_norm = x_train_n[train_normal_mask]

    return x_norm, x_train_n, x_val_n, x_test_n, y_train, y_val, y_test


class DeepSVDDNet(nn.Module):
    def __init__(self, in_dim: int, rep_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, rep_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VanillaAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.dec = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def pairwise_logistic_rank_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.1,
    max_pairs: int = 32768,
) -> tuple[torch.Tensor, int]:
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return scores.new_zeros(()), 0
    total = int(pos.numel()) * int(neg.numel())
    if total <= max_pairs:
        diff = pos[:, None] - neg[None, :]
    else:
        pick = max_pairs
        pos_idx = torch.randint(0, pos.numel(), (pick,), device=scores.device)
        neg_idx = torch.randint(0, neg.numel(), (pick,), device=scores.device)
        diff = pos[pos_idx] - neg[neg_idx]
    return F.softplus(-(diff - margin)).mean(), min(total, max_pairs)


def run_iforest(x_norm: np.ndarray, x_test_n: np.ndarray, y_test: np.ndarray, seed: int) -> dict[str, float]:
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1)
    model.fit(x_norm)
    scores = -model.decision_function(x_test_n)
    return eval_metrics(y_test, scores)


def run_drl(x_norm: np.ndarray, x_test_n: np.ndarray, y_test: np.ndarray, seed: int) -> dict[str, float]:
    set_seed(seed)
    cfg = {
        "data_dim": x_norm.shape[1],
        "hidden_dim": 64,
        "en_nlayers": 3,
        "de_nlayers": 3,
        "prototype_num": 5,
        "diversity": True,
        "plearn": False,
        "input_info": True,
        "input_info_ratio": 0.1,
        "cl": True,
        "cl_ratio": 0.06,
    }
    batch_size = 8192
    model = PN(cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)

    x_train_t = torch.tensor(x_norm, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_train_t), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(DRL_EPOCHS):
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            loss = model(xb).mean()
            if cfg["input_info"]:
                h = model.encoder(xb)
                x_tilde = model.decoder(h)
                loss = loss + cfg["input_info_ratio"] * (1 - F.cosine_similarity(x_tilde, xb, dim=-1)).mean()
            if cfg["cl"]:
                h_phi = model.phi(xb)
                pick = max(2, int(h_phi.shape[0] * 0.8))
                sel = torch.randperm(h_phi.shape[0], device=h_phi.device)[:pick]
                h_phi = h_phi[sel]
                sim = h_phi @ h_phi.T
                denom = (
                    torch.sqrt((h_phi**2).sum(-1, keepdim=True))
                    @ torch.sqrt((h_phi.T**2).sum(0, keepdim=True))
                )
                sim = sim / (denom + 1e-8)
                loss = loss + cfg["cl_ratio"] * (sim - torch.eye(h_phi.shape[0], device=h_phi.device)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

    x_test_t = torch.tensor(x_test_n, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(x_test_t), batch_size=batch_size, shuffle=False)
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in test_loader:
            out = model(xb.to(DEVICE))
            if isinstance(out, tuple):
                out = out[0]
            scores.append(out.view(-1).detach().cpu().numpy())
    return eval_metrics(y_test, np.concatenate(scores))


def run_svdd(
    x_norm: np.ndarray,
    x_val_n: np.ndarray,
    x_test_n: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, float]:
    set_seed(seed)
    batch_size = 4096
    patience = 10
    eval_every = 10

    model = DeepSVDDNet(x_norm.shape[1], rep_dim=32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)

    train_loader = DataLoader(TensorDataset(torch.tensor(x_norm, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    model.eval()
    with torch.no_grad():
        z0 = torch.cat([model(xb.to(DEVICE)) for (xb,) in train_loader], dim=0)
        center = z0.mean(dim=0)
        eps = 0.1
        center[(center.abs() < eps) & (center < 0)] = -eps
        center[(center.abs() < eps) & (center >= 0)] = eps

    rng = np.random.default_rng(seed)
    hold_idx = rng.choice(len(x_norm), size=max(1, len(x_norm) // 10), replace=False)
    x_sub = np.delete(x_norm, hold_idx, axis=0)
    sub_loader = DataLoader(TensorDataset(torch.tensor(x_sub, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val_n, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test_n, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    best_ap = -1.0
    best_state = None
    bad = 0
    model.train()
    for epoch in range(SVDD_EPOCHS):
        running = 0.0
        steps = 0
        for (xb,) in sub_loader:
            xb = xb.to(DEVICE)
            z = model(xb)
            loss = torch.sum((z - center) ** 2, dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item())
            steps += 1
        if (running / max(steps, 1)) < 1e-5:
            break
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_scores = np.concatenate([
                    torch.sum((model(xb.to(DEVICE)) - center) ** 2, dim=1).detach().cpu().numpy()
                    for (xb,) in val_loader
                ])
            val_ap = float(average_precision_score(y_val, val_scores)) if np.unique(y_val).size > 1 else float("nan")
            if np.isfinite(val_ap) and val_ap > best_ap + 1e-8:
                best_ap = val_ap
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scores = np.concatenate([
            torch.sum((model(xb.to(DEVICE)) - center) ** 2, dim=1).detach().cpu().numpy()
            for (xb,) in test_loader
        ])
    return eval_metrics(y_test, test_scores)


def train_ae(
    x_norm: np.ndarray,
    x_val_n: np.ndarray,
    x_test_n: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    use_rank: bool,
    seed: int,
) -> dict[str, float]:
    set_seed(seed)
    batch_size = 10000
    patience = 10
    eval_every = 10
    rank_lambda = 1.5
    rank_margin = 0.1

    model = VanillaAE(x_norm.shape[1], latent_dim=8).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(x_norm, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val_n, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test_n, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    if AE_RANK_POOL_MODE == "none":
        x_anom_pool = np.empty((0, x_norm.shape[1]), dtype=x_norm.dtype)
    else:
        x_anom_pool = x_train_n_global[y_train == 1]
    anom_t = torch.tensor(x_anom_pool, dtype=torch.float32) if (use_rank and len(x_anom_pool) > 0) else None

    best_ap = -1.0
    best_state = None
    bad = 0
    for epoch in range(AE_EPOCHS):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(DEVICE)
            recon = model(xb)
            loss = crit(recon, xb)
            if use_rank and anom_t is not None:
                n_anom = min(len(xb), len(anom_t))
                idx = torch.randperm(len(anom_t))[:n_anom]
                xa = anom_t[idx].to(DEVICE)
                recon_a = model(xa)
                mse_n = ((recon - xb) ** 2).mean(dim=1)
                mse_a = ((recon_a - xa) ** 2).mean(dim=1)
                all_scores = torch.cat([mse_n, mse_a])
                all_labels = torch.cat([
                    torch.zeros(len(mse_n), device=DEVICE),
                    torch.ones(len(mse_a), device=DEVICE),
                ])
                rank_loss, _ = pairwise_logistic_rank_loss(all_scores, all_labels, rank_margin)
                loss = loss + rank_lambda * rank_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_scores = np.concatenate([
                    ((model(xb.to(DEVICE)) - xb.to(DEVICE)) ** 2).mean(dim=1).detach().cpu().numpy()
                    for (xb,) in val_loader
                ])
            val_ap = float(average_precision_score(y_val, val_scores)) if np.unique(y_val).size > 1 else float("nan")
            if np.isfinite(val_ap) and val_ap > best_ap + 1e-8:
                best_ap = val_ap
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scores = np.concatenate([
            ((model(xb.to(DEVICE)) - xb.to(DEVICE)) ** 2).mean(dim=1).detach().cpu().numpy()
            for (xb,) in test_loader
        ])
    return eval_metrics(y_test, test_scores)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("model", as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"),
        auprc_std=("auprc", "std"),
        p127_mean=("p@127", "mean"),
        p127_std=("p@127", "std"),
        num_seeds=("seed", "nunique"),
    )
    return grouped.sort_values(["auprc_mean", "auroc_mean"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    global x_train_n_global

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[lanl-baselines] device={DEVICE}")
    print(f"[lanl-baselines] seeds={SEEDS}")
    print(f"[lanl-baselines] by-seed csv={BY_SEED_CSV}")
    print(f"[lanl-baselines] summary csv={SUMMARY_CSV}")

    x_norm, x_train_n, x_val_n, x_test_n, y_train, y_val, y_test = load_split_data()
    x_train_n_global = x_train_n
    print(
        f"[lanl-baselines] train={len(x_train_n)} val={len(x_val_n)} test={len(x_test_n)} "
        f"| train_pos={int(y_train.sum())} val_pos={int(y_val.sum())} test_pos={int(y_test.sum())}"
    )
    print(f"[lanl-baselines] normal train pool={len(x_norm)} | p@k={P_AT_K}")

    rows: list[dict[str, float | int | str]] = []
    done_pairs: set[tuple[str, int]] = set()
    model_order = ["Isolation Forest", "DRL", "Deep SVDD", "AE", "AE+Rank"]

    if BY_SEED_CSV.exists():
        try:
            prev = pd.read_csv(BY_SEED_CSV)
            required = {"model", "seed", "auroc", "auprc", "p@127"}
            if required.issubset(prev.columns):
                prev = prev[prev["model"].isin(model_order)].dropna(subset=["model", "seed"]).copy()
                prev["seed"] = prev["seed"].astype(int)
                prev = prev.sort_values(["model", "seed"]).drop_duplicates(subset=["model", "seed"], keep="last")
                rows = prev.to_dict("records")
                done_pairs = set(zip(prev["model"].astype(str), prev["seed"].astype(int)))
                if done_pairs:
                    print(f"[resume] loaded {len(done_pairs)} completed runs from {BY_SEED_CSV}")
        except Exception as exc:
            print(f"[resume] failed to read checkpoint: {exc}")

    t0 = time.time()
    for seed in SEEDS:
        print(f"\n[seed {seed}] starting")
        if ("Isolation Forest", seed) not in done_pairs:
            set_seed(seed)
            res = run_iforest(x_norm, x_test_n, y_test, seed)
            rows.append({**EVAL_SET, "model": "Isolation Forest", "seed": seed, **res})
            done_pairs.add(("Isolation Forest", seed))
            print(f"  [done] Isolation Forest auprc={res['auprc']:.4f}")
        if ("DRL", seed) not in done_pairs:
            res = run_drl(x_norm, x_test_n, y_test, seed)
            rows.append({**EVAL_SET, "model": "DRL", "seed": seed, **res})
            done_pairs.add(("DRL", seed))
            print(f"  [done] DRL auprc={res['auprc']:.4f}")
        if ("Deep SVDD", seed) not in done_pairs:
            res = run_svdd(x_norm, x_val_n, x_test_n, y_val, y_test, seed)
            rows.append({**EVAL_SET, "model": "Deep SVDD", "seed": seed, **res})
            done_pairs.add(("Deep SVDD", seed))
            print(f"  [done] Deep SVDD auprc={res['auprc']:.4f}")
        if ("AE", seed) not in done_pairs:
            res = train_ae(x_norm, x_val_n, x_test_n, y_train, y_val, y_test, use_rank=False, seed=seed)
            rows.append({**EVAL_SET, "model": "AE", "seed": seed, **res})
            done_pairs.add(("AE", seed))
            print(f"  [done] AE auprc={res['auprc']:.4f}")
        if ("AE+Rank", seed) not in done_pairs:
            res = train_ae(x_norm, x_val_n, x_test_n, y_train, y_val, y_test, use_rank=True, seed=seed)
            rows.append({**EVAL_SET, "model": "AE+Rank", "seed": seed, **res})
            done_pairs.add(("AE+Rank", seed))
            print(f"  [done] AE+Rank auprc={res['auprc']:.4f}")

        by_seed_df = pd.DataFrame(rows).sort_values(["model", "seed"]).drop_duplicates(subset=["model", "seed"], keep="last")
        by_seed_df.to_csv(BY_SEED_CSV, index=False)
        print(f"[seed {seed}] checkpoint saved -> {BY_SEED_CSV}")

    by_seed_df = pd.DataFrame(rows).sort_values(["model", "seed"]).drop_duplicates(subset=["model", "seed"], keep="last")
    summary_df = summarize(by_seed_df)
    by_seed_df.to_csv(BY_SEED_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print("\n[done] LANL tabular baselines complete")
    print(f"[done] by-seed: {BY_SEED_CSV}")
    print(f"[done] summary: {SUMMARY_CSV}")
    print(f"[done] elapsed: {(time.time() - t0) / 60.0:.1f} min")


if __name__ == "__main__":
    x_train_n_global = np.empty((0, 0), dtype=np.float32)
    main()