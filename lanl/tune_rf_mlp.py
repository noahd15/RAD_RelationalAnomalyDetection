#!/usr/bin/env python3
"""
Small cross-validated hyperparameter search for the RF/MLP supervised-detector rows
on LANL, mirroring scripts/rad_vs_supervised_rdl.py's RF_MLP_TUNE mode. Does not
touch the production RF used for rule mining (run_lanl_six_ablations.RF_PARAMS) --
this only searches a separate detector model and reports the best configs found.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import run_lanl_six_ablations as m

SEED = 42


def main() -> None:
    base_df = pd.read_parquet(m.PARQUET_PATH)
    base_df = base_df.sort_values(["time", "auth_id"]).reset_index(drop=True)
    base_df["auth_id"] = pd.to_numeric(base_df["auth_id"], errors="coerce")
    base_df = base_df.dropna(subset=["auth_id"]).copy()
    base_df["auth_id"] = base_df["auth_id"].astype(np.int64)

    graph = torch.load(m.GRAPH_PATH, map_location="cpu", weights_only=False)
    auth = graph["auth"]
    auth_ids = auth.auth_id.view(-1).cpu().numpy().astype(np.int64)
    g_train = auth.train_mask.view(-1).cpu().numpy().astype(bool)
    g_y = auth.y.view(-1).cpu().numpy().astype(np.int64)

    split_df = pd.DataFrame({"auth_id": auth_ids, "g_y": g_y, "g_train": g_train})
    merged = base_df.merge(split_df, on="auth_id", how="inner").sort_values("auth_id").reset_index(drop=True)

    X_df = m._prepare_rf_training_matrix(merged)
    y = merged["g_y"].to_numpy(dtype=np.int64)
    train_idx = np.where(merged["g_train"].to_numpy(dtype=bool))[0]
    X_train, y_train = X_df.iloc[train_idx], y[train_idx]

    print(f"Train rows: {len(X_train):,}  pos={int(y_train.sum()):,}  features={X_train.shape[1]}")

    print("[tune] Grid-searching RF (cv=3, scoring=average_precision)...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        param_grid={"n_estimators": [200, 400, 800], "max_depth": [10, 20, None], "min_samples_leaf": [1, 5, 10]},
        scoring="average_precision", cv=3, n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    print(f"[tune] RF best params: {rf_grid.best_params_}  cv_auprc={rf_grid.best_score_:.4f}")

    print("[tune] Grid-searching MLP (cv=3, scoring=average_precision)...")
    scaler = StandardScaler().fit(X_train)
    mlp_grid = GridSearchCV(
        MLPClassifier(activation="relu", learning_rate_init=1e-3, max_iter=200,
                      early_stopping=True, n_iter_no_change=10, random_state=SEED),
        param_grid={"hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)], "alpha": [1e-5, 1e-4, 1e-3]},
        scoring="average_precision", cv=3, n_jobs=-1,
    )
    mlp_grid.fit(scaler.transform(X_train), y_train)
    print(f"[tune] MLP best params: {mlp_grid.best_params_}  cv_auprc={mlp_grid.best_score_:.4f}")

    print("=" * 70)
    print("LANL RF/MLP TUNING SUMMARY")
    print(f"  RF  best: {rf_grid.best_params_}  cv_auprc={rf_grid.best_score_:.4f}")
    print(f"  MLP best: {mlp_grid.best_params_}  cv_auprc={mlp_grid.best_score_:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
