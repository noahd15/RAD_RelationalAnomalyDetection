"""
Validation fit/report partition shared by the Amazon and H&M user-churn scripts.

RelBench hides the test labels for these tasks, so models are evaluated on a held-out
partition of the validation split. The RAD scripts (hm_rad_vs_supervised_rdl.py,
rad_vs_supervised_rdl.py) define that partition inline; these helpers reproduce it
exactly so the baselines and RDL runners select on the same fit rows and score the same
report rows. eval_fingerprint() lets a run check that every model scored identical rows,
not just the same number of rows.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

VAL_FIT_FRAC = float(os.environ.get("VAL_FIT_FRAC", "0.6"))


def fresh_table(table):
    """A new Table holding a copy of `table.df`.

    RelBench caches get_task() and Task.get_table() (lru_cache), so every call returns the
    same Table object. Reassigning or editing its .df changes what later get_table() calls in
    the same process return. Take a fresh_table() of every get_table() result before changing it.
    """
    from relbench.base import Table

    return Table(
        df=table.df.copy(),
        fkey_col_to_pkey_table=dict(table.fkey_col_to_pkey_table),
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


def partition_val(val_df: pd.DataFrame, seed: int, frac: float = VAL_FIT_FRAC) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the val table into (fit, report) exactly as the RAD scripts do."""
    val_df = val_df.reset_index(drop=True)
    perm = np.random.default_rng(seed).permutation(len(val_df))
    n_fit = int(round(frac * len(val_df)))
    fit_df = val_df.iloc[np.sort(perm[:n_fit])].copy()
    report_df = val_df.iloc[np.sort(perm[n_fit:])].copy()
    return fit_df, report_df


def _row_keys(df: pd.DataFrame) -> set:
    ts = pd.to_datetime(df["timestamp"]).astype("int64").to_numpy()
    return set(zip(df["customer_id"].astype(np.int64).to_numpy(), ts))


def check_disjoint(fit_df: pd.DataFrame, report_df: pd.DataFrame, tag: str) -> None:
    """Print the fit/report overlap (row index and (customer_id, timestamp)); raise if non-zero."""
    n_idx = len(set(fit_df.index) & set(report_df.index))
    n_key = len(_row_keys(fit_df) & _row_keys(report_df))
    print(f"[split-check] {tag}: fit={len(fit_df):,} rows | report={len(report_df):,} rows | "
          f"row-index intersection={n_idx} | (customer_id, timestamp) intersection={n_key}", flush=True)
    if n_idx or n_key:
        raise RuntimeError(f"{tag}: fit and report partitions overlap")


def downsample_positions(labels: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """Sorted positions to keep so positives make up `rate`; same draw as the RAD scripts."""
    labels = np.asarray(labels)
    neg = np.where(labels == 0)[0]
    pos = np.where(labels == 1)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(labels))
    n_tgt = min(int(round((rate / (1.0 - rate)) * len(neg))), len(pos))
    if n_tgt >= len(pos):
        return np.arange(len(labels))
    keep_pos = np.random.default_rng(seed).choice(pos, size=n_tgt, replace=False)
    return np.sort(np.concatenate([neg, keep_pos]))


def eval_fingerprint(df: pd.DataFrame, labels: np.ndarray) -> str:
    """Order-independent hash of the scored (customer_id, timestamp, label) rows."""
    ts = pd.to_datetime(df["timestamp"]).astype("int64").to_numpy()
    rows = sorted(zip(df["customer_id"].astype(np.int64).to_numpy().tolist(), ts.tolist(),
                      np.asarray(labels).astype(np.int64).tolist()))
    return hashlib.sha1(repr(rows).encode()).hexdigest()[:12]
