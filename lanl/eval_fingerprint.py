"""Evaluation-set fingerprint shared by the LANL Table 4 scripts.

Every LANL row is scored on auth.test_mask of lanl/data/sampled_hetero_graph.pt. eval_fp is an
order-independent hash of the scored (auth_id, label) rows, so rows from different scripts can be
checked to come from the same evaluation set (as eval_split.eval_fingerprint does for H&M/Amazon).
"""
import hashlib

import numpy as np


def auth_eval_fingerprint(auth) -> dict:
    mask = auth.test_mask.view(-1).cpu().numpy().astype(bool)
    ids = auth.auth_id.view(-1).cpu().numpy().astype(np.int64)[mask]
    y = auth.y.view(-1).cpu().numpy().astype(np.int64)[mask]
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("eval fingerprint: duplicate auth_id in the test rows")
    rows = sorted(zip(ids.tolist(), y.tolist()))
    out = {"eval_fp": hashlib.sha1(repr(rows).encode()).hexdigest()[:12], "n_eval": int(len(rows)), "n_pos": int(y.sum())}
    print(f"eval_fingerprint: {out['eval_fp']} n_eval={out['n_eval']} n_pos={out['n_pos']}", flush=True)
    return out
