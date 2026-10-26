"""
Shared helpers that keep multi-seed ablation outputs traceable.

Every metric row records the evaluation set it was scored on (n_eval users,
n_pos anomalies), and a run refuses to merge rows from different evaluation
sets. This guards against the H&M Table 4 mix-up, where RAD rows from one
label configuration were combined with baseline rows from another.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

# Environment variables that change the unexpected-churn labels or the evaluation set.
LABEL_ENV_VARS = (
    "TARGET_ANOM_RATE",
    "TRAIN_EQ_RATE",
    "ATRISK_TAU_A",
    "ATRISK_TAU_B",
    "ATRISK_TAU_ALPHA",
    "ATRISK_TAU_MULTIPLIER",
)

_RAD_FINAL_RE = re.compile(
    r"FINAL COMPARISON \(\w+ Set.*?At-Risk Filtered Labels\).*?"
    r"\w+:\s+([\d,]+) users\s+\|\s+([\d,]+) anomalies.*?"
    r"RAD \(rule-augmented\)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)",
    flags=re.S,
)


def parse_rad_final(txt: str) -> dict:
    """Parse RAD's at-risk filtered metrics and the evaluation-set size from a RAD script's stdout."""
    m = _RAD_FINAL_RE.search(txt)
    if not m:
        raise RuntimeError("Could not parse RAD at-risk filtered metrics from output")
    return {
        "n_eval": int(m.group(1).replace(",", "")),
        "n_pos": int(m.group(2).replace(",", "")),
        "auprc": float(m.group(3)),
        "auroc": float(m.group(4)),
        "p@k": float(m.group(5)),
        "lift": float(m.group(6)),
    }


def check_same_eval_set(rows: pd.DataFrame, seed: int) -> None:
    """Raise if the models for one seed were not scored on the same evaluation set."""
    sizes = rows[["model", "n_eval", "n_pos"]].drop_duplicates(subset=["n_eval", "n_pos"])
    if len(sizes) > 1:
        raise RuntimeError(
            f"seed={seed}: models were evaluated on different sets "
            f"(n_eval/n_pos differ):\n{rows[['model', 'n_eval', 'n_pos']].to_string(index=False)}"
        )


def write_run_config(out_dir: Path, seeds: list[int]) -> Path:
    """Record the label-defining settings next to the metrics so results can be traced to their config."""
    cfg = {
        "seeds": seeds,
        "label_env": {k: os.environ.get(k) for k in LABEL_ENV_VARS},
    }
    path = Path(out_dir) / "run_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    print(f"[config] Wrote {path}")
    return path
