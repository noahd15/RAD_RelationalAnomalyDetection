"""
Shared helpers that keep multi-seed ablation outputs traceable.

Every metric row records the evaluation set it was scored on (n_eval users,
n_pos anomalies, and eval_fp, a hash of the scored rows), and a run refuses to merge rows from different evaluation
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


_FINAL_BLOCK_RE = re.compile(
    r"FINAL COMPARISON \(\w+ Set.*?At-Risk Filtered Labels\)(.*?)(?=FINAL COMPARISON|\Z)", flags=re.S
)
_FP_RE = re.compile(r"eval_fingerprint:\s+([0-9a-f]+)")
_NUM = r"([0-9]*\.?[0-9]+|nan)"
# Rows the RAD scripts print in the at-risk filtered FINAL COMPARISON block, all scored on the same rows.
FINAL_ROW_MODELS = (
    "Supervised RDL GNN",
    "RF (supervised detector)",
    "MLP (supervised detector)",
    "RAD (rule-augmented)",
)


def parse_eval_fingerprint(txt: str) -> str | None:
    """First eval_fingerprint printed in `txt` (None for output from scripts that predate it)."""
    m = _FP_RE.search(txt)
    return m.group(1) if m else None


def parse_rad_final(txt: str) -> dict:
    """Parse RAD's at-risk filtered metrics and the evaluation-set size from a RAD script's stdout."""
    m = _RAD_FINAL_RE.search(txt)
    if not m:
        raise RuntimeError("Could not parse RAD at-risk filtered metrics from output")
    block = _FINAL_BLOCK_RE.search(txt)
    return {
        "n_eval": int(m.group(1).replace(",", "")),
        "n_pos": int(m.group(2).replace(",", "")),
        "auprc": float(m.group(3)),
        "auroc": float(m.group(4)),
        "p@k": float(m.group(5)),
        "lift": float(m.group(6)),
        "eval_fp": parse_eval_fingerprint(block.group(1)) if block else None,
    }


def parse_final_rows(txt: str) -> dict[str, dict]:
    """Parse every model row of the at-risk filtered FINAL COMPARISON block, keyed by model name."""
    rad = parse_rad_final(txt)
    block = _FINAL_BLOCK_RE.search(txt).group(1)
    out = {}
    for name in FINAL_ROW_MODELS:
        m = re.search(re.escape(name) + r"\s+" + r"\s+".join([_NUM] * 4), block)
        if not m:
            raise RuntimeError(f"Could not parse '{name}' from the FINAL COMPARISON block")
        auprc, auroc, pak, lift = (float(g) for g in m.groups())
        out[name] = {"auprc": auprc, "auroc": auroc, "p@k": pak, "lift": lift,
                     "n_eval": rad["n_eval"], "n_pos": rad["n_pos"], "eval_fp": rad["eval_fp"]}
    return out


def check_same_eval_set(rows: pd.DataFrame, seed: int) -> None:
    """Raise if the models for one seed were not scored on the same evaluation set."""
    cols = ["n_eval", "n_pos"]
    # eval_fp hashes the scored (customer_id, timestamp, label) rows, so it also catches
    # two different evaluation sets that happen to have the same size.
    if "eval_fp" in rows.columns and rows["eval_fp"].notna().any():
        cols.append("eval_fp")
    sizes = rows[["model"] + cols].drop_duplicates(subset=cols)
    if len(sizes) > 1:
        raise RuntimeError(
            f"seed={seed}: models were evaluated on different sets "
            f"({'/'.join(cols)} differ):\n{rows[['model'] + cols].to_string(index=False)}"
        )


def write_run_config(out_dir: Path, seeds: list[int], extra: dict | None = None) -> Path:
    """Record the label-defining settings next to the metrics so results can be traced to their config."""
    cfg = {
        "seeds": seeds,
        "label_env": {k: os.environ.get(k) for k in LABEL_ENV_VARS},
        **(extra or {}),
    }
    path = Path(out_dir) / "run_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    print(f"[config] Wrote {path}")
    return path
