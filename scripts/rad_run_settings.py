"""RAD run settings shared by the 10-seed runners and the per-seed tuning launcher.

`run_env(dataset, env)` gives the env vars the runners set for every RAD arm (before the arm's
hyperparameters). Values the caller's environment sets take precedence, as before, except the
Amazon latent size, which is fixed.
"""

from __future__ import annotations

RAD_SCRIPT = {"hm": "scripts/hm_rad_vs_supervised_rdl.py", "amazon": "scripts/rad_vs_supervised_rdl.py"}
RUN_PREFIX = {"hm": "hm10seed", "amazon": "amazon10seed"}
# lambda_struct of the with-BCE arm under the August hyperparameters (the final_20261028 rows).
AUGUST_BCE_LAMBDA_STRUCT = {"hm": 1.0, "amazon": 0.1}

_DEFAULTS = {
    "hm": {"LATENT_DIM": "48", "EPOCHS": "30", "PATIENCE": "5", "WARMUP_EPOCHS": "5",
           "WARMUP_RANK_SCALE": "0.25", "WARMUP_STRUCT_SCALE": "1.0"},
    "amazon": {"EPOCHS": "60", "PATIENCE": "10", "WARMUP_EPOCHS": "15",
               "WARMUP_RANK_SCALE": "0.25", "WARMUP_STRUCT_SCALE": "1.0"},
}
_FIXED = {"hm": {}, "amazon": {"LATENT_DIM": "32"}}


def run_env(dataset: str, env: dict) -> dict[str, str]:
    out = {k: env.get(k, v) for k, v in _DEFAULTS[dataset].items()}
    out.update(_FIXED[dataset])
    return out
