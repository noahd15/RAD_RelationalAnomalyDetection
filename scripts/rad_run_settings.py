"""RAD run settings shared by the 10-seed runners and the per-seed tuning launcher.

`run_env(dataset, env)` gives the env vars the runners set for every RAD arm (before the arm's
hyperparameters). Values the caller's environment sets take precedence, as before, except the
Amazon latent size, which is fixed.

`deterministic_mode(env)` reads RAD_DETERMINISTIC (0 or 1). `rad_command` then runs the RAD script
through scripts/deterministic_run.py, and `check_mode_line` checks the run's output says so.
"""

from __future__ import annotations

from pathlib import Path

DETERMINISTIC_WRAPPER = Path(__file__).resolve().parent / "deterministic_run.py"
MODE_LINE_PREFIX = "deterministic_mode: true"  # printed by deterministic_run.py

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


def deterministic_mode(env) -> bool:
    value = env.get("RAD_DETERMINISTIC", "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"RAD_DETERMINISTIC must be 0 or 1, got {value!r}")
    return value == "1"


def rad_command(python: str, script, deterministic: bool) -> list[str]:
    return [python, str(DETERMINISTIC_WRAPPER), str(script)] if deterministic else [python, str(script)]


def check_mode_line(txt: str, deterministic: bool, label: str) -> None:
    """Raise unless the run printed the deterministic-mode line exactly when it should have."""
    printed = any(line.startswith(MODE_LINE_PREFIX) for line in txt.splitlines())
    if printed != deterministic:
        raise RuntimeError(f"{label}: deterministic mode expected {deterministic}, run output says {printed}")
