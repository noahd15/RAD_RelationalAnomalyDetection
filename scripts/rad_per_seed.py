"""Per-seed RAD hyperparameters, and the reuse of an earlier run's rows and rule tables.

Used by `run_rad_tuning_per_seed.py` (writes the per-seed studies) and by the two 10-seed runners
(`run_full_ablation_hm_10seed.py`, `run_full_ablation_10seed.py`) in their tuned-rerun mode:

- RAD_HPARAMS_FILE: per-seed hyperparameters, one row per (dataset, seed). The with-BCE arm uses
  the tuned lambda_struct; the no-BCE arms use lambda_struct 0 and the other three tuned values.
- REUSE_BASELINES_FROM: an earlier run folder (final_20261028). Its baseline rows and Supervised
  MLP row are reported unchanged; the rerun's MLP row must match it exactly. Each RAD arm must
  rebuild the rule table that run built for the same seed and arm.

Usage: python scripts/rad_per_seed.py combine OUT.csv STUDY_ROOT [STUDY_ROOT ...]
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

TUNED = ("lambda_struct", "lambda_rank", "alpha_feat", "alpha_struct")
HP_COLUMNS = (
    "dataset", "seed", *TUNED, "tune_metric", "best_value", "best_trial", "n_trials", "n_complete",
    "n_pruned", "rule_table_md5", "n_rules", "train_fp", "fit_fp", "n_fit", "n_fit_pos", "deterministic_mode",
    "study_dir",
)
REPORTED_METRICS = ("auroc", "auprc", "p@k", "lift")
ROW_COLUMNS = ("model", *REPORTED_METRICS, "n_eval", "n_pos", "eval_fp", "seed", "source")
MLP_MODEL = "Supervised MLP"
_INPUTS_FP_RE = re.compile(r"rad_inputs_fp: train=(\S+) fit=(\S+)")
# Hashes are read as text: an all-digit hex string would otherwise come back as a number.
_STR_COLUMNS = {"eval_fp": str, "rule_table_md5": str, "train_fp": str, "fit_fp": str}


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Hyperparameter file ─────────────────────────────────────────────────────
def study_row(study_dir: Path) -> dict:
    s = json.loads((Path(study_dir) / "best.json").read_text())
    return {
        "dataset": s["dataset"], "seed": s["seed"], **{k: s["best_params"][k] for k in TUNED},
        "tune_metric": s["tune_metric"], "best_value": s["best_value"], "best_trial": s["best_trial"],
        "n_trials": s["n_trials"], "n_complete": s["n_complete"], "n_pruned": s["n_pruned"],
        "rule_table_md5": s["rule_table_md5"], "n_rules": s["n_rules"], "train_fp": s["train_fp"],
        "fit_fp": s["fit_fp"], "n_fit": s["n_fit"], "n_fit_pos": s["n_fit_pos"],
        "deterministic_mode": s["deterministic_mode"], "study_dir": Path(study_dir).as_posix(),
    }


def combine(out_path, study_roots) -> pd.DataFrame:
    """Write the hyperparameter file from the seed*/best.json files under each study root."""
    out_path = Path(out_path)
    if out_path.exists():
        raise RuntimeError(f"{out_path} exists; refusing to overwrite it.")
    rows = [study_row(d) for root in study_roots for d in sorted(Path(root).glob("seed*"))]
    df = pd.DataFrame(rows, columns=list(HP_COLUMNS)).sort_values(["dataset", "seed"])
    if df.duplicated(["dataset", "seed"]).any():
        raise RuntimeError(f"duplicate (dataset, seed) rows:\n{df[['dataset', 'seed', 'study_dir']]}")
    df.to_csv(out_path, index=False)
    print(f"[hparams] wrote {out_path} ({len(df)} rows)")
    return df


def load_hparams(path, dataset: str, seeds) -> dict[int, dict]:
    """Per-seed hyperparameters for `dataset`; raise unless every seed has exactly one row."""
    # round_trip: pandas' default float parser can change the last digit of best_value.
    df = pd.read_csv(path, dtype=_STR_COLUMNS, float_precision="round_trip")
    missing_cols = set(HP_COLUMNS) - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"{path} is missing columns {sorted(missing_cols)}")
    df = df[df["dataset"] == dataset]
    out = {}
    for seed in seeds:
        rows = df[df["seed"] == seed]
        if len(rows) != 1:
            raise RuntimeError(f"{path}: expected one row for {dataset} seed {seed}, found {len(rows)}")
        out[int(seed)] = rows.iloc[0].to_dict()
    return out


def check_deterministic_studies(hparams: dict[int, dict], path) -> None:
    """A deterministic rerun can only reproduce studies that also ran in deterministic mode."""
    bad = sorted(s for s, hp in hparams.items() if str(hp["deterministic_mode"]) != "True")  # bool or numpy bool
    if bad:
        raise RuntimeError(f"{path}: seeds {bad} were not tuned in deterministic mode")


def variant_env(hp: dict, lambda_struct: float) -> dict[str, str]:
    """Hyperparameter env vars for one RAD arm."""
    return {
        "LAMBDA_STRUCT": str(float(lambda_struct)),
        "LAMBDA_RANK": str(float(hp["lambda_rank"])),
        "ALPHA_FEAT": str(float(hp["alpha_feat"])),
        "ALPHA_STRUCT": str(float(hp["alpha_struct"])),
    }


def check_inputs_fp(txt: str, hp: dict, label: str) -> None:
    """Raise unless the RAD run trained and early-stopped on the rows its study used."""
    m = _INPUTS_FP_RE.search(txt)
    if not m:
        raise RuntimeError(f"{label}: no rad_inputs_fp line in the RAD output")
    got = {"train_fp": m.group(1), "fit_fp": m.group(2)}
    want = {"train_fp": hp["train_fp"], "fit_fp": hp["fit_fp"]}
    if got != want:
        raise RuntimeError(f"{label}: RAD inputs {got} differ from the study's {want}")
    print(f"[inputs] {label}: train={got['train_fp']} fit={got['fit_fp']} match the study")


_RUN_HP_RE = re.compile(r"^rad_hp: (\{.*\})$", flags=re.M)


def check_run_hp(txt: str, hp: dict, lambda_struct: float, label: str) -> dict:
    """Raise unless the RAD run's printed settings are its study's base settings plus this arm's values."""
    m = _RUN_HP_RE.search(txt)
    if not m:
        raise RuntimeError(f"{label}: no rad_hp line in the RAD output")
    got = json.loads(m.group(1))
    want = json.loads((Path(hp["study_dir"]) / "best.json").read_text())["base_hp"]
    want.update({k: float(hp[k]) for k in TUNED}, lambda_struct=float(lambda_struct))
    if got != want:
        diff = {k: (got.get(k), want.get(k)) for k in sorted(set(got) | set(want)) if got.get(k) != want.get(k)}
        raise RuntimeError(f"{label}: RAD settings differ from the study's (run, expected): {diff}")
    print(f"[hp] {label}: RAD settings match the study plus lambda_struct={float(lambda_struct)}")
    return got


_BEST_FIT_RE = re.compile(r"^rad_best_fit_metric: (\w+)=(\S+)$", flags=re.M)


def check_best_fit(txt: str, hp: dict, label: str, exact: bool) -> float:
    """The rerun's best fit-partition metric; with `exact`, raise unless it equals the study's best value."""
    m = _BEST_FIT_RE.search(txt)
    if not m:
        raise RuntimeError(f"{label}: no rad_best_fit_metric line in the RAD output")
    metric, got = m.group(1), float(m.group(2))
    if metric != hp["tune_metric"]:
        raise RuntimeError(f"{label}: rerun metric {metric} differs from the study's {hp['tune_metric']}")
    want = float(hp["best_value"])
    if exact and got != want:
        raise RuntimeError(f"{label}: best fit {metric} {got!r} differs from the study's best trial {want!r}")
    print(f"[best-fit] {label}: {metric}={got!r}, study best {want!r} "
          f"({'equal' if got == want else 'not compared: not deterministic mode'})")
    return got


# ── Earlier run: rule tables, baseline rows, MLP row ────────────────────────
def find_rule_table(rules_root, run_prefix: str, seed: int, lambda_struct: float, mode: str) -> Path:
    """The customer_rule_table.pt of the one rules/<run_tag>/ folder for this seed and arm."""
    pat = re.compile(rf"{re.escape(run_prefix)}_seed{seed}_ls{re.escape(str(float(lambda_struct)))}"
                     rf"_{re.escape(mode)}_[0-9a-f]{{8}}")
    dirs = [d for d in Path(rules_root).iterdir() if d.is_dir() and pat.fullmatch(d.name)]
    if len(dirs) != 1:
        raise RuntimeError(f"{rules_root}: expected one rules folder for {run_prefix} seed {seed} "
                           f"ls{float(lambda_struct)} {mode}, found {[d.name for d in dirs]}")
    path = dirs[0] / "customer_rule_table.pt"
    if not path.is_file():
        raise RuntimeError(f"{path} is missing")
    return path


def load_reused_rows(reuse_dir, seed: int) -> tuple[pd.DataFrame, dict]:
    """Baseline rows (source baselines_script) and the Supervised MLP row for `seed`."""
    df = pd.read_csv(Path(reuse_dir) / "per_seed_metrics.csv", dtype=_STR_COLUMNS)
    df = df[df["seed"] == seed]
    base = df[df["source"] == "baselines_script"]
    mlp = df[df["model"] == MLP_MODEL]
    if base.empty or len(mlp) != 1:
        raise RuntimeError(f"{reuse_dir}: seed {seed} has {len(base)} baseline rows and "
                           f"{len(mlp)} {MLP_MODEL} rows (need >0 and 1)")
    return base[list(ROW_COLUMNS)].reset_index(drop=True), mlp.iloc[0][list(ROW_COLUMNS)].to_dict()


def check_mlp_match(reused: dict, rerun: dict, seed: int) -> None:
    """Raise unless the rerun's MLP row equals the reused one (it does not depend on RAD settings)."""
    keys = (*REPORTED_METRICS, "n_eval", "n_pos", "eval_fp")
    diff = {k: (reused[k], rerun[k]) for k in keys if reused[k] != rerun[k]}
    if diff:
        raise RuntimeError(f"seed={seed}: rerun {MLP_MODEL} differs from the reused row: {diff}")
    print(f"[mlp] seed={seed}: rerun {MLP_MODEL} matches the reused row")


def check_label_env(reuse_dir, label_env: dict) -> None:
    """Raise unless the reused run used the same label settings as this one."""
    cfg = json.loads((Path(reuse_dir) / "run_config.json").read_text())
    if cfg["label_env"] != label_env:
        raise RuntimeError(f"{reuse_dir} label_env {cfg['label_env']} differs from this run's {label_env}")


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] != "combine":
        sys.exit(__doc__)
    combine(sys.argv[2], sys.argv[3:])
