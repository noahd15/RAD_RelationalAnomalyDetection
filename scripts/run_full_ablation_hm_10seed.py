#!/usr/bin/env python3
"""
Run full H&M ablation over multiple seeds and report mean +/- std.

Models included:
- Isolation Forest
- DRL
- Deep SVDD
- AE
- AE+Rank
- Supervised RDL
- RAD (with BCE)
- RAD (no BCE)
- Supervised MLP (MLP supervised detector, scored inside the RAD (no BCE) run)

Outputs:
- results/full_ablation_hm_10seed/per_seed_metrics.csv
- results/full_ablation_hm_10seed/mean_std_metrics.csv
- results/full_ablation_hm_10seed/mean_std_metrics_pretty.csv
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from ablation_provenance import check_same_eval_set, parse_final_rows, parse_rad_final, write_run_config


ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)

BASELINES_SCRIPT = ROOT_DIR / "scripts" / "hm_user_churn_baselines.py"
RAD_SCRIPT = ROOT_DIR / "scripts" / "hm_rad_vs_supervised_rdl.py"
OUT_DIR = ROOT_DIR / "results" / os.environ.get("OUT_SUBDIR", "full_ablation_hm_10seed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(ROOT_DIR / "results" / "full_ablation_hm").mkdir(parents=True, exist_ok=True)


RAD_HP = {"LAMBDA_RANK": "2.0", "ALPHA_FEAT": "0.5", "ALPHA_STRUCT": "2.0"}
RAD_BCE_LAMBDA_STRUCT = 1.0
RAD_HP_SOURCE = (
    'scripts/optuna_hm_tune.py, study hm48_optuna_postfix (30 trials, TPE seed 42, objective: val AUPRC, latent_dim 48 fixed); best trial #24 value 0.044057 = {lambda_struct: 1.0, lambda_rank: 2.0, alpha_feat: 0.5, alpha_struct: 2.0}. Record: results/hm/optuna_hm48_postfix/{best_trial.txt,trials.csv,study.db} and results/hm_optuna_postfix_run.log in commit e8c5a5e (branch wip/main-local-changes-20261026). Caveat: the tuner scored the full RelBench val table, which includes the report partition used here.'
)
RAD_HP_NOTE = 'LATENT_DIM 48 fixed (not tuned); EPOCHS/PATIENCE/WARMUP defaults date from 6574b66 with no tuning record.'

# RF_SEED_VARIANT=1 adds a RAD (no BCE) arm fed the RF-sampled seed rules directly (no LLM),
# the validation-only comparison the LANL six-ablations runs as rf_sampled_rules.
RF_SEED_VARIANT = os.environ.get("RF_SEED_VARIANT", "0").strip() == "1"


def _run_baselines(seed: int) -> pd.DataFrame:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["SEED"] = str(seed)
    baselines_dir = OUT_DIR / "baselines" / f"seed{seed}"
    env["BASELINES_OUT_DIR"] = baselines_dir.as_posix()

    cmd = [sys.executable, str(BASELINES_SCRIPT)]
    print(f"[seed={seed}] Running H&M baselines script", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Baselines script exited {proc.returncode} (seed={seed})")

    csv_path = baselines_dir / "baselines_metrics.csv"
    if not csv_path.exists():
        raise RuntimeError(f"Baselines script did not produce expected file: {csv_path}")

    df = pd.read_csv(csv_path)
    df["seed"] = int(seed)
    return df


def _run_rad_variant(seed: int, lambda_struct: float, tune_metric: str = "auprc",
                     rf_rule_mode: str = "llm") -> tuple[pd.Series, dict]:
    run_tag = f"hm10seed_seed{seed}_ls{lambda_struct}_{rf_rule_mode}_{uuid.uuid4().hex[:8]}"
    env = os.environ.copy()
    env.update(
        {
            "SEED": str(seed),
            "RUN_TAG": run_tag,
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "1"),
            "LATENT_DIM": env.get("LATENT_DIM", "48"),
            "LAMBDA_STRUCT": str(lambda_struct),
            # Source of these values: RAD_HP_SOURCE (recorded in run_config.json).
            **RAD_HP,
            "TUNE_METRIC": tune_metric,
            "RF_RULE_MODE": rf_rule_mode,
            "RULES_OUT_DIR": (OUT_DIR / "rules" / run_tag).as_posix(),
            "EPOCHS": env.get("EPOCHS", "30"),
            "PATIENCE": env.get("PATIENCE", "5"),
            "WARMUP_EPOCHS": env.get("WARMUP_EPOCHS", "5"),
            "WARMUP_RANK_SCALE": env.get("WARMUP_RANK_SCALE", "0.25"),
            "WARMUP_STRUCT_SCALE": env.get("WARMUP_STRUCT_SCALE", "1.0"),
            "USE_CACHED_LLM_RULES": env.get("USE_CACHED_LLM_RULES", "1"),
        }
    )

    cmd = [sys.executable, str(RAD_SCRIPT)]
    print(f"[seed={seed}] Running H&M RAD variant lambda_struct={lambda_struct}", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"RAD script exited {proc.returncode} (seed={seed}, lambda_struct={lambda_struct})")

    txt = proc.stdout or ""
    parsed = parse_rad_final(txt)

    return pd.Series(
        {
            "auprc": parsed["auprc"],
            "auroc": parsed["auroc"],
            "p@k": parsed["p@k"],
            "lift": parsed["lift"],
            "n_eval": parsed["n_eval"],
            "n_pos": parsed["n_pos"],
            "eval_fp": parsed["eval_fp"],
            "seed": int(seed),
        }
    ), parse_final_rows(txt)


def _alias_models(df: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        "IForest": "Isolation Forest",
        "VanillaAE": "AE",
        "AE-Rank": "AE+Rank",
        "DeepSVDD": "Deep SVDD",
        "DRL": "DRL",
        # The baselines script's own RDL model. Kept as an internal cross-check against the
        # Supervised RDL runner (run_10seed_supervised_rdl_weighted.py); not a Table 4 row.
        "Supervised RDL": "Supervised RDL (baselines-script cross-check, not reported)",
    }
    out = df.copy()
    out["model"] = out["model"].map(lambda x: alias_map.get(x, x))
    return out


def _mean_std_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["auroc", "auprc", "p@k", "lift"]
    grp = df.groupby("model")[metrics]
    agg = pd.concat([grp.mean().add_suffix("_mean"), grp.std().add_suffix("_std")], axis=1).reset_index()
    cols = ["model"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    agg = agg[cols]

    pretty = agg.copy()
    for m in metrics:
        pretty[m] = pretty.apply(
            lambda r: f"{r[f'{m}_mean']:.4f} +/- {r[f'{m}_std']:.4f}", axis=1
        )
    pretty = pretty[["model", "auroc", "auprc", "p@k", "lift"]]

    agg = agg.sort_values(["auprc_mean", "auroc_mean"], ascending=[False, False]).reset_index(drop=True)
    pretty = pretty.set_index("model").loc[agg["model"]].reset_index()
    return agg, pretty


def main() -> None:
    n_seeds = int(os.environ.get("N_SEEDS", "10"))
    start_seed = int(os.environ.get("START_SEED", "0"))
    seeds = list(range(start_seed, start_seed + n_seeds))

    print("=" * 90)
    print(f"FULL H&M ABLATION OVER SEEDS | n_seeds={n_seeds} | seeds={seeds}")
    print("=" * 90)

    write_run_config(OUT_DIR, seeds, extra={
        "rad_hyperparameters": {**RAD_HP, "BCE_arm_LAMBDA_STRUCT": RAD_BCE_LAMBDA_STRUCT},
        "rad_hyperparameter_source": RAD_HP_SOURCE,
        "rad_hyperparameter_note": RAD_HP_NOTE,
        "rf_seed_variant": RF_SEED_VARIANT,
        "rules_out_dir": (OUT_DIR / "rules").as_posix(),
        "use_cached_llm_rules": os.environ.get("USE_CACHED_LLM_RULES", "1"),
        "hm_include_broad_rules": os.environ.get("HM_INCLUDE_BROAD_RULES", "1"),
    })
    all_rows: list[dict] = []
    for seed in seeds:
        baselines = _run_baselines(seed)
        baselines = _alias_models(baselines)
        baselines["source"] = "baselines_script"

        rad_bce, _              = _run_rad_variant(seed=seed, lambda_struct=RAD_BCE_LAMBDA_STRUCT)
        rad_no_bce, final_no_bce = _run_rad_variant(seed=seed, lambda_struct=0.0)
        mlp = final_no_bce["MLP (supervised detector)"]
        rf_seed_rows = []
        if RF_SEED_VARIANT:
            rad_rf, _ = _run_rad_variant(seed=seed, lambda_struct=0.0, rf_rule_mode="rf_seed")
            rf_seed_rows.append({
                "model": "RAD (RF-sampled rules, no BCE)",
                **{k: rad_rf[k] for k in ("auroc", "auprc", "p@k", "lift", "n_eval", "n_pos", "eval_fp")},
                "seed": int(seed),
                "source": "rad_vs_script",
            })

        rad_rows = pd.DataFrame(
            [
                {
                    "model": "RAD (with BCE)",
                    "auroc": float(rad_bce["auroc"]),
                    "auprc": float(rad_bce["auprc"]),
                    "p@k":   float(rad_bce["p@k"]),
                    "lift":  float(rad_bce["lift"]),
                    "n_eval": int(rad_bce["n_eval"]),
                    "n_pos": int(rad_bce["n_pos"]),
                    "eval_fp": rad_bce["eval_fp"],
                    "seed":  int(seed),
                    "source": "rad_vs_script",
                },
                {
                    "model": "RAD (no BCE)",
                    "auroc": float(rad_no_bce["auroc"]),
                    "auprc": float(rad_no_bce["auprc"]),
                    "p@k":   float(rad_no_bce["p@k"]),
                    "lift":  float(rad_no_bce["lift"]),
                    "n_eval": int(rad_no_bce["n_eval"]),
                    "n_pos": int(rad_no_bce["n_pos"]),
                    "eval_fp": rad_no_bce["eval_fp"],
                    "seed":  int(seed),
                    "source": "rad_vs_script",
                },
                {
                    "model": "Supervised MLP",
                    **{k: mlp[k] for k in ("auroc", "auprc", "p@k", "lift", "n_eval", "n_pos", "eval_fp")},
                    "seed":  int(seed),
                    "source": "rad_vs_script",
                },
            ] + rf_seed_rows
        )

        combined = pd.concat(
            [
                baselines[["model", "auroc", "auprc", "p@k", "lift", "n_eval", "n_pos", "eval_fp", "seed", "source"]],
                rad_rows,
            ],
            ignore_index=True,
        )
        check_same_eval_set(combined, seed)
        all_rows.extend(combined.to_dict(orient="records"))

    per_seed = pd.DataFrame(all_rows)
    mean_std, pretty = _mean_std_report(per_seed)

    per_seed_out = OUT_DIR / "per_seed_metrics.csv"
    mean_std_out = OUT_DIR / "mean_std_metrics.csv"
    pretty_out   = OUT_DIR / "mean_std_metrics_pretty.csv"

    per_seed.to_csv(per_seed_out, index=False)
    mean_std.to_csv(mean_std_out, index=False)
    pretty.to_csv(pretty_out, index=False)

    print(f"[done] Wrote per-seed metrics: {per_seed_out}")
    print(f"[done] Wrote mean/std metrics: {mean_std_out}")
    print(f"[done] Wrote pretty report:    {pretty_out}")
    print("\nMean +/- std summary:")
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
