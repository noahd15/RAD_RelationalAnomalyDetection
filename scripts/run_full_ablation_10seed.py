#!/usr/bin/env python3
"""
Run full ablation over multiple seeds and report mean +/- std.

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
- results/amazon/full_ablation_10seed/per_seed_metrics.csv
- results/amazon/full_ablation_10seed/mean_std_metrics.csv
- results/amazon/full_ablation_10seed/mean_std_metrics_pretty.csv
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import rad_per_seed
from ablation_provenance import (LABEL_ENV_VARS, check_same_eval_set, parse_final_rows, parse_rad_final,
                                 write_run_config)
from rad_run_settings import RUN_PREFIX, check_mode_line, deterministic_mode, rad_command, run_env


ROOT_DIR = Path(__file__).resolve().parents[1]
os.chdir(ROOT_DIR)

BASELINES_SCRIPT = ROOT_DIR / "scripts" / "amazon_user_churn_baselines.py"
RAD_SCRIPT = ROOT_DIR / "scripts" / "rad_vs_supervised_rdl.py"
OUT_DIR = ROOT_DIR / "results" / os.environ.get("OUT_SUBDIR", "amazon/full_ablation_10seed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


RAD_HP = {"LAMBDA_RANK": "2.0", "ALPHA_FEAT": "1.0", "ALPHA_STRUCT": "1.0"}
RAD_BCE_LAMBDA_STRUCT = 0.1
RAD_HP_SOURCE = (
    'scripts/optuna_rad32_tune.py, study rad32_optuna_postfix (30 trials, TPE seed 42, objective: val AUPRC, latent_dim 32 fixed); best trial #6 value 0.028929 = {lambda_struct: 0.1, lambda_rank: 2.0, alpha_feat: 1.0, alpha_struct: 1.0}. Record: results/amazon/optuna_rad32_postfix/{best_trial.txt,trials.csv,study.db} and results/amazon_optuna_postfix_run.log in commit e8c5a5e (branch wip/main-local-changes-20261026). Caveat: the tuner scored the full RelBench val table, which includes the report partition used here.'
)
RAD_HP_NOTE = 'LATENT_DIM 32 fixed (not tuned); EPOCHS/PATIENCE/WARMUP defaults date from 6574b66 with no tuning record.'

# Tuned rerun (docs/table4_reporting_plan.md, "RAD hyperparameters"): RAD_HPARAMS_FILE gives each
# seed's hyperparameters, and REUSE_BASELINES_FROM (the final_20261028 folder) supplies the baseline
# and Supervised MLP rows and the rule tables each RAD arm must rebuild. Set both or neither.
RAD_HPARAMS_FILE = os.environ.get("RAD_HPARAMS_FILE", "").strip()
REUSE_BASELINES_FROM = os.environ.get("REUSE_BASELINES_FROM", "").strip()
if bool(RAD_HPARAMS_FILE) != bool(REUSE_BASELINES_FROM):
    raise RuntimeError("Set RAD_HPARAMS_FILE and REUSE_BASELINES_FROM together.")
# RAD_DETERMINISTIC=1 runs each RAD arm through scripts/deterministic_run.py.
DETERMINISTIC = deterministic_mode(os.environ)

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
    print(f"[seed={seed}] Running baselines script", flush=True)
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
                     rf_rule_mode: str = "llm", hp: dict | None = None,
                     expect_rule_table: Path | None = None) -> tuple[pd.Series, dict]:
    run_tag = f"amazon10seed_seed{seed}_ls{lambda_struct}_{rf_rule_mode}_{uuid.uuid4().hex[:8]}"
    env = os.environ.copy()
    env.update(
        {
            "SEED": str(seed),
            "RUN_TAG": run_tag,
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "1"),
            # Per-seed values from RAD_HPARAMS_FILE, or the August values (RAD_HP_SOURCE).
            **(rad_per_seed.variant_env(hp, lambda_struct) if hp is not None
               else {"LAMBDA_STRUCT": str(lambda_struct), **RAD_HP}),
            "TUNE_METRIC": tune_metric,
            "RF_RULE_MODE": rf_rule_mode,
            "RULES_OUT_DIR": (OUT_DIR / "rules" / run_tag).as_posix(),
            # One raw_results file per arm (scores, labels and the feature/structure error terms).
            "SAVE_RAW_RESULTS_PATH": (OUT_DIR / "raw" / f"{run_tag}.pt").as_posix(),
            # LATENT_DIM, EPOCHS, PATIENCE, WARMUP_*: shared with the per-seed studies.
            **run_env("amazon", env),
        }
    )
    if expect_rule_table is not None:
        env["RULE_TABLE_EXPECT"] = expect_rule_table.as_posix()

    cmd = rad_command(sys.executable, RAD_SCRIPT, DETERMINISTIC)
    print(f"[seed={seed}] Running RAD variant lambda_struct={lambda_struct}", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"RAD script exited {proc.returncode} (seed={seed}, lambda_struct={lambda_struct})")

    txt = proc.stdout or ""
    label = f"seed={seed} ls{lambda_struct} {rf_rule_mode}"
    check_mode_line(txt, DETERMINISTIC, label)
    if hp is not None:
        rad_per_seed.check_inputs_fp(txt, hp, label)
        rad_per_seed.check_run_hp(txt, hp, lambda_struct, label)
        if rf_rule_mode == "llm" and lambda_struct == float(hp["lambda_struct"]):
            # The with-BCE LLM arm is the study's selected model; in deterministic mode it must
            # reach exactly the study's best value.
            rad_per_seed.check_best_fit(txt, hp, label, exact=DETERMINISTIC)
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
    agg = (
        df.groupby("model", as_index=False)[metrics]
        .agg(["mean", "std"])
    )
    agg.columns = [
        "model",
        "auroc_mean",
        "auroc_std",
        "auprc_mean",
        "auprc_std",
        "p@k_mean",
        "p@k_std",
        "lift_mean",
        "lift_std",
    ]

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
    print(f"FULL ABLATION OVER SEEDS | n_seeds={n_seeds} | seeds={seeds}")
    print("=" * 90)

    hparams = rules_root = None
    if RAD_HPARAMS_FILE:
        hparams = rad_per_seed.load_hparams(RAD_HPARAMS_FILE, "amazon", seeds)
        if DETERMINISTIC:
            rad_per_seed.check_deterministic_studies(hparams, RAD_HPARAMS_FILE)
        rad_per_seed.check_label_env(REUSE_BASELINES_FROM, {k: os.environ.get(k) for k in LABEL_ENV_VARS})
        rules_root = Path(REUSE_BASELINES_FROM) / "rules"
        hp_config = {
            "rad_hyperparameters_file": RAD_HPARAMS_FILE,
            "rad_hyperparameters_file_md5": rad_per_seed.file_md5(RAD_HPARAMS_FILE),
            "rad_hyperparameters_per_seed": {str(s): {k: hparams[s][k] for k in rad_per_seed.TUNED}
                                             for s in seeds},
            "reuse_baselines_from": REUSE_BASELINES_FROM,
            "reused_per_seed_metrics_md5": rad_per_seed.file_md5(
                Path(REUSE_BASELINES_FROM) / "per_seed_metrics.csv"),
        }
    else:
        hp_config = {
            "rad_hyperparameters": {**RAD_HP, "BCE_arm_LAMBDA_STRUCT": RAD_BCE_LAMBDA_STRUCT},
            "rad_hyperparameter_source": RAD_HP_SOURCE,
            "rad_hyperparameter_note": RAD_HP_NOTE,
        }
    write_run_config(OUT_DIR, seeds, extra={
        **hp_config,
        "rad_run_env": run_env("amazon", os.environ),
        "deterministic_mode": DETERMINISTIC,
        "rf_seed_variant": RF_SEED_VARIANT,
        "rules_out_dir": (OUT_DIR / "rules").as_posix(),
        "raw_results_dir": (OUT_DIR / "raw").as_posix(),
        "use_cached_llm_rules": os.environ.get("USE_CACHED_LLM_RULES", "0 (default: LLM regenerates rules)"),
    })
    all_rows: list[dict] = []
    for seed in seeds:
        hp = expect = None
        if REUSE_BASELINES_FROM:
            baselines, mlp_reused = rad_per_seed.load_reused_rows(REUSE_BASELINES_FROM, seed)
            hp = hparams[seed]
            # Rule table each arm must rebuild: the one the same arm built in the reused run.
            expect = {(ls, mode): rad_per_seed.find_rule_table(rules_root, RUN_PREFIX["amazon"], seed, ls, mode)
                      for ls, mode in ((RAD_BCE_LAMBDA_STRUCT, "llm"), (0.0, "llm"), (0.0, "rf_seed"))}
            if rad_per_seed.file_md5(expect[(0.0, "llm")]) != hp["rule_table_md5"]:
                raise RuntimeError(f"seed={seed}: {expect[(0.0, 'llm')]} is not the rule table the study used")
            if rad_per_seed.file_md5(expect[(RAD_BCE_LAMBDA_STRUCT, "llm")]) != hp["rule_table_md5"]:
                # The with-BCE LLM arm reruns the study's selected model only on the study's rule table.
                msg = f"seed={seed}: the with-BCE LLM arm's rule table is not the one the study used"
                if DETERMINISTIC:
                    raise RuntimeError(msg)
                print(f"[note] {msg}", flush=True)
        else:
            baselines = _run_baselines(seed)
            baselines = _alias_models(baselines)
            baselines["source"] = "baselines_script"

        rad_bce, _ = _run_rad_variant(
            seed=seed, lambda_struct=float(hp["lambda_struct"]) if hp else RAD_BCE_LAMBDA_STRUCT,
            hp=hp, expect_rule_table=expect and expect[(RAD_BCE_LAMBDA_STRUCT, "llm")])
        rad_no_bce, final_no_bce = _run_rad_variant(
            seed=seed, lambda_struct=0.0, hp=hp, expect_rule_table=expect and expect[(0.0, "llm")])
        mlp = final_no_bce["MLP (supervised detector)"]
        if REUSE_BASELINES_FROM:
            # The reported MLP row is the reused one; the rerun's copy is only a match check.
            rad_per_seed.check_mlp_match(mlp_reused, mlp, seed)
            mlp = mlp_reused
        rf_seed_rows = []
        if RF_SEED_VARIANT:
            rad_rf, _ = _run_rad_variant(seed=seed, lambda_struct=0.0, rf_rule_mode="rf_seed", hp=hp,
                                         expect_rule_table=expect and expect[(0.0, "rf_seed")])
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
                    "p@k": float(rad_bce["p@k"]),
                    "lift": float(rad_bce["lift"]),
                    "n_eval": int(rad_bce["n_eval"]),
                    "n_pos": int(rad_bce["n_pos"]),
                    "eval_fp": rad_bce["eval_fp"],
                    "seed": int(seed),
                    "source": "rad_vs_script",
                },
                {
                    "model": "RAD (no BCE)",
                    "auroc": float(rad_no_bce["auroc"]),
                    "auprc": float(rad_no_bce["auprc"]),
                    "p@k": float(rad_no_bce["p@k"]),
                    "lift": float(rad_no_bce["lift"]),
                    "n_eval": int(rad_no_bce["n_eval"]),
                    "n_pos": int(rad_no_bce["n_pos"]),
                    "eval_fp": rad_no_bce["eval_fp"],
                    "seed": int(seed),
                    "source": "rad_vs_script",
                },
                {
                    "model": "Supervised MLP",
                    **{k: mlp[k] for k in ("auroc", "auprc", "p@k", "lift", "n_eval", "n_pos", "eval_fp")},
                    "seed": int(seed),
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
    pretty_out = OUT_DIR / "mean_std_metrics_pretty.csv"

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
