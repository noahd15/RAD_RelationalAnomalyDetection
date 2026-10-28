"""Per-seed Optuna study for RAD (H&M and Amazon), run from inside the RAD scripts.

The RAD scripts call `run_study` from their tuning mode (RAD_TUNE_TRIALS > 0, section 4b), after
the data prep, identity-proxy masking, at-risk labels and fit/report split, and before any RF,
LLM or report-partition code. Each trial trains RAD with `rad_core` (the code the reported runs
use) on the seed's train rows and scores the seed's fit partition. The report partition is never
loaded.

Search space, sampler and pruner are the August tuners' (`scripts/optuna_hm_tune.py`,
`scripts/optuna_rad32_tune.py`). Every other setting (epochs, patience, warm-up, latent size,
loaders) is the RAD script's own `_hp` for the run, and every trial starts from
`seed_everything(seed)`, as the reported run does before building its model.
"""

import gc
import hashlib
import json
import os
from pathlib import Path

import optuna
import pandas as pd
import torch
from torch_geometric.seed import seed_everything

import rad_core

SEARCH_SPACE = {
    "lambda_struct": [0.1, 0.2, 0.5, 1.0, 2.0],
    "lambda_rank": [0.25, 0.5, 1.0, 2.0, 4.0],
    "alpha_feat": [0.5, 1.0, 2.0],
    "alpha_struct": [0.5, 1.0, 2.0],
}
TPE_SEED = 42
PRUNER_KWARGS = dict(n_startup_trials=4, n_warmup_steps=2, interval_steps=1)


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_study(*, dataset, seed, n_trials, out_dir, rule_table_path, data, col_stats_dict,
              loader_train, loader_val, val_labels, train_df, fit_df, entity_table, base_hp,
              device, tune_metric, val_roc_auc=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    study_db = out_dir / "study.db"
    for p in (study_db, out_dir / "best.json", out_dir / "trials.csv"):
        if p.exists():
            raise RuntimeError(f"{p} already exists; refusing to reuse or overwrite a study.")
    if data is None or col_stats_dict is None:
        raise RuntimeError("Tuning mode needs the cached graph; it does not build one.")

    rule_md5 = file_md5(rule_table_path)
    rt = torch.load(rule_table_path, weights_only=False)
    rule_table, n_rules = rt["rule_table"], int(rt["n_rules"])
    if n_rules == 0:
        raise RuntimeError(f"{rule_table_path} has no rules.")
    if rule_table.size(0) != data[entity_table].num_nodes:
        raise RuntimeError(f"{rule_table_path} has {rule_table.size(0)} rows, "
                           f"graph has {data[entity_table].num_nodes} {entity_table} nodes.")
    rule_table_gpu = rule_table.to(device)

    train_fp = rad_core.inputs_fingerprint(train_df)
    fit_fp = rad_core.inputs_fingerprint(fit_df)
    n_fit, n_fit_pos = int(len(val_labels)), int(val_labels.sum())
    if n_fit != len(fit_df):
        raise RuntimeError(f"val_labels has {n_fit} rows, fit table has {len(fit_df)}.")

    print("=" * 70)
    print(f"RAD per-seed study | dataset={dataset} seed={seed} trials={n_trials} metric={tune_metric}")
    print(f"  rule table: {rule_table_path} (md5 {rule_md5}, {n_rules} rules)")
    print(f"  rad_inputs_fp: train={train_fp} fit={fit_fp}")
    print(f"  fit partition: {n_fit} rows, {n_fit_pos} positives")
    print(f"  base hp: {json.dumps(base_hp, sort_keys=True)}")
    print("=" * 70, flush=True)

    def objective(trial):
        hp = dict(base_hp)
        for name, choices in SEARCH_SPACE.items():
            hp[name] = trial.suggest_categorical(name, choices)
        print(f"\nTRIAL {trial.number:03d} | " + " ".join(f"{k}={hp[k]}" for k in SEARCH_SPACE), flush=True)

        def on_epoch(epoch, val_target):
            trial.report(val_target, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned at epoch {epoch} with {tune_metric}={val_target:.4f}")

        seed_everything(seed)
        model = rad_core.build_model(data, col_stats_dict, hp, n_rules, device)
        opt = rad_core.make_optimizer(model, hp)
        try:
            best_metric, best_state, _ = rad_core.fit(
                model, opt, hp, loader_train, loader_val, val_labels, entity_table,
                rule_table_gpu, device, tune_metric, val_roc_auc=val_roc_auc, on_epoch=on_epoch)
            if best_state is None:
                raise RuntimeError("Trial ended without a best checkpoint.")
            return best_metric
        finally:
            del model, opt
            gc.collect()
            torch.cuda.empty_cache()

    def save_trials(study):
        study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)

    def after_trial(study, trial):
        save_trials(study)
        print(f"TRIAL {trial.number:03d} DONE | state={trial.state.name} | value={trial.value}", flush=True)

    study = optuna.create_study(
        study_name=f"rad_{dataset}_seed{seed}",
        storage=f"sqlite:///{study_db}",
        direction="maximize",
        load_if_exists=False,
        pruner=optuna.pruners.MedianPruner(**PRUNER_KWARGS),
        sampler=optuna.samplers.TPESampler(seed=TPE_SEED),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[after_trial], gc_after_trial=True)
    save_trials(study)

    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    if states.get("COMPLETE", 0) == 0:
        raise RuntimeError(f"No trial completed: {states}")
    if states.get("FAIL", 0):
        raise RuntimeError(f"{states['FAIL']} trial(s) failed: {states}")
    best = study.best_trial
    summary = {
        "dataset": dataset, "seed": int(seed), "tune_metric": tune_metric,
        "best_trial": best.number, "best_value": best.value, "best_params": best.params,
        "n_trials": len(study.trials), "n_complete": states.get("COMPLETE", 0),
        "n_pruned": states.get("PRUNED", 0),
        "search_space": SEARCH_SPACE, "tpe_seed": TPE_SEED, "pruner": PRUNER_KWARGS,
        "rule_table_path": str(rule_table_path), "rule_table_md5": rule_md5, "n_rules": n_rules,
        "train_fp": train_fp, "fit_fp": fit_fp, "n_fit": n_fit, "n_fit_pos": n_fit_pos,
        "base_hp": base_hp,
    }
    (out_dir / "best.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("=" * 70)
    print(f"STUDY COMPLETE | best trial {best.number} | {tune_metric}={best.value:.4f} | {best.params}")
    print(f"  trials: {states}")
    print("=" * 70, flush=True)
    return summary
