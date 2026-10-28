# Table 4 reporting plan

Applies to the H&M, Amazon and LANL columns of Table 4 (CIKM '26).

## Code version

Every Table 4 number comes from a 10-seed run (seeds 0–9) of tag `table4-final-20261026b`.

This tag supersedes `table4-final-20261026`. On that tag, cached RelBench tables were changed in place, so the baselines script's RDL model and the RDL runner's seeds 1–9 scored rows other than the reporting partition.

- Each run_config.json records `code_version.git_commit` and `code_version.git_tag`.
- The launch scripts refuse to start in three cases: HEAD is not this tag, a tracked file is modified, or the input data checksums differ from the recorded ones.
- A row from any other commit is not reported.
- Exception: the H&M and Amazon headline RAD rows come from the per-seed-tuned rerun (see "RAD hyperparameters" below), which runs from a later tag. That tag is recorded in this file when it is created.

## Evaluation rows

Each row records the fingerprint (`eval_fp`) of the rows it was scored on.

- **H&M and Amazon:**
  - The rows are the reporting partition of the RelBench val table. `VAL_FIT_FRAC=0.6` puts 60% of it in the fit partition, and the other 40% is the disjoint reporting partition.
  - The partition is drawn per seed. Within a seed, every model is scored on the same rows; different seeds use different rows.
  - `eval_fp` hashes the scored (customer_id, timestamp, label) rows (`scripts/eval_split.py`).
  - The ablation runners refuse to merge rows whose `n_eval`, `n_pos` or `eval_fp` differ (`ablation_provenance.check_same_eval_set`).
- **LANL:**
  - The rows are `auth.test_mask` of `lanl/data/sampled_hetero_graph.pt`.
  - `eval_fp` hashes the scored (auth_id, label) rows (`lanl/eval_fingerprint.py`).
  - Every model and seed is scored on the same rows.
  - `eval_fp` comes from `auth.test_mask`, so it is the same in every row by construction. Each row also records `test_rows` and `test_pos`, counted from the labels that were actually scored.
- A table is built only if `scripts/checks/check_eval_rows.py` (added in commit `cdd4a7b`) passes on every per-seed CSV of the dataset:
  - H&M and Amazon (`--per-seed`): within each seed, every row has the same `eval_fp`, `n_eval` and `n_pos`.
  - LANL (`--require-scored-counts`): every row has the same `eval_fp`, `n_eval` and `n_pos`, and each row has `test_rows == n_eval` and `test_pos == n_pos`.
  - Its tests: `python scripts/checks/test_check_eval_rows.py`.

## Rows

**Headline RAD row:** RAD with LLM rules generated in each run, no BCE.

- H&M and Amazon: row `RAD (no BCE)`, with `USE_CACHED_LLM_RULES=0`.
  - H&M also sets `HM_INCLUDE_BROAD_RULES=0`.
- LANL: variant `rad_llm_rules_no_bce`, with `LANL_LLM_RULES=per_run`.
  - The cached rule file `lanl/data/parsed_rules_df.csv` is never used for a reported row. It was selected with test labels.

**Ablation rows** (RF-sampled rules and BCE), reported below the headline row:

| | H&M / Amazon row | LANL variant |
|---|---|---|
| LLM rules, with BCE | `RAD (with BCE)` (λ_struct 1.0 on H&M, 0.1 on Amazon) | `llm_rules_with_bce` |
| RF-sampled rules, no BCE | `RAD (RF-sampled rules, no BCE)` (`RF_SEED_VARIANT=1`) | `rf_sampled_rules_no_bce` |
| RF-sampled rules, with BCE | — | `rf_sampled_rules_with_bce` |
| RF decision features | — | `rf_decision_features_no_bce`, `rf_decision_features_with_bce` |

**Baseline rows:**

- Isolation Forest, DRL, Deep SVDD, AE, AE+Rank.
- Supervised RDL, weighted and unweighted.
- Supervised MLP (see below).
- Not reported: the LANL RF supervised detector (decision 2026-10-28). Table 4 has no RF detector row.
- Not reported: the baselines script's own RDL model, labeled `Supervised RDL (baselines-script cross-check, not reported)` in the H&M and Amazon CSVs. It is a check on the Supervised RDL runner (`scripts/run_10seed_supervised_rdl_weighted.py`), which gives the reported RDL rows.

**Supervised MLP row:**

- Source:
  - LANL: `mlp_supervised_detector` in `lanl_six_ablations_by_seed.csv`.
  - H&M and Amazon: `Supervised MLP` from the RAD (no BCE) run of each seed. These rows take `n_eval`, `n_pos` and `eval_fp` from that run (they score the same label array) and have no scored-row count of their own.
- Settings shared by all datasets: sklearn `MLPClassifier`, relu, learning rate 1e-3, `max_iter` 200, early stopping (10% internal slice of the training rows, patience 10), `random_state` = seed, standardized features, no class weighting.
- LANL: (128, 64, 32), α = 1e-5, chosen by `lanl/tune_rf_mlp.py` with 3-fold CV on the train rows (average precision). It is trained on all `train_mask` rows.
- H&M and Amazon: (64, 32), α = 1e-4, untuned defaults. Trained on a class-balanced sample of about 160k rows of the training table.

**Cells:** each cell is the mean ± sample std (ddof = 1) over the 10 seeds, computed from the per-seed CSVs. The table also gives `n_eval` and `n_pos`; on H&M and Amazon they can vary by seed, so it gives their range.

**Rule logs:** each run's `rule_generation.json` records:

- `n_candidates_parsed` and `n_rejected_missing_columns`;
- `n_verbatim_prompt_copies`, the number of generated rules that are exact copies of an example predicate written in the prompt;
- `response_truncated`, true if the LLM response stopped before closing its JSON array (`max_new_tokens` is 4000 on all three datasets);
- `seed_fallback_used`. A run that generates its rules stops with an error instead of falling back to the RF seed rules, so this is false for every LLM-rules row. It is also false for the RF-sampled-rules rows: those runs load the 30 sampled RF rules directly and never take the fallback path.

## RAD hyperparameters (H&M and Amazon)

Decided 2026-10-28, before any 10-seed RAD number was seen.

The August studies (`hm48_optuna_postfix`, `rad32_optuna_postfix`) scored the full RelBench val table. That includes each seed's reporting partition: across the 10 seeds the reporting partitions cover about 99.4% of the val rows. So the RAD hyperparameters are re-selected per seed.

**Headline RAD rows: the per-seed-tuned rerun.** This covers every H&M and Amazon RAD row: `RAD (no BCE)`, `RAD (with BCE)` and `RAD (RF-sampled rules, no BCE)`.

- **Studies:** one Optuna study per dataset and seed (2 × 10 studies).
  - Each study runs the RAD script's own data preparation, model and training code. It has no separate copy of the model.
  - It validates on that seed's fit partition only. The reporting partition is never scored.
  - It uses that seed's LLM rule table from `results/<dataset>/final_20261028/rules/`, read-only.
  - It applies the RAD script's identity-proxy masking: `customer.postal_code` on H&M, `customer.customer_name` on Amazon.
  - Search space (same as the August studies):
    - λ_struct ∈ {0.1, 0.2, 0.5, 1.0, 2.0}
    - λ_rank ∈ {0.25, 0.5, 1.0, 2.0, 4.0}
    - α_feat ∈ {0.5, 1.0, 2.0}
    - α_struct ∈ {0.5, 1.0, 2.0}
  - Sampler and pruning: TPE sampler seed 42, 30 trials, objective fit-partition AUPRC, median pruner (4 startup trials, 2 warm-up epochs).
- **Hyperparameter file:** the selected hyperparameters go in a committed file with one row per dataset and seed. The runners read that seed's row.
  - The with-BCE row uses the selected λ_struct.
  - The no-BCE rows use λ_struct = 0 with the selected λ_rank, α_feat and α_struct.
- **Rerun:** only the RAD rows are re-run (10 seeds, all variants), from a tag that includes the hyperparameter file.
  - The rerun reuses the final_20261028 rows for the baselines and the baselines-script RDL cross-check.
  - `check_same_eval_set` must confirm identical `n_eval`, `n_pos` and `eval_fp` across the reused and rerun rows for every seed.
  - Each rerun must rebuild exactly the rule table its study used; otherwise it stops.
- **Supervised MLP:** the reported row stays the final_20261028 row. The rerun's MLP row is a match check only, and the rerun stops if it differs.
- **Supervised RDL:** unchanged. It comes from `scripts/run_10seed_supervised_rdl_weighted.py` and does not depend on the RAD hyperparameters.
- **Reported with the table:** the selected hyperparameters for each dataset and seed, and each study's best fit-partition AUPRC with its count of completed and pruned trials.

**Secondary:** the final_20261028 RAD rows, with the August hyperparameters. They are reported as secondary rows, with the caveat that their hyperparameters were tuned on the full val table.

**Deadline:** if the tuned rerun is not finished by 2026-11-05, the headline H&M and Amazon RAD rows are the final_20261028 rows, and errata item 10 is disclosed with the table. "Finished" means all 10 seeds of both datasets, with every check passing. Tuned and untuned rows are not mixed within the table.

**LANL:** unchanged. There is no Optuna record for the LANL RAD hyperparameters (see the caveats).

## H&M label configuration

The unexpected-churn label settings from §6.3.3 are exported by the launch script and echoed into each run.log. The H&M and Amazon runs use the same values.

```
ATRISK_TAU_A=0.05  ATRISK_TAU_B=10.0  ATRISK_TAU_ALPHA=0.02  ATRISK_TAU_MULTIPLIER=1.0  TARGET_ANOM_RATE=0.015
```

## Caveats to state with the table

- For the secondary (final_20261028) RAD rows: the H&M hyperparameters were tuned (study `hm48_optuna_postfix`) on broad-rule features. Both August studies scored the full RelBench val table, which includes the reporting partition. They also trained RAD differently from the RAD scripts: the rank loss used raw rather than per-batch z-scored errors, with no warm-up, and 10 epochs with patience 3.
- No Optuna record exists for the LANL RAD hyperparameters. Their source is author recollection; see `run_config.json`.
- The paper text needs corrections, which the authors will make (full list in `docs/paper_errata.md`). For example, the LANL MLP was chosen by 3-fold CV on the train rows, not "selected using the validation split".
- Results from before this tag used misaligned flat features (H&M, Amazon), all-time review aggregates (Amazon) and zero-filled categorical rule columns (LANL). Results from before this tag also include the cached-table re-split (H&M and Amazon RDL), and Amazon LLM-rules rows that were silently built from RF seed rules after a truncated LLM response. They are superseded and not reported.
