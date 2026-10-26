# Table 4 reporting plan

Applies to the H&M, Amazon and LANL columns of Table 4 (CIKM '26).

## Code version

Every Table 4 number comes from a 10-seed run (seeds 0–9) of tag `table4-final-20261026`.

- Each run_config.json records `code_version.git_commit` and `code_version.git_tag`.
- The launch scripts refuse to start in three cases: HEAD is not this tag, a tracked file is modified, or the input data checksums differ from the recorded ones.
- A row from any other commit is not reported.

## Evaluation rows

Within each dataset, every row is scored on the same evaluation rows, and each row records their fingerprint (`eval_fp`).

- **H&M and Amazon:**
  - The rows are the reporting partition of the RelBench val table. `VAL_FIT_FRAC=0.6` puts 60% of it in the fit partition, and the other 40% is the disjoint reporting partition.
  - `eval_fp` hashes the scored (customer_id, timestamp, label) rows (`scripts/eval_split.py`).
  - The ablation runners refuse to merge rows whose `n_eval`, `n_pos` or `eval_fp` differ (`ablation_provenance.check_same_eval_set`).
- **LANL:**
  - The rows are `auth.test_mask` of `lanl/data/sampled_hetero_graph.pt`.
  - `eval_fp` hashes the scored (auth_id, label) rows (`lanl/eval_fingerprint.py`).
- A table is built only if every row in it has the same `eval_fp`, `n_eval` and `n_pos`.

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
- Supervised MLP:
  - H&M and Amazon: (64, 32), α = 1e-4.
  - LANL: (128, 64, 32), α = 1e-5, from `lanl/tune_rf_mlp.py`.
- LANL only: the RF supervised detector.

**Cells:** each cell is the mean ± sample std (ddof = 1) over the 10 seeds, computed from the per-seed CSVs. The table also gives `n_eval` and `n_pos`.

**Rule logs:** each run's `rule_generation.json` records `n_verbatim_prompt_copies`, the number of generated rules that are exact copies of an example predicate written in the prompt.

## H&M label configuration

The unexpected-churn label settings from §6.3.3 are exported by the launch script and echoed into each run.log. The H&M and Amazon runs use the same values.

```
ATRISK_TAU_A=0.05  ATRISK_TAU_B=10.0  ATRISK_TAU_ALPHA=0.02  ATRISK_TAU_MULTIPLIER=1.0  TARGET_ANOM_RATE=0.015
```

## Caveats to state with the table

- The H&M RAD hyperparameters were tuned (study `hm48_optuna_postfix`) on broad-rule features. The tuner scored the full RelBench val table, which includes the reporting partition.
- No Optuna record exists for the LANL RAD hyperparameters. Their source is author recollection; see `run_config.json`.
- Results from before this tag used misaligned flat features (H&M, Amazon), all-time review aggregates (Amazon) and zero-filled categorical rule columns (LANL). They are superseded and not reported.
