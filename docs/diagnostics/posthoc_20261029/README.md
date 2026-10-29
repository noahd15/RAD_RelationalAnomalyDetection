# Post-hoc diagnostics on the final_20261028 runs (exploratory)

**Post hoc.** These were run on 2026-10-29, after the final_20261028 10-seed results were seen. They are exploratory and change nothing in `docs/table4_reporting_plan.md`. No model was trained for them, and nothing under `results/` or `lanl/results/` was written.

Scripts:
- `scripts/diagnostics/table4_posthoc_diagnostics.py`: one run per dataset, CPU only. Logs are in `logs/`.
- `scripts/diagnostics/table4_posthoc_summary.py`: builds the summary CSVs in this folder.
- Section 5 uses the `--heuristics-only` mode of the first script (outputs in `heuristics/`) and the `--heuristics` mode of the second (`heuristic_summary.csv`).

## How the rows were obtained

- **H&M and Amazon.**
  - The final runs do not save labels or row lists. So, for each seed, the diagnostics script re-executes the data-preparation part of the RAD script: from the RelBench task tables through the end of SECTION 4 (val fit/report split, activity features, at-risk filter, downsampling). It runs with the reported label env and on CPU.
  - Each seed's rebuilt report rows reproduce the final run's `eval_fp`, `n_eval` and `n_pos` exactly (`*/rebuilt_rows_check.csv`), and the script stops otherwise.
  - The rule tables are the saved `results/<ds>/final_20261028/rules/*/customer_rule_table.pt` files. Within each seed, the no-BCE and with-BCE LLM tables have the same md5.
- **LANL.**
  - Each seed's LLM rule matrix is rebuilt from `rules/seed_N/llm_parsed_rules_df.csv` with the run's own `_rule_matrix_for_auth_ids`. Its per-rule match counts equal those in `six_ablations.log` for all 10 seeds.
  - The eval set is the run's (`ad800208e39b`, 63,627 rows, 127 positives).
  - LANL RF-sampled rules are not included: `lanl_rf_sampled_rules.csv` was overwritten each seed, so only seed 9's file survives.

## 1. Rule information content

**Method.** Logistic regression on the rule columns alone: sklearn, C = 1, lbfgs, no class weights. It is fit on the train rows (the at-risk-filtered, downsampled train rows that RAD trains on; LANL `train_mask`) and scores the report rows. The output is `*/rules_only_lr.csv`, and `rad_vs_rules_only_lr.csv` pairs it by seed with the RAD rows of the final runs.

Means over 10 seeds:

| Dataset | Rules | Report coverage (all / positives) | Rules-only LR AUROC | AUPRC | RAD row | RAD AUROC | AUPRC | RAD − LR AUPRC |
|---|---|---|---|---|---|---|---|---|
| H&M | LLM (15) | 0.68 / 0.50 | 0.637 | 0.0244 | RAD (no BCE) | 0.618 | 0.0280 | +0.0035 |
| H&M | LLM (15) | | | | RAD (with BCE) | 0.627 | 0.0317 | +0.0073 |
| H&M | RF-sampled (30) | 0.28 / 0.22 | 0.501 | 0.0161 | RAD (RF-sampled, no BCE) | 0.567 | 0.0238 | +0.0077 |
| Amazon | LLM (15–20) | 0.012 / 0.011 | 0.501 | 0.0151 | RAD (no BCE) | 0.685 | 0.0295 | +0.0144 |
| Amazon | LLM | | | | RAD (with BCE) | 0.689 | 0.0304 | +0.0153 |
| Amazon | RF-sampled (30) | 0.058 / 0.048 | 0.500 | 0.0151 | RAD (RF-sampled, no BCE) | 0.689 | 0.0303 | +0.0152 |
| LANL | LLM (2–20) | 0.016 / 0.34 | 0.661 | 0.0238 | rad_llm_rules_no_bce | 0.985 | 0.2161 | +0.192 |
| LANL | LLM | | | | llm_rules_with_bce | 0.951 | 0.2939 | +0.270 |

Prevalence on the report rows: 0.015 for H&M and Amazon, 0.002 for LANL.

**Reading:**
- **H&M:** the LLM rules alone come close to RAD. Rules-only LR has a higher AUROC than either RAD row, and an AUPRC within 0.004–0.007 of them.
- **Amazon:** the rules carry no information on the report rows. They fire on about 1% of rows (0% on one seed), and the LR AUROC is 0.50. RAD's Amazon signal therefore does not come from the rules.
- **LANL:** RAD is far above the rules alone.
- These are 10-seed means. Per-seed values are in the CSVs.

**No-rules GAE comparison:** not possible. None of the final_20261028 runs has a no-rules GAE variant.
- The H&M and Amazon arms are RAD with BCE, RAD without BCE and RF-sampled rules, plus the tabular baselines.
- The LANL variants all use rules or RF features.
- `results/{hm,amazon}/gae_norules_10seed` are from August, made with older code and labels. They are not comparable and were not used.

## 2. Score decomposition

Not possible from the saved outputs. The feature-error and structure-error terms are not stored separately anywhere:
- **H&M and Amazon:** the RAD scripts compute both terms (`vf, vs = _score(loader_test)`), but `save_obj` keeps only the combined `test_scores`, plus `norm_stats`.
- **H&M and Amazon runners:** the 10-seed runners do not set `SAVE_RAW_RESULTS_PATH`, so every arm overwrote `weights/rad_{hm,amazon}_user_churn.pt`. Only the last arm of seed 9 survives.
- **LANL:** `six_ablations_raw/*.npz` keep only `y_score`. `run_gae_single` computes `fe_te` and `se_te` but returns only the combined score.

**What saving them would cost (code only, nothing run):**
- **H&M and Amazon scripts:** add the raw `vf`, `vs`, `orig_vf` and `orig_vs` arrays to `save_obj` (about 4 lines per script).
- **10-seed runners:** pass each arm its own `SAVE_RAW_RESULTS_PATH` (about 2 lines per runner).
- **LANL:** have `run_gae_single` also return `fe_te` and `se_te` for the test rows, and add them to the npz (about 6 lines).
- **Storage:** small. It is two float32 arrays over the report rows (about 5.7k rows for H&M, 42k for Amazon, 63.6k for LANL).
- **Timing:**
  - If added, it should go in after the refactor equivalence check, because the comparer treats a new key in `raw_results.pt` as a difference. It also has to go in before the study-code tag, so the tuned reruns carry it.
  - LANL is not rerun under the plan, so LANL terms would need a new LANL run.

## 3. Stability (LANL)

Per seed: `lanl_stability.csv`. Correlations: `lanl_stability_corr.csv` (n = 10 seeds).
- **Rank-loss spikes: not available.** `run_gae_single` logs no per-epoch losses, and `six_ablations.log` has no epoch lines. The `max_rank_loss_spike` column is therefore empty.
- **Rules kept.** "Kept" is the LLM rule count after the val-support filter:
  - seed 1 kept 2 of 20;
  - seed 2 kept 5 of 13 (a truncated response);
  - every other seed kept 14–20.
- **AUPRC vs rules kept:**
  - Headline `rad_llm_rules_no_bce`: Pearson −0.22 (p = 0.55), Spearman −0.43 (p = 0.22). Leaving out one seed at a time, Pearson ranges from −0.43 to −0.12.
  - `llm_rules_with_bce`: Pearson −0.73 (p = 0.017), Spearman −0.52 (p = 0.12). Leaving out one seed at a time, Pearson ranges from −0.80 to −0.50.
- **AUPRC vs rules-only LR AUPRC:**
  - no BCE: Spearman −0.42 (p = 0.23);
  - with BCE: Spearman −0.27 (p = 0.45).
  - Seeds whose rules are more informative on their own do not get higher RAD AUPRC.
- With 10 seeds and no correction for the four tests, none of this is more than suggestive.

## 4. Activity-normality check (H&M and Amazon)

**Method.** Activity features are recomputed with the script's own `_activity_features_for_targets` on the report rows. Results: `*/activity_by_label.csv`, with means over seeds in `activity_summary.csv`. The AUROC column scores each feature as a churn predictor on the report rows; below 0.5 means positives have lower values. The table also includes the churners that the at-risk filter removed from the report partition (labelled 0 and dropped), for context.

| Dataset | Feature | Positives (mean) | Negatives (mean) | AUROC pos vs neg (range over seeds) | Removed at-risk churners (mean) |
|---|---|---|---|---|---|
| H&M | recent_txns (30 d) | 4.80 | 7.49 | 0.39 (0.35–0.42) | 0.00 |
| H&M | prev_txns_30d | 2.47 | 5.21 | 0.38 (0.34–0.41) | 3.92 |
| H&M | trend_w | 2.60 | 2.39 | 0.55 (0.52–0.60) | −0.74 |
| H&M | avg_activity | 3.64 | 6.35 | 0.36 (0.32–0.40) | 1.96 |
| Amazon | recent_reviews (30 d) | 0.86 | 1.47 | 0.38 (0.37–0.40) | 0.00 |
| Amazon | prev_reviews | 0.22 | 0.87 | 0.39 (0.39–0.41) | 1.50 |
| Amazon | trend_w | 0.71 | 0.69 | 0.51 (0.49–0.53) | −0.56 |
| Amazon | avg_activity | 0.54 | 1.17 | 0.36 (0.35–0.38) | 0.75 |
| Amazon | days_since_last_review | 68.3 | 50.5 | 0.61 (0.60–0.64) | 44.9 |

**Reading: partly confirmed.**
- **Trend:** the churners kept as positives have a trend similar to non-churners, slightly higher on H&M. The churners with falling trends were removed by the filter (mean trend −0.74 on H&M, −0.56 on Amazon). This similarity holds largely by construction: the filter selects on trend.
- **Activity level:** the kept positives are less active than average, not more "normal". Recent, previous and average activity are about 25–65% of the negatives' level (lowest for Amazon previous reviews), and on Amazon their last review is about 18 days older.
- So the unexpected churners look normal on trend, but not on activity level.

## 5. Trivial heuristic (H&M and Amazon)

**Post hoc, added 2026-10-29. Not part of Table 4.**

**Method.**
- A single activity feature is used as the anomaly score on the same report rows as above: the rebuilt rows match each final run's `eval_fp`, `n_eval` and `n_pos` for all 10 seeds.
- The features come from the script's own `_activity_features_for_targets` (30-day lookback). The sign is chosen so that higher means more likely to churn:
  - H&M: −recent_txns.
  - Amazon: −recent_reviews, and days_since_last_review as is.
- No fitting is involved. AUROC uses sklearn `roc_auc_score`; AUPRC uses `average_precision_score` (ties in these count features are handled by sklearn's default).
- Per seed: `heuristics/{hm,amazon}/heuristic_scores.csv`. Paired with the RAD rows by seed: `heuristic_summary.csv`. Logs: `logs/heuristics_{hm,amazon}.log`.

Means over 10 seeds (range over seeds in brackets):

| Dataset | Heuristic | AUROC | AUPRC |
|---|---|---|---|
| H&M | −recent_txns (30 d) | 0.607 (0.577–0.650) | 0.0203 (0.0182–0.0230) |
| Amazon | −recent_reviews (30 d) | 0.616 (0.597–0.635) | 0.0205 (0.0193–0.0220) |
| Amazon | days_since_last_review | 0.613 (0.595–0.641) | 0.0219 (0.0209–0.0244) |

Prevalence is 0.015 on both datasets, so random AUPRC is about 0.015.

The same seeds' RAD rows from the final runs, for comparison (means; "seeds" counts how many seeds have RAD above the heuristic):

| Dataset | RAD row | AUROC | AUPRC | vs heuristic: ΔAUROC (seeds) | ΔAUPRC (seeds) |
|---|---|---|---|---|---|
| H&M | RAD (no BCE) | 0.618 | 0.0280 | +0.011 (5/10) | +0.0076 (8/10) |
| H&M | RAD (with BCE) | 0.627 | 0.0317 | +0.020 (9/10) | +0.0114 (10/10) |
| H&M | RAD (RF-sampled, no BCE) | 0.567 | 0.0238 | −0.040 (2/10) | +0.0035 (7/10) |
| Amazon | RAD (no BCE) | 0.685 | 0.0295 | +0.069 / +0.072 (10/10) | +0.0091 / +0.0077 (10/10) |
| Amazon | RAD (with BCE) | 0.689 | 0.0304 | +0.073 / +0.076 (10/10) | +0.0099 / +0.0085 (10/10) |
| Amazon | RAD (RF-sampled, no BCE) | 0.689 | 0.0303 | +0.073 / +0.076 (10/10) | +0.0098 / +0.0084 (10/10) |

For Amazon, the two Δ values are against −recent_reviews and against days_since_last_review, respectively.

**Reading:**
- **H&M:** one activity count reaches about the AUROC of RAD (no BCE): 0.607 vs 0.618, with RAD ahead on only 5 of 10 seeds. It also beats the RF-sampled RAD row on AUROC. On AUPRC, RAD stays ahead: 0.0280–0.0317 vs 0.0203.
- **Amazon:** RAD is above both heuristics on every seed, by about 0.07 AUROC and 0.008–0.010 AUPRC.
- These AUROCs are 1 − the positives-vs-negatives feature AUROCs in section 4, as expected.

