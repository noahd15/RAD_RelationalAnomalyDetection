# H&M column of Table 4: provenance problem and correction

**Status:** needs a decision from the authors before the paper or poster is updated.

## Summary

The H&M column of Table 4 (CIKM '26 paper) combines rows that were scored on
three different evaluation sets, and the RAD rows come from a run whose
unexpected-churn label settings differ from the ones in Section 6.3.3.
Re-deriving the column from a single run with the paper's stated settings gives
much lower RAD numbers on H&M. RAD then trails the RDL baselines on H&M, but
it **still has the best average AUROC and AUPRC rank across the three datasets**.
LANL and Amazon are unaffected.

## Where each H&M row in Table 4 came from

| Table 4 row (H&M) | Source | Label settings (λ0, λ1, λ2, τ) | Evaluated on |
|---|---|---|---|
| RAD (rules, no BCE / BCE) | `results/hm/aggr_fast/full_ablation_10seed.log`, copied into `results/hm/full_ablation_hm_10seed/per_seed_metrics.csv` | 0.08, 6.0, 0.05, τ = 0.6 | 13,804 users / **207** anomalies |
| RDL weighted | `results/hm/rdl_weighted_10seed/` | same as above | 13,804 / **207** |
| RDL unweighted, AE, AE+Rank, DRL, Deep SVDD, Isolation Forest | baselines section of the `aggr_fast` run | 0.08, 6.0, 0.05, **τ ignored** (bug, see below) | 14,337 / **215** |
| Paper, §6.3.3 | — | **0.05, 10.0, 0.02, τ = 1.0** | 14,219 / **213** |

Evidence:

- All 20 RAD rows in the old `per_seed_metrics.csv` appear verbatim in the
  `aggr_fast` log. None of them appear in the `run.log` stored in the same folder.
- P@k equals hits / number of anomalies, so it reveals the anomaly count. The
  RAD values are multiples of 1/207 (e.g. 27/207 = 0.1304) and the baseline
  values are multiples of 1/215.

## Corrected H&M numbers (paper settings, one evaluation set)

From `results/hm/full_ablation_hm_10seed/run.log` (10 seeds, 14,219 users / 213
anomalies). The tables in that folder are now rebuilt from the log with
`scripts/rebuild_ablation_from_log.py`.

| Model | AUROC | AUPRC | P@k |
|---|---|---|---|
| RAD (rules, no BCE) | 0.667 ± 0.018 | 0.031 ± 0.003 | 0.044 ± 0.013 |
| RAD (rules, BCE) | 0.676 ± 0.025 | 0.030 ± 0.003 | 0.037 ± 0.011 |
| Supervised RDL (same run) | 0.698 ± 0.026 | 0.036 ± 0.004 | 0.060 ± 0.018 |
| DRL | 0.474 ± 0.021 | 0.014 ± 0.001 | 0.008 ± 0.007 |
| AE+Rank | 0.454 ± 0.028 | 0.013 ± 0.001 | 0.002 ± 0.002 |
| AE | 0.397 ± 0.027 | 0.012 ± 0.001 | 0.002 ± 0.005 |
| Deep SVDD | 0.392 ± 0.025 | 0.011 ± 0.001 | 0.004 ± 0.004 |
| Isolation Forest | 0.388 ± 0.026 | 0.011 ± 0.001 | 0.002 ± 0.003 |

Table 4 lists RDL weighted and unweighted separately. Runs with the paper
settings exist only for 5 seeds: `rdl_weighted_5seed_rankboost` (0.706 / 0.040 / 0.070)
and `rdl_unweighted_5seed_rankboost` (0.698 / 0.039 / 0.076), both on 213
anomalies. The H&M **Supervised MLP** row (0.642 / 0.027 / 0.039) has no source
file in this repository and should be re-checked.

## Effect on average ranks

Ranks are recomputed with the paper's method (per-dataset rank, ties averaged,
mean over LANL, Amazon, H&M); with the old H&M column this method reproduces the
published ranks exactly. The corrected H&M column uses the table above, the
5-seed RDL runs, and the unverified MLP row.

| Model | AUROC (paper → corrected) | AUPRC | P@k |
|---|---|---|---|
| RAD (rules, no BCE) | 2.0 → **3.0** (still best) | 1.2 → **1.8** (still best) | 2.0 → 2.7 |
| RAD (rules, BCE) | 3.3 → 3.7 | 1.8 → 2.5 | 2.3 → 3.3 |
| RDL weighted | 4.7 → 4.0 | 4.8 → 3.8 | 3.7 → 3.2 |
| RDL unweighted | 4.3 → 3.7 | 3.8 → 3.5 | 3.3 → **2.5** (now best) |
| Supervised MLP | 4.0 → 4.0 | 4.7 → 4.7 | 4.7 → 4.3 |

What changes in the text:

- §8.1 "On H&M, RAD achieves the best performance across AUROC, AUPRC, and
  precision@k" does not hold with the stated settings. RDL is ahead on all three.
- The abstract claim ("best average rank on AUROC and AUPRC") still holds.
- Under the `aggr_fast` label settings, RAD does beat RDL weighted on the
  *same* 207-anomaly set (AUPRC 0.143 vs 0.035). The H&M result therefore
  depends on the label definition. If the authors prefer those settings, §6.3.3
  and every H&M row (including the baselines) must use them consistently.

## Bugs that allowed this (fixed on this branch)

1. **Baselines ignored `ATRISK_TAU_MULTIPLIER`.** With τ ≠ 1, the baselines
   and RAD were labeled differently (215 vs 207 anomalies). Fixed in
   `hm_user_churn_baselines.py` and `amazon_user_churn_baselines.py`.
2. **Shared, stale baselines CSV.** The H&M runners read
   `results/full_ablation_hm/baselines_metrics.csv`, while the baselines script
   writes to `results/hm/full_ablation_hm/`, so they could merge a stale file
   from another run. Each run now writes and reads its own
   `<OUT_DIR>/baselines/seed<N>/baselines_metrics.csv`.
3. **No record of the evaluation set.** Every metric row now carries `n_eval`
   and `n_pos`, and a run fails if the models for one seed were scored on
   different sets (`scripts/ablation_provenance.py`). Each run also writes a
   `run_config.json` with its label settings.
4. The 5-seed runners pointed at `scripts/active/…`, which no longer exists.

## Open question: model selection on the evaluation set

The H&M and Amazon logs say *"test split has no 'churn' labels; using validation
split for final evaluation"*, and RAD's early-stopping "best val AUPRC" equals
its reported AUPRC. Model selection and evaluation therefore appear to use the
same split. §8.6 describes "a held-out partition of the validation split";
please confirm the code actually holds that partition out.

## Suggested next steps

1. The authors decide which label settings define the H&M benchmark.
2. Re-run H&M (RAD, both RDL variants, tabular baselines, supervised MLP) for
   10 seeds with those settings using the fixed runner, then regenerate Table 4
   and the ranks from that single output.
3. Post an arXiv v2 and ask the CIKM proceedings chairs about an erratum if
   the ACM version is already final.
