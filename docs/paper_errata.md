# Paper errata (CIKM '26 RAD paper vs code)

Places where the paper text does not match the code at tag `table4-final-20261026b`
(commit 3cf7fd5). The authors edit the paper; this file only records the discrepancies.
Page numbers are from `pdftotext RAD.pdf`.

## Model and training description

1. **Mean vs sum aggregation.**
   - Paper p5: "a two-layer heterogeneous GraphSAGE model with mean aggregation", and that it "matches the RDL baseline encoder in Section 7.3". p8 §7.2: "two-layer relation-aware GraphSAGE encoder with mean aggregation".
   - H&M and Amazon (RAD and RDL): relbench `HeteroGraphSAGE` with `aggr="sum"`, i.e. sum within each relation and sum across relations.
     - `scripts/hm_rad_vs_supervised_rdl.py:1509`
     - `scripts/rad_vs_supervised_rdl.py:1542`
     - `scripts/run_10seed_supervised_rdl_weighted.py:232`
     - `scripts/hm_user_churn_baselines.py:860`
   - LANL RAD: `SAGEConv` with its default mean aggregation within each relation, and `HeteroConv(..., aggr="sum")` across relations (`lanl/run_lanl_six_ablations.py:782-783, 789-790`).

2. **Adam vs AdamW.**
   - Paper p8 §7.2: "Adam optimization".
   - All RAD and RDL models use AdamW:
     - `scripts/hm_rad_vs_supervised_rdl.py:1550` (RAD), `:1288` (RDL)
     - `scripts/rad_vs_supervised_rdl.py:1584` (RAD), `:1306` (RDL)
     - `lanl/run_lanl_six_ablations.py:1147`
     - `scripts/run_10seed_supervised_rdl_weighted.py:423`
   - The tabular baselines (DRL, Deep SVDD, AE) use Adam (`scripts/hm_user_churn_baselines.py:425, 495, 608`).

3. **What "no BCE" sets to zero.**
   - Paper p8 §7.2: "the structure reconstruction weight α controls the contribution of the binary cross-entropy edge reconstruction term; no-BCE ablations set α to zero."
   - The code has two separate weights:
     - The BCE loss weight `lambda_struct` is 0 in no-BCE runs (`scripts/hm_rad_vs_supervised_rdl.py:1583`).
     - The score weight `alpha_struct` stays nonzero: H&M 2.0 (`scripts/run_full_ablation_hm_10seed.py:46`), Amazon 1.0, LANL 1.7857 (`lanl/run_lanl_six_ablations.py:81`).
   - So no-BCE runs still add the z-scored structure error, softplus(−edge logit), to the anomaly score (`scripts/hm_rad_vs_supervised_rdl.py:1616-1619`, `lanl/run_lanl_six_ablations.py:1228`). The edge decoder is then trained only through the rank loss (the rank loss is computed on the combined score, `:1579-1583`).
   - **Bias:** neutral against the baselines, because the reported RAD numbers are what the code computes. It changes what the BCE ablation (§8.3) shows: removing the BCE loss term, not removing structure from the score. The paper's reading ("edge reconstruction should be treated as an optional auxiliary objective") overstates what was removed.
   - **Reported numbers:** none; the text is fixed and the code kept (decision 2026-10-28). Making the code match the current text (α_struct = 0) would have changed every no-BCE row, including the headline row.

## Supervised RDL baseline

4. **Table 3 marks RDL GNN as not supervised.**
   - Paper p8 Table 3: "RDL GNN [11] ✗ ✓ ✗ ✗", where "Sup." means "direct supervision using ground-truth anomaly labels".
   - The text calls it "supervised RDL" (p7, p9, p10), and the code trains it with BCE on ground-truth labels (`scripts/run_10seed_supervised_rdl_weighted.py:447-450`).

5. **RDL training details missing from the paper.**
   - The reported RDL rows come from `scripts/run_10seed_supervised_rdl_weighted.py`. The paper does not state any of the following:
     - Training positives are downsampled to `TARGET_ANOM_RATE` (1.5%) after the at-risk filter, the same label budget as RAD (`:399-407`).
     - Weighted variant: `pos_weight = train_neg / train_pos` after downsampling, about 65.7 on H&M (`:425-428`). Unweighted variant: plain BCE.
     - `customer.postal_code` is zeroed as an identity-proxy feature on H&M (`:97`), as in the H&M RAD script (`scripts/hm_rad_vs_supervised_rdl.py:52, 109`).
     - Optimization: AdamW lr 1e-3, weight decay 1e-4, gradient clip 2.0, batch 1024, neighbors [32, 32], at most 10 epochs × 2000 steps, early stopping on fit-partition AUPRC with patience 3 (`:71-74, 267-281, 423`).

## Labels and data

6. **H&M label settings** (`docs/HM_TABLE4_CORRECTION.md`).
   - Paper p7 §6.3.3: λ0 = 0.05, λ1 = 10.0, λ2 = 0.02.
   - The published H&M RAD rows used 0.08 / 6.0 / 0.05 with τ = 0.6. The published H&M baseline rows used the same λ values with τ ignored.
   - The rerun uses the paper's settings: `ATRISK_TAU_A=0.05 ATRISK_TAU_B=10.0 ATRISK_TAU_ALPHA=0.02 ATRISK_TAU_MULTIPLIER=1.0 TARGET_ANOM_RATE=0.015`, echoed into every run.log.

7. **H&M rules description.**
   - Paper p5 says the final rules are RF-derived predicates refined by the LLM and ranked by support, confidence and lift.
   - The published H&M runs read a cached rule file that held only two hand-crafted "broad" rules, repeated 16 times (32 rows; `data/hm/hm_parsed_rules_df.csv`, from `scripts/hm_rad_vs_supervised_rdl.py:990-993`): `recent_txns_30d <= 1` and `recent_txns_30d = 0 AND total_txns > 0`.
   - Both condition on recent transactions, an input of the D(u) label filter (`:1036-1037`).
   - The rerun generates LLM rules in each run (`USE_CACHED_LLM_RULES=0`) and drops the broad rules (`HM_INCLUDE_BROAD_RULES=0`, `:1038-1041`). `rule_generation.json` records `broad_rules_included` (`:1169`).

## Hyperparameter selection

8. **LANL MLP selection.**
   - Paper p8 §7.2: "For LANL, MLP hyperparameters are selected using the validation split."
   - They were chosen by 3-fold CV on the train rows with `lanl/tune_rf_mlp.py` and are passed in as `LANL_MLP_HIDDEN=128,64,32 LANL_MLP_ALPHA=1e-5` (`lanl/run_lanl_six_ablations.py:64-66`).
   - **Bias:** neutral. The MLP was tuned without using any val or test rows, which is at least as clean as the text claims. It's not known whether tuning on val would have helped or hurt the MLP.
   - **Reported numbers:** none; this is a text fix.

9. **LANL RF detector** (not reported). It uses fixed settings (1200 trees, `max_depth` None, `min_samples_leaf` 1, `class_weight` balanced_subsample), not the tuner's best.
   - The tuner's best was `max_depth` 20, `min_samples_leaf` 5, 800 trees, with CV AUPRC 0.0292 (`lanl/results/mlp_tuning_20261026/run.log`). The fixed settings were not scored by that CV, so the size of the difference is unknown.
   - **Bias:** if the row is reported, probably slightly for RAD, since the RF does not get its own tuned settings. Not measured.
   - **Reported numbers:** none. Decision 2026-10-28: no RF detector row is reported in Table 4.

10. **RAD hyperparameters and the reporting partition.**
    - Paper p8 §7.2: hyperparameters are "selected using Optuna on the validation split … then fixed for held-out test evaluation".
    - The H&M (`hm48_optuna_postfix`) and Amazon (`rad32_optuna_postfix`) tuners scored the full RelBench val table, which includes the reporting partition (`scripts/run_full_ablation_hm_10seed.py:49`, `scripts/run_full_ablation_10seed.py:48`).
    - The August tuners also trained RAD differently from the RAD scripts. Their rank loss scored raw errors, not per-batch z-scored ones (`scripts/optuna_hm_tune.py:342`, `scripts/optuna_rad32_tune.py:344`). They had no warm-up and ran 10 epochs with patience 3; the runs use 30/5 on H&M and 60/10 on Amazon, with warm-up.
    - No Optuna record exists for the LANL RAD hyperparameters.
    - **Resolution (reporting plan, commit c1f3056):** the H&M and Amazon headline RAD rows come from a per-seed study on each seed's fit partition, run by the RAD scripts' own code. If that rerun is not finished by 2026-11-05, the final_20261028 rows are reported and this item is disclosed.

11. **Label-based checkpoint selection for "unsupervised" baselines.**
    - Paper p8 Table 3 lists Deep SVDD and AE with no supervision.
    - Both choose their checkpoint by AUPRC on a labelled validation partition (`scripts/hm_user_churn_baselines.py:536, 648`; `lanl/run_lanl_tabular_baselines.py:302, 389`). On H&M and Amazon that is the fit partition, never the report rows.
    - **Bias:** against RAD, if anything. These baselines get label information the text says they don't have. RAD also stops early on validation AUPRC (§7.2), so in practice both are selected the same way.
    - **Reported numbers:** none; this is a text fix (state that all neural models, baselines included, select checkpoints on validation AUPRC).

13. **Per-seed numbers are not reproducible from the same seed and code.**
    - The RAD and RDL scripts call only `seed_everything(seed)`, which seeds Python, NumPy, torch and CUDA. They set no `torch.use_deterministic_algorithms`, no `cudnn.deterministic` and no `CUBLAS_WORKSPACE_CONFIG`, so nondeterministic GPU kernels (e.g. the order of the sums in the GNNs' sum aggregation) change the result from run to run.
    - Example: `results/hm/smoke_20261026_b` and `results/hm/final_20261028` ran the same commit (3cf7fd5), with the same environment, GPU type, seed 0 and byte-identical rule tables. Their H&M seed-0 rows (AUPRC / AUROC):
      - `RAD (with BCE)`: 0.0454 / 0.584 vs 0.0478 / 0.645
      - `RAD (no BCE)`: 0.0327 / 0.663 vs 0.0192 / 0.515
      - Amazon `RAD (with BCE)`: 0.0303 / 0.692 vs 0.0290 / 0.678
      - The supervised RDL GNN inside the H&M RAD script drifts too: AUPRC 0.0317, 0.0280 and 0.0286 for the three arms of one seed.
      - The Supervised MLP rows and the LLM and RF rule tables do repeat exactly.
    - **Bias:** none in a fixed direction. It adds variance to every GPU-trained row, RAD and the neural baselines alike.
    - **Reported numbers:** results are reported as 10-seed means ± std, and comparisons as paired-by-seed bootstrap CIs (`docs/table4_reporting_plan.md`, "Comparisons"). If the deterministic-mode check passes, the per-seed studies and tuned reruns run in deterministic mode, and each run_config.json records whether it did. The paper should not imply that a seed reproduces a per-seed number.

## Results text

12. **§8.1 H&M claim.**
    - Paper p9: "On H&M, RAD achieves the best performance across AUROC, AUPRC, and precision@k."
    - This does not hold with the §6.3.3 label settings in the earlier correction (`docs/HM_TABLE4_CORRECTION.md`).
    - To re-check against the 10-seed rerun, together with the other result sentences that depend on the H&M numbers: §8.2 "On H&M, the no-BCE variant is stronger across all three metrics", §8.4, and the average ranks in Table 4.

## Not paper text

- `docs/table4_reporting_plan.md` said `seed_fallback_used` "is true by design for the RF-sampled-rules rows". It is false: those runs load the 30 sampled RF rules directly and never take the fallback path. Fixed in commit c1f3056.
