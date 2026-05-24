# Scripts Guide

This submission keeps only the scripts needed to reproduce the paper results, all directly under `scripts/`.

## Recommended entrypoints
- `scripts/run_amazon_full_ablation.sh`
- `scripts/run_hm_full_ablation.sh`
- `scripts/run_amazon_rad_no_rules.sh`
- `scripts/run_hm_rad_no_rules.sh`
- `scripts/run_amazon_supervised_rdl_weighted.sh`
- `scripts/run_hm_supervised_rdl_weighted.sh`
- `scripts/run_rule_source_ablation.sh`
- `scripts/run_label_sensitivity_sweep.sh`

## Notes
- The wrappers activate `rad/bin/activate` if that environment exists.
- Optional LLM-backed rule generation requires `HF_TOKEN` in the environment; no token is stored in this package.
- Helper modules and paper-specific runners live alongside those entrypoints in the same folder.