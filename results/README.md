# Results Layout

This directory is organized by dataset domain.

## Folders
- `amazon/`: Amazon experiment outputs.
- `hm/`: H&M experiment outputs.
- `shared/`: Cross-dataset notebooks and combined analyses.

## Backward compatibility
Legacy top-level names are kept as symbolic links.
Examples:
- `results/full_ablation` -> `results/amazon/full_ablation`
- `results/full_ablation_hm` -> `results/hm/full_ablation_hm`

## Rule for new runs
Write dataset-specific outputs to `results/amazon/` or `results/hm/`.
