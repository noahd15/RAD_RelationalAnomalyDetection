# LANL Data Guide

This submission omits the derived LANL data payload because it is too large for the GitHub-backed package.

## Expected files

Place these files in `lanl/data/`:
- `flat_features.parquet`
- `sampled_hetero_graph.pt`
- `parsed_rules_df.csv`
- `sample.duckdb` for `lanl/rdl_lanl.py` only

## Which scripts use them
- `lanl/run_lanl_six_ablations.py` uses `flat_features.parquet`, `sampled_hetero_graph.pt`, and `parsed_rules_df.csv`.
- `lanl/run_lanl_tabular_baselines.py` uses `sampled_hetero_graph.pt`.
- `lanl/rdl_lanl.py` uses `sampled_hetero_graph.pt` and `sample.duckdb`.

## How to get the files

- If you have the prepared LANL experiment bundle from the original run environment, copy those four files into this directory.
- If you are rebuilding from raw LANL source data, regenerate the same derived artifacts used in the main workspace and place the outputs here with the exact filenames above.

The final LANL experiment outputs are already included under `lanl/results/` and `results/lanl/`.