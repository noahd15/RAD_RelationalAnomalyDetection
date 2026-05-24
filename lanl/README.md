# LANL Minimal Transfer Bundle

This bundle is the smallest set of code and data to run LANL experiments.

## Included scripts
- run_lanl_six_ablations.py
- rdl_lanl.py
- monitor_run.sh

## Included data (if present at bundle time)
- data/flat_features.parquet
- data/sampled_hetero_graph.pt
- data/parsed_rules_df.csv
- data/sample.duckdb (only needed by rdl_lanl.py)

## Run on the target machine
1. Create and activate a Python environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Run LANL ablations:
   python run_lanl_six_ablations.py
4. Optional RDL baseline:
   python rdl_lanl.py

Outputs are written into results/.
