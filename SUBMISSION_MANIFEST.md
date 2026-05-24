# Conference Submission Manifest (2026-05-24)

This folder is a minimal submission copy of the RAD project for conference artifact review.

## Included
- `DRL/` including `DRL/Data/` for the tabular anomaly-detection baselines.
- `lanl/` code and `lanl/results/`.
- `scripts/`, flattened to only the paper-needed runners and helpers.
- `notebooks/active/`.
- `results/` with baselines, ablations, sweeps, summary CSVs, and rendered outputs.
- `weights/` with saved RAD checkpoints used by the user-churn baselines.
- `artifacts/shared/rad_latent_sweep_results.csv` and the compatibility symlink `artifacts/rad_latent_sweep_results.csv`.

## Excluded Intentionally
- `data/amazon/` and `data/hm/` payloads.
- `lanl/data/` payloads.
- Large duplicated artifact blobs such as `artifacts/amazon/amazon_churn_graph.pt`.
- `results/tmp/`.
- Virtual environments, logs outside experiment results, prior submission folders, and workspace-only archives.

## Data Retrieval
- Amazon and H&M instructions: `data/README.md`
- LANL instructions: `lanl/data/README.md`

## Notes
- The omitted Amazon/H&M/LANL data were left out because they are too large for a GitHub-backed submission copy.
- The included `results/` tree already contains the baselines, ablations, sweeps, and summary tables produced in the main workspace.
- No Hugging Face token is stored in this submission copy; optional LLM-backed rule generation reads `HF_TOKEN` from the environment.