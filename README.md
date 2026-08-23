# RAD: Rule-Augmented Relational Anomaly Detection

[![Conference](https://img.shields.io/badge/CIKM%2726-Link-blue)](https://cikm2026.diag.uniroma1.it/)

This is the official implementation of **RAD: Rule-Augmented Relational Anomaly Detection** from the CIKM'26 paper.

## Overview

RAD is a rule-augmented graph neural network approach for detecting anomalies in relational data. The method combines:

1. **Heterogeneous Graph Neural Networks (GNN)**: Graph-SAGE based encoders for relational structure learning
2. **Autoencoder Architecture**: Bottleneck latent space for unsupervised anomaly detection
3. **LLM-Mined Rules**: Symbolic rule extraction using large language models (LLaMA 3.1)
4. **Structural Consistency Loss**: Binary cross-entropy loss ensuring rule-consistent predictions

**Key Innovation**: LLM-generated rules are incorporated as learned features in the GNN latent space, enabling the model to leverage both neural learning and interpretable symbolic reasoning.

## Repository Structure

```
RAD_RelationalAnomalyDetection/
├── scripts/                         # Main experiment runners
│   ├── rad_vs_supervised_rdl.py     # Primary RAD evaluation on Amazon user-churn
│   ├── hm_rad_vs_supervised_rdl.py  # RAD evaluation on H&M user-churn
│   ├── run_full_ablation_*seed.py   # Multi-seed ablation studies
│   ├── run_rule_source_ablations.py # Rule source comparison (LLM vs RF vs RF-decision)
│   └── ... (sweeps, baselines, analysis)
├── notebooks/active/                # Interactive analysis and experiments
│   ├── rad_amazon_user_churn.ipynb  # Amazon end-to-end walkthrough
│   └── (rule extraction, model validation)
├── lanl/                            # LANL cyber-security dataset variant
│   ├── rdl_lanl.py                  # LANL-specific training pipeline
│   └── data/README.md               # LANL data retrieval instructions
├── DRL/                             # DRL baseline (comparison only)
├── artifacts/                       # Pre-computed artifacts and sweeps
├── results/                         # Experimental outputs (logs, ablations, CSVs)
├── data/                            # Dataset cache (see `data/README.md` for downloads)
├── requirements.txt                 # Python dependencies
├── SUBMISSION_MANIFEST.md           # Artifact submission details
└── README.md                        # This file
```

## Requirements

Python 3.9+ with dependencies:

```
torch>=1.12.0
torch-geometric>=2.1.0
relbench>=0.1.0
pytorch_frame
sentence-transformers  # For text embeddings (GloVe)
duckdb                # For database operations
scikit-learn          # For RF rule mining
transformers          # For optional LLM rule generation
bitsandbytes          # For 4-bit quantized LLM inference
```

Install all requirements:

```bash
pip install -r requirements.txt
```

**Optional**: For LLM-backed rule generation (uses HuggingFace Llama-3.1-8B-Instruct):
```bash
export HF_TOKEN="your_huggingface_token"
```

## Quick Start

### 1. Download Datasets

Datasets are hosted on RelBench and will auto-download on first run:

```bash
python scripts/rad_vs_supervised_rdl.py
```

Alternatively, see `data/README.md` for manual download instructions.

### 2. Run RAD on Amazon User-Churn

Basic single-seed run:

```bash
python scripts/rad_vs_supervised_rdl.py
```

With custom hyperparameters:

```bash
SEED=42 LATENT_DIM=32 EPOCHS=30 LAMBDA_STRUCT=0.1 LAMBDA_RANK=1.0 \
  python scripts/rad_vs_supervised_rdl.py
```

### 3. Run Multi-Seed Ablation (Full Paper Results)

5-seed ablation on Amazon with and without rules:

```bash
N_SEEDS=5 python scripts/run_full_ablation_5seed.py
```

Compare rule sources (LLM vs Random Forest vs RF decision trees):

```bash
python scripts/run_rule_source_ablations.py
```

### 4. Explore Results

Summary metrics are saved to CSV:

```
results/amazon/full_ablation_5seed/per_seed_metrics.csv
results/shared/rule_source_ablation/mean_std_metrics.csv
```

Interactive analysis via Jupyter:

```bash
jupyter notebook notebooks/active/rad_amazon_user_churn.ipynb
```

## Core Components

### RADAutoencoder

The main model architecture:

```python
class RADAutoencoder(torch.nn.Module):
    """
    A heterogeneous graph autoencoder with rule augmentation.
    
    Architecture:
    - HeteroEncoder: Input feature encoding (text embeddings, numeric normalization)
    - HeteroTemporalEncoder: Temporal embedding (seed_time, historical context)
    - HeteroGraphSAGE: Relational structure encoding
    - to_latent: Linear projection to bottleneck space
    - rule_proj: Optional rule feature projection
    - Decoder GNN: Reconstructs from latent + rules
    - feat_decoder: Final feature reconstruction head
    - rel_diag: Per-edge-type scaling parameters for structural loss
    
    Losses:
    - Reconstruction MSE (features)
    - Structural BCE (rule consistency via causal edges)
    - Ranking loss (anomaly pool discrimination)
    """
```

### Rule Mining Pipeline

**Random Forest extraction** (always runs):
1. Train RF classifier on labeled anomalies (if available)
2. Extract decision paths from top trees
3. Convert paths to SQL predicates
4. Score by validation precision and support

**LLM-based generation** (optional, cached):
1. Sample top 30 RF-discovered rules as seed prompts
2. Query LLaMA-3.1-8B to generate semantically similar rules
3. Parse and validate rule SQL predicates
4. Cache results in `data/<dataset>/<dataset>_parsed_rules_df.csv`

**Rule feature representation**:
- Binary violation matrix: (N_entities, N_rules)
- Linearly projected into latent space: `rule_proj(rule_violations) → latent_dim`
- Added to bottleneck embeddings to bias reconstruction

### Datasets

| Dataset | Task | Entities | Anomalies | Relational Graph |
|---------|------|----------|-----------|------------------|
| **Amazon** (rel-amazon) | User churn prediction | 182K customers | ~1.5% | customer → order → product, customer → rating |
| **H&M** (rel-hm) | Customer churn | 66K customers | ~1.5% | customer → transaction → product, customer → article |
| **LANL** | Insider threat detection | 4.9K users | ~0.8% | user → host → event (cyber logs) |

Download via `relbench`:

```python
from relbench.datasets import get_dataset
from relbench.tasks import get_task

dataset = get_dataset("rel-amazon", download=True)
task = get_task("rel-amazon", "user-churn", download=True)
```

## Hyperparameters & Configuration

Common environment variables (see scripts for full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `SEED` | `0` | Random seed |
| `LATENT_DIM` | `32` (Amazon), `48` (H&M) | Bottleneck dimension |
| `CHANNELS` | `128` | Hidden feature dimension |
| `LAMBDA_STRUCT` | `0.1` | Structural BCE loss weight |
| `LAMBDA_RANK` | `1.0` | Ranking loss weight |
| `ALPHA_FEAT` | `1.0` | Reconstruction loss weight |
| `ALPHA_STRUCT` | `1.0` | Structural loss weight |
| `EPOCHS` | `30` | Training epochs |
| `PATIENCE` | `5` | Early stopping patience |
| `WARMUP_EPOCHS` | `5` | Warmup phase length |
| `USE_CACHED_LLM_RULES` | `1` | Use cached LLM rules (avoid regeneration) |
| `RAD_DISABLE_RULES` | `0` | Disable rules entirely (ablation) |
| `RF_RULE_MODE` | `llm` | Rule source: `llm`, `rf_seed`, or `rf_decision` |

Example: Run RAD without rules (ablation):

```bash
RAD_DISABLE_RULES=1 python scripts/rad_vs_supervised_rdl.py
```

Run with different rule source:

```bash
RF_RULE_MODE=rf_decision python scripts/rad_vs_supervised_rdl.py
```

## Training & Evaluation

### Loss Function

The model optimizes a weighted combination:

```
Loss = alpha_feat * MSE(x, x̂)                          # Reconstruction
     + alpha_struct * lambda_struct * BCE(ŷ, rule_violations)  # Structural
     + lambda_rank * ranking_loss(anomaly_pool)        # Ranking
```

Where:
- **Reconstruction MSE**: Fidelity to input features
- **Structural BCE**: Forces latent embeddings to align with rule violations
- **Ranking Loss**: Pushes anomalies in "at-risk pool" to higher scores

### Evaluation Metrics

- **AUROC**: Area under ROC curve (full set)
- **AUPRC**: Area under precision-recall curve (full set)
- **Partial AUROC**: AUROC at FPR ≤ 0.1 (practical focus)
- **P@K**: Precision at top-K predicted anomalies
- **Lift**: Ratio of anomaly rate in top-K vs global rate

## Baselines Included

For comparison, the repository includes implementations of:

- **Isolation Forest**: Tree-based outlier detection
- **Vanilla AE**: Standard tabular autoencoder (flattened features)
- **AE+Rank**: AE with anomaly-pool ranking
- **Deep SVDD**: Deep support vector data description
- **DRL**: Deep reinforcement learning baseline (from `DRL/` subdirectory)
- **Supervised RDL GNN**: Relational learning with labels (upper bound)

Run all baselines:

```bash
python DRL/baselines_runner.py
```

## Advanced Usage

### Rule Source Ablation

Compare LLM-generated rules vs Random Forest samples vs RF decision trees:

```bash
python scripts/run_rule_source_ablations.py
```

Outputs:
- Per-seed metrics across all rule sources
- Mean/std aggregates
- Ablation comparison plots

### Latent Dimension Sweep

Optimize bottleneck size (Amazon):

```bash
LATENT_DIM=4 python scripts/sweep_rad_latent_dims.py   # Runs across [4, 8, 16, 32, 48]
```

### Label Sensitivity

Study impact of anomaly label availability:

```bash
python scripts/run_label_sensitivity_rad_vs_rdl.py
```

### LANL Cyber Dataset

Evaluate on insider threat detection:

```bash
cd lanl
python rdl_lanl.py
```

See `lanl/data/README.md` for dataset setup.

## Key Results

### Amazon User-Churn (5-seed average)

| Model | AUROC | AUPRC | P@K | Lift |
|-------|-------|-------|-----|------|
| Isolation Forest | 0.614 | 0.028 | 0.29 | 1.09 |
| AE | 0.635 | 0.032 | 0.32 | 1.15 |
| AE+Rank | 0.681 | 0.041 | 0.41 | 1.52 |
| Deep SVDD | 0.699 | 0.054 | 0.47 | 1.89 |
| **RAD (rules + BCE)** | **0.752** | **0.089** | **0.58** | **2.41** |

### H&M User-Churn Results

Comparable improvements observed on H&M; see `results/full_ablation_hm/` for details.

---

**Last Updated**: August 2026  
**Maintainer**: Noah D.
