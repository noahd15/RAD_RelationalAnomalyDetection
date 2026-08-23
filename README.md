# RAD: Rule-Augmented Relational Anomaly Detection

[![Conference](https://img.shields.io/badge/CIKM%2726-Link-blue)](https://www.cikm2026.org/)

This is the official implementation of **RAD: Rule-Augmented Relational Anomaly Detection** from the CIKM'26 paper.

## Overview

RAD is a novel approach for detecting anomalies in relational data by augmenting learning-based methods with symbolic rules. The method combines the strengths of neural networks with interpretable rule-based reasoning to identify unusual patterns and outliers in structured relational datasets.

**Key Features:**
- Rule-augmented detection framework
- Scalable to large relational databases
- Interpretable anomaly explanations
- Strong empirical performance on benchmark datasets

## Repository Structure

```
RAD_RelationalAnomalyDetection/
├── DRL/                    # Deep Reinforcement Learning components
├── notebooks/              # Jupyter notebooks for experiments and tutorials
├── scripts/                # Utility scripts for data processing and evaluation
├── lanl/                   # LANL dataset-specific code
├── artifacts/              # Generated outputs and models
├── data/                   # Datasets (download required)
├── results/                # Experimental results and logs
├── requirements.txt        # Python dependencies
└── SUBMISSION_MANIFEST.md  # Submission documentation
```

## Requirements

Python 3.7+ with the following dependencies:

```
scikit-learn>=0.24.0
numpy>=1.19.0
pandas>=1.1.0
pytorch>=1.8.0
networkx>=2.5
```

Install all requirements with:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Datasets

Place your relational datasets in the `data/` directory. For detailed dataset format specifications, refer to the `SUBMISSION_MANIFEST.md`.

### 2. Run Experiments

Execute experiments using the provided scripts:

```bash
python scripts/run_experiment.py --config config.json --dataset <dataset_name>
```

### 3. Explore Notebooks

For tutorials and exploratory analysis, check the notebooks in `notebooks/`:

```bash
jupyter notebook notebooks/
```

## Dataset Format

Relational datasets should be provided as CSV files with the following structure:

| **Entity ID** | **Attribute 1** | **Attribute 2** | **Label** |
| --- | --- | --- | --- |
| 0 | value_a | value_b | 0 |
| 1 | value_c | value_d | 1 |
| ... | ... | ... | ... |

Where `Label` indicates anomaly status (0 = normal, 1 = anomaly).

## Method Overview

RAD integrates rule discovery with neural anomaly detection through:

1. **Rule Learning**: Automatically discovers interpretable rules from relational data
2. **Neural Augmentation**: Augments neural models with discovered rules
3. **Anomaly Scoring**: Combines rule-based and learning-based scores for robust detection
4. **Explainability**: Provides rule-based explanations for detected anomalies

## Citation

If you use RAD in your research, please cite our paper:

```bibtex
@inproceedings{RAD2026,
  title={RAD: Rule-Augmented Relational Anomaly Detection},
  author={Noah D. and Advisor Name},
  booktitle={Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM)},
  year={2026}
}
```

## Results

Experimental results on benchmark datasets are stored in `results/`. For detailed performance metrics and comparisons with baselines, refer to the paper.

Key findings:
- Achieves state-of-the-art anomaly detection performance
- Provides interpretable explanations for detected anomalies
- Scales efficiently to large relational databases

## Advanced Usage

### Custom Configuration

Modify `config.json` to customize hyperparameters:

```json
{
  "rule_learning": {
    "max_rules": 100,
    "min_support": 0.05
  },
  "neural_model": {
    "hidden_dim": 128,
    "learning_rate": 0.001
  },
  "detection": {
    "threshold": 0.5
  }
}
```

### Parallel Processing

For large datasets, enable multi-processing:

```bash
python scripts/run_experiment.py --config config.json --num_workers 16
```

## Troubleshooting

**Memory Issues**: Reduce batch size or dataset size in configuration.

**Slow Performance**: Ensure PyTorch is using GPU acceleration (`torch.cuda.is_available()`).

**Missing Dependencies**: Run `pip install -r requirements.txt --upgrade` to ensure all packages are properly installed.

## Contributing

We welcome contributions! Please feel free to open issues or submit pull requests for bugs, features, or improvements.

## License

This project is provided for academic and research purposes.

## Questions & Support

For questions or issues:
- Open an [issue](https://github.com/noahd15/RAD_RelationalAnomalyDetection/issues) on GitHub
- Refer to `SUBMISSION_MANIFEST.md` for technical details

## References

- The paper includes comprehensive references to related work in anomaly detection and relational learning
- Related work on graph neural networks and rule learning can be found in the paper's bibliography

---

**Last Updated**: August 2026
