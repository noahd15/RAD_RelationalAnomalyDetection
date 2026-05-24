# Data Retrieval Guide

This submission omits the large Amazon and H&M data payloads used by the user-churn experiments. The code in `scripts/` expects those files under this `data/` directory.

## Expected layout

### Amazon
Place the following files in `data/amazon/`:
- `relbench_rdl_graph_cache.pt`
- `relbench_rdl_col_stats_cache.pt`
- `amazon_parsed_rules_df.csv`
- `amazon_customer_rule_table.pt`
- `amazon_churn_rules_tree.csv`
- `amazon_churn_rules_tree_llm.csv`

### H&M
Place the following files in `data/hm/`:
- `relbench_rdl_graph_cache_hm.pt`
- `relbench_rdl_col_stats_cache_hm.pt`
- `hm_causal_price_article_cache.pkl`
- `hm_parsed_rules_df.csv`
- `hm_customer_rule_table.pt`

### Shared
Optional shared files go in `data/shared/`.

## How to get Amazon and H&M raw data

The raw relational datasets come from RelBench and are downloaded by the project code.

From the repository root:

```bash
python - <<'PY'
from relbench.datasets import get_dataset
from relbench.tasks import get_task

get_dataset("rel-amazon", download=True)
get_task("rel-amazon", "user-churn", download=True)
get_dataset("rel-hm", download=True)
get_task("rel-hm", "user-churn", download=True)
PY
```

## How to regenerate the cached files

### Amazon
- `scripts/amazon_user_churn_baselines.py` will build `relbench_rdl_graph_cache.pt` and `relbench_rdl_col_stats_cache.pt` automatically if they are missing.
- `scripts/rad_vs_supervised_rdl.py` or `notebooks/active/rad_amazon_user_churn.ipynb` writes `amazon_parsed_rules_df.csv` and `amazon_customer_rule_table.pt`.

### H&M
- `scripts/hm_user_churn_baselines.py` will build `relbench_rdl_graph_cache_hm.pt`, `relbench_rdl_col_stats_cache_hm.pt`, and `hm_causal_price_article_cache.pkl` automatically if they are missing.
- `scripts/hm_rad_vs_supervised_rdl.py` or `notebooks/active/rad_hm_user_churn.ipynb` writes `hm_parsed_rules_df.csv` and `hm_customer_rule_table.pt`.

## If you already have the prepared cache bundle

If you have the exact cache bundle from the original experiment environment, copy the prepared files directly into `data/amazon/` and `data/hm/` instead of rebuilding them.