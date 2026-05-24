#!/usr/bin/env python3
"""Report total rows across all tables and rows in the task target table.

Usage:
  source rad/bin/activate
    python scripts/report_dataset_rows.py
    python scripts/report_dataset_rows.py --datasets amazon
    python scripts/report_dataset_rows.py --datasets amazon hm
"""

from __future__ import annotations

import argparse

from relbench.datasets import get_dataset
from relbench.tasks import get_task


DATASET_NAMES = {
    "amazon": "rel-amazon",
    "hm": "rel-hm",
}
TASK_NAME = "user-churn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_NAMES),
        default=sorted(DATASET_NAMES),
        help="Datasets to inspect.",
    )
    return parser.parse_args()


def report_dataset_rows(dataset_key: str) -> None:
    relbench_name = DATASET_NAMES[dataset_key]
    dataset = get_dataset(relbench_name, download=True)
    task = get_task(relbench_name, TASK_NAME, download=True)
    db = dataset.get_db()

    table_row_counts = {
        table_name: len(table) for table_name, table in db.table_dict.items()
    }
    total_rows = sum(table_row_counts.values())
    target_table = task.entity_table
    target_rows = table_row_counts[target_table]

    print(f"dataset={dataset_key}")
    print(f"  total_rows_all_tables={total_rows}")
    print(f"  target_table={target_table}")
    print(f"  target_table_rows={target_rows}")
    print("  rows_per_table=")
    for table_name in sorted(table_row_counts):
        print(f"    {table_name}: {table_row_counts[table_name]}")


def main() -> None:
    args = parse_args()
    for index, dataset_key in enumerate(args.datasets):
        if index:
            print()
        report_dataset_rows(dataset_key)


if __name__ == "__main__":
    main()