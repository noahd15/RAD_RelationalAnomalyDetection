from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_JSONL_PATH = ROOT_DIR / "artifacts" / "results.jsonl"


def append_results_jsonl(record: Dict[str, Any], path: Path = RESULTS_JSONL_PATH) -> None:
    """Append one structured result record (method, task, seed, split, metrics, ...).

    Used instead of scraping stdout with regex or overwriting a single fixed-path CSV
    each seed: every run appends its own line, so per-seed/per-method history survives
    crashes and reruns and downstream consumers don't have to reparse printed tables.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("run_tag", os.environ.get("RUN_TAG", ""))
    with open(path, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")


def data_file(dataset: str, filename: str) -> str:
    return str(ROOT_DIR / "data" / dataset / filename)


def artifact_file(dataset: str, filename: str) -> str:
    return str(ROOT_DIR / "artifacts" / dataset / filename)


def results_dir(dataset: str, name: str) -> Path:
    return ROOT_DIR / "results" / dataset / name


def amazon_data(filename: str) -> str:
    return data_file("amazon", filename)


def hm_data(filename: str) -> str:
    return data_file("hm", filename)


def shared_data(filename: str) -> str:
    return data_file("shared", filename)
