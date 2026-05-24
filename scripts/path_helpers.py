from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


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
