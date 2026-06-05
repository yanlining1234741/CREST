"""Unified path resolution for CREST."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    for key in ("CREST_DATA_ROOT", "QSBA_DATA_ROOT"):
        v = os.environ.get(key)
        if v:
            return Path(v).resolve()
    return (REPO_ROOT.parent / "mbeir_aligned").resolve()


def dataset_dir(name: str) -> Path:
    return data_root() / "data" / name


def output_dir(name: str) -> Path:
    return data_root() / "outputs" / name


def cross_encoder_dir(name: str, k: int, m: int) -> Path:
    return data_root() / "cross_encoder" / f"{name}_K{k}_M{m}"


def config_path(name: str) -> Path:
    p = REPO_ROOT / "configs" / "datasets" / f"{name}.yaml"
    if p.exists():
        return p
    # legacy fallbacks
    for suffix in ("_mbeir", "_task3", "_task0", "_task4", ""):
        legacy = REPO_ROOT / "configs" / f"{name}{suffix}.yaml"
        if legacy.exists():
            return legacy
    raise FileNotFoundError(f"No config for dataset '{name}'")
