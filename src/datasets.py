"""Dataset registry: hyperparameters and paper settings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

BuildMode = Literal["standard", "capped"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    K: int
    M: int
    build_mode: BuildMode
    paper_r1: Optional[float] = None  # reference R@1 for sanity check
    n_queries: Optional[int] = None
    pool_size: Optional[int] = None
    ce_epochs: int = 12
    router_epochs: Optional[int] = None  # None = use config default


# Main paper Table 1 settings (M-BEIR protocol, cross-encoder rerank)
DATASETS: Dict[str, DatasetSpec] = {
    "flickr": DatasetSpec(
        name="flickr",
        K=64,
        M=8,
        build_mode="standard",
        paper_r1=77.98,
        n_queries=5000,
        pool_size=1000,
        router_epochs=60,
    ),
    "mscoco": DatasetSpec(
        name="mscoco",
        K=128,
        M=8,
        build_mode="standard",
        paper_r1=53.65,
        n_queries=24809,
        pool_size=5000,
    ),
    "visualnews_task3": DatasetSpec(
        name="visualnews_task3",
        K=512,
        M=6,
        build_mode="standard",
        paper_r1=19.59,
        n_queries=19898,
        pool_size=537568,
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
    return DATASETS[name]
