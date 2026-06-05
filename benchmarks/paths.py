"""Paths for efficiency benchmarks (relative to CREST-github repo)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = Path(os.environ.get("WORK_DIR", Path(__file__).resolve().parent))
RESULTS = REPO_ROOT / "results" / "efficiency"
WORK = REPO_ROOT / "benchmarks" / "work"

# M-BEIR data root: sibling folder or env override
_data_default = REPO_ROOT.parent / "mbeir_aligned"
def _data_root() -> Path:
    v = os.environ.get("CREST_DATA_ROOT")
    if v:
        return Path(v)
    return _data_default


MBEIR = _data_root()

CREST_ROOT = REPO_ROOT

DATA = {
    "flickr": MBEIR / "data" / "flickr",
    "mscoco": MBEIR / "data" / "mscoco",
    "visualnews_task3": MBEIR / "data" / "visualnews_task3",
}

OUT = {
    "flickr": MBEIR / "outputs" / "flickr",
    "mscoco": MBEIR / "outputs" / "mscoco",
    "visualnews_task3": MBEIR / "outputs" / "visualnews_task3",
}

CE = {
    "flickr": MBEIR / "cross_encoder" / "flickr_K64_M8",
    "mscoco": MBEIR / "cross_encoder" / "mscoco_K128_M8",
    "visualnews_task3": MBEIR / "cross_encoder" / "vn_task3_K512_M6",
}

CREST_CONFIG = {
    "flickr": CREST_ROOT / "configs" / "datasets" / "flickr.yaml",
    "mscoco": CREST_ROOT / "configs" / "datasets" / "mscoco.yaml",
    "visualnews_task3": CREST_ROOT / "configs" / "datasets" / "visualnews_task3.yaml",
}
