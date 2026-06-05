#!/usr/bin/env python3
"""Write M-BEIR config YAMLs with portable paths from CREST_DATA_ROOT."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = {
    "flickr_mbeir.yaml": {
        "data": {
            "image_emb_path": "{root}/data/flickr/image_embeddings.pt",
            "text_emb_path": "{root}/data/flickr/image_embeddings.pt",
            "embed_dim": 768,
            "device": "cuda",
        },
        "query_aware": {"alpha": 1.0, "l2_normalize": True, "output_path": "outputs/query_aware_emb.pt"},
        "sinkhorn": {
            "K": 64,
            "M": 8,
            "epsilon": 0.01,
            "n_sinkhorn_iters": 5,
            "n_em_iters": 30,
            "init": "kmeans++",
            "output_path": "outputs/assignment.pt",
        },
        "router": {
            "hidden_dim": 512,
            "n_layers": 2,
            "dropout": 0.1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 256,
            "epochs": 60,
            "val_split": 0.1,
            "output_path": "outputs/router.pt",
        },
        "eval": {
            "top_B": [1, 3, 5],
            "top_K": [1, 5, 10],
            "candidate_pool_size": 1000,
            "rerank_with_dense": True,
        },
        "seed": 42,
        "log_level": "INFO",
        "output_dir": "{root}/outputs/flickr",
    },
    "mscoco_mbeir.yaml": {
        "data": {
            "image_emb_path": "{root}/data/mscoco/image_embeddings.pt",
            "text_emb_path": "{root}/data/mscoco/text_embeddings.pt",
            "embed_dim": 768,
            "device": "cuda",
        },
        "query_aware": {"alpha": 1.0, "l2_normalize": True, "output_path": "outputs/query_aware_emb.pt"},
        "sinkhorn": {
            "K": 128,
            "M": 8,
            "epsilon": 0.01,
            "n_sinkhorn_iters": 5,
            "n_em_iters": 30,
            "init": "kmeans++",
            "output_path": "outputs/assignment.pt",
        },
        "router": {
            "hidden_dim": 512,
            "n_layers": 2,
            "dropout": 0.1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 256,
            "epochs": 30,
            "val_split": 0.1,
            "output_path": "outputs/router.pt",
        },
        "eval": {
            "top_B": [1, 3, 5],
            "top_K": [1, 5, 10],
            "candidate_pool_size": 5000,
            "rerank_with_dense": True,
        },
        "seed": 42,
        "log_level": "INFO",
        "output_dir": "{root}/outputs/mscoco",
    },
    "vn_task3.yaml": {
        "data": {
            "image_emb_path": "{root}/data/visualnews_task3/image_embeddings.pt",
            "text_emb_path": "{root}/data/visualnews_task3/text_embeddings.pt",
            "embed_dim": 768,
            "device": "cuda",
        },
        "query_aware": {"alpha": 1.0, "l2_normalize": True, "output_path": "outputs/query_aware_emb.pt"},
        "sinkhorn": {
            "K": 512,
            "M": 6,
            "epsilon": 0.01,
            "n_sinkhorn_iters": 5,
            "n_em_iters": 30,
            "init": "kmeans++",
            "output_path": "outputs/assignment.pt",
        },
        "router": {
            "hidden_dim": 512,
            "n_layers": 2,
            "dropout": 0.1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 256,
            "epochs": 30,
            "val_split": 0.1,
            "output_path": "outputs/router.pt",
        },
        "eval": {
            "top_B": [1, 3, 5],
            "top_K": [1, 5, 10],
            "candidate_pool_size": 537568,
            "rerank_with_dense": True,
        },
        "seed": 42,
        "log_level": "INFO",
        "output_dir": "{root}/outputs/visualnews_task3",
    },
}


def _fill(obj, root: str):
    if isinstance(obj, dict):
        return {k: _fill(v, root) for k, v in obj.items()}
    if isinstance(obj, str):
        return obj.format(root=root)
    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        default=os.environ.get(
            "CREST_DATA_ROOT",
            os.environ.get("QSBA_DATA_ROOT", str(REPO.parent / "mbeir_aligned")),
        ),
        help="M-BEIR aligned data directory (embeddings + outputs)",
    )
    args = p.parse_args()
    root = str(Path(args.data_root).resolve())
    out_dir = REPO / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tpl in TEMPLATES.items():
        cfg = _fill(tpl, root)
        path = out_dir / name
        path.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
        print(f"Wrote {path}")
    print(f"\nSet: export CREST_DATA_ROOT={root}")


if __name__ == "__main__":
    main()
