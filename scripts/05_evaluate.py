"""05: End-to-End 评估。

用法:
    python scripts/05_evaluate.py --config configs/coco_siglip2.yaml \\
        --assignment outputs/assignment_K256_M1.pt \\
        --router outputs/router_K256.pt

输出:
    outputs/eval_K<K>.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_embeddings
from src.evaluate import evaluate_end_to_end, format_eval_table
from src.router import load_router
from src.utils import (ensure_dir, get_device, load_config,
                       set_seed, setup_logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "05_evaluate.log"))

    # 加载所有材料
    logger.info("Loading embeddings...")
    emb = load_embeddings(
        cfg["data"]["image_emb_path"],
        cfg["data"]["text_emb_path"],
        expected_dim=cfg["data"]["embed_dim"],
    )
    assignment_blob = torch.load(args.assignment, map_location="cpu", weights_only=False)

    logger.info(f"Loading router from {args.router}")
    router = load_router(
        args.router,
        hidden_dim=cfg["router"]["hidden_dim"],
        n_layers=cfg["router"]["n_layers"],
        dropout=cfg["router"]["dropout"],
    )

    K = assignment_blob["K"]
    output = args.output or str(out_dir / f"eval_K{K}.json")

    device = str(get_device(cfg["data"]["device"]))
    logger.info(f"Evaluating on {device}...")

    metrics = evaluate_end_to_end(
        router=router,
        text_features=emb.text_features,
        text_image_ids=emb.text_image_ids,
        image_features=emb.image_features,
        image_ids=emb.image_ids,
        hard_assignment=assignment_blob["hard_assignment"],
        top_B_list=tuple(cfg["eval"]["top_B"]),
        top_K_list=tuple(cfg["eval"]["top_K"]),
        device=device,
        candidate_pool_size=cfg["eval"]["candidate_pool_size"],
    )

    # 打印格式化表
    table = format_eval_table(
        metrics,
        tuple(cfg["eval"]["top_B"]),
        tuple(cfg["eval"]["top_K"]),
    )
    logger.info(f"\n{table}")

    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✓ Saved metrics to {output}")


if __name__ == "__main__":
    main()
