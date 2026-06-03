"""04: 训练 Router (Stage 3)。

用法:
    python scripts/04_train_router.py --config configs/coco_siglip2.yaml \\
        --assignment outputs/assignment_K256_M1.pt

输出:
    outputs/router_K<K>.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_embeddings
from src.router import (RouterDataset, RouterMLP, save_router, train_router)
from src.utils import (ensure_dir, get_device, load_config,
                       set_seed, setup_logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n-train-queries", type=int, default=None)
    parser.add_argument("--use-all-queries", action="store_true",
        help="显式允许用全部query训练(仅当test已分离). 单池不传此项且不传--n-train-queries会报错")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "04_train_router.log"))

    epochs = args.epochs if args.epochs is not None else cfg["router"]["epochs"]

    # 加载原始 text features 和 assignment
    logger.info("Loading text embeddings and assignment...")
    emb = load_embeddings(
        cfg["data"]["image_emb_path"],
        cfg["data"]["text_emb_path"],
        expected_dim=cfg["data"]["embed_dim"],
    )
    assignment_blob = torch.load(args.assignment, map_location="cpu", weights_only=False)
    K = assignment_blob["K"]
    M = assignment_blob["M"]
    hard = assignment_blob["hard_assignment"]
    img_ids = assignment_blob["image_ids"]
    logger.info(f"K={K}, M={M}, N_items={hard.shape[0]}")

    output = args.output or str(out_dir / f"router_K{K}.pt")

    # 构造 dataset (query -> bucket(s))
    ds = RouterDataset(
        text_features=emb.text_features,
        text_image_ids=emb.text_image_ids,
        image_ids=img_ids,
        hard_assignment=hard,
    )
    logger.info(f"Router dataset size: {len(ds)}")
    # 限制只用前 N 条 query（避免 router 见过 test query）
    if args.n_train_queries is not None:
        from torch.utils.data import Subset
        n_limit = min(args.n_train_queries, len(ds))
        ds = Subset(ds, list(range(n_limit)))
        logger.info(f"Limited dataset to first {n_limit} queries (train-only)")
    elif args.use_all_queries:
        logger.warning(f"Using ALL {len(ds)} queries (--use-all-queries). "
                       f"确保 test 已分离, 否则泄漏!")
    else:
        raise SystemExit(
            "ERROR: 未指定 --n-train-queries 且未设 --use-all-queries.\n"
            "  单池数据集(text=[train;test]): 必须传 --n-train-queries N (只用前N个train)\n"
            "  已分离数据集: 传 --use-all-queries 明确允许.\n"
            "  此检查防止 router 训练泄漏 test query (历史 bug 修复)."
        )

    # train/val split
    n_val = int(cfg["router"]["val_split"] * len(ds))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["router"]["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["router"]["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # 模型
    model = RouterMLP(
        embed_dim=cfg["data"]["embed_dim"],
        K=K,
        hidden_dim=cfg["router"]["hidden_dim"],
        n_layers=cfg["router"]["n_layers"],
        dropout=cfg["router"]["dropout"],
    )

    device = str(get_device(cfg["data"]["device"]))
    logger.info(f"Training on {device}...")

    log = train_router(
        model, train_loader, val_loader,
        epochs=epochs,
        lr=cfg["router"]["lr"],
        weight_decay=cfg["router"]["weight_decay"],
        device=device,
        top_B_list=tuple(cfg["eval"]["top_B"]),
    )

    save_router(model, log, output)
    logger.info(f"✓ Saved to {output}")

    best = max(log, key=lambda x: x.get("recall@1", 0))
    logger.info(f"Best val: {json.dumps(best, indent=2)}")


if __name__ == "__main__":
    main()
