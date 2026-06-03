"""训练 cross-encoder rerank. OPTIMIZED."""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_embeddings
from src.cross_encoder import (
    CrossEncoderRerank, CrossEncoderDataset,
    train_cross_encoder, save_cross_encoder
)
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-train-queries", type=int, default=100000)
    parser.add_argument("--n-negatives", type=int, default=15)
    parser.add_argument("--hn-ratio", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-hn-pool", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "07_train_cross_encoder.log"))

    logger.info("Loading embeddings...")
    emb = load_embeddings(
        cfg["data"]["image_emb_path"],
        cfg["data"]["text_emb_path"],
        expected_dim=cfg["data"]["embed_dim"],
    )
    assignment_blob = torch.load(args.assignment, map_location="cpu", weights_only=False)
    hard_assignment = assignment_blob["hard_assignment"]
    K = assignment_blob["K"]
    M = assignment_blob["M"]

    logger.info(f"K={K}, M={M}, N_items={emb.image_features.shape[0]}")
    
    n_q = min(args.n_train_queries, emb.text_features.shape[0])
    logger.info(f"Limited dataset to first {n_q} queries")
    
    train_text_feat = emb.text_features[:n_q]
    train_text_targets = emb.text_image_ids[:n_q]

    dataset = CrossEncoderDataset(
        text_features=train_text_feat,
        text_targets=train_text_targets,
        image_features=emb.image_features,
        hard_assignment=hard_assignment,
        n_negatives=args.n_negatives,
        hn_ratio=args.hn_ratio,
        seed=cfg["seed"],
        max_hn_per_target=args.max_hn_pool,
    )
    
    logger.info(f"Cross-encoder dataset size: {len(dataset)}")
    
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    
    # 关键: num_workers=0 (避免 fork dataset 复制内存)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    
    model = CrossEncoderRerank(
        embed_dim=cfg["data"]["embed_dim"],
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    
    device = str(get_device(cfg["data"]["device"]))
    logger.info(f"Training on {device}...")
    
    train_log = train_cross_encoder(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-5,
        temperature=args.temperature,
        device=device,
        verbose=True,
    )
    
    save_cross_encoder(model, train_log, args.output)
    logger.info(f"✓ Saved to {args.output}")
    
    final_acc = train_log[-1]['val_acc']
    print(f"\nFinal val_acc = {final_acc:.4f}")


if __name__ == "__main__":
    main()
