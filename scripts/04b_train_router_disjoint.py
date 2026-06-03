"""Router 训练 - disjoint 模式. 吃 text_with_buckets.pt 的 (train_features, train_buckets)."""
import argparse, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.router import RouterDatasetDisjoint, RouterMLP, save_router, train_router
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--text-buckets", required=True, help="text_with_buckets.pt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config); set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "04b_router_disjoint.log"))
    epochs = args.epochs or cfg["router"]["epochs"]
    device = str(get_device(cfg["data"]["device"]))

    blob = torch.load(args.text_buckets, map_location="cpu", weights_only=False)
    train_feat = blob["train_features"]
    train_buckets = blob["train_buckets"]   # [Q, M]
    K = int(blob["K"])
    logger.info(f"Disjoint router: train queries={train_feat.shape[0]}, K={K}, M={train_buckets.shape[1]}")

    ds = RouterDatasetDisjoint(train_feat, train_buckets, K)
    n_val = int(cfg["router"]["val_split"] * len(ds))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"]))
    train_loader = DataLoader(train_ds, batch_size=cfg["router"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["router"]["batch_size"], shuffle=False)

    model = RouterMLP(embed_dim=cfg["data"]["embed_dim"], K=K,
        hidden_dim=cfg["router"]["hidden_dim"], n_layers=cfg["router"]["n_layers"],
        dropout=cfg["router"]["dropout"])
    logger.info(f"Training on {device}...")
    log = train_router(model, train_loader, val_loader, epochs=epochs,
        lr=cfg["router"]["lr"], weight_decay=cfg["router"]["weight_decay"],
        top_B_list=tuple(cfg["eval"]["top_B"]), device=device)
    save_router(model, log, args.output)
    logger.info(f"Saved router to {args.output}")

if __name__ == "__main__":
    main()
