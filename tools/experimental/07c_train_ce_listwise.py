"""CE Listwise rerank (disjoint) - FAST.
预计算 cand pool, 避免 __getitem__ 内 np.concatenate/unique."""
import argparse, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cross_encoder import CrossEncoderRerank, save_cross_encoder
from src.router import load_router
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger


class CEDatasetListwise(Dataset):
    def __init__(self, train_q, train_target_row, train_img, test_img,
                 router_buckets, bucket_to_test_rows, n_cand=200, seed=42):
        self.q = train_q
        self.tgt_row = train_target_row.long().numpy()
        self.tgt_emb = train_img
        self.test_emb = test_img
        self.N_test = test_img.shape[0]
        self.n_cand = n_cand
        self.n_neg = n_cand - 1
        self.rng = np.random.default_rng(seed)
        
        # ★ 预计算: 每个 query 的 cand pool (合并 top-B 桶 + unique), 一次性做完
        print(f"[Dataset] Pre-computing cand pool for {len(self.q)} queries...", flush=True)
        t0 = time.time()
        self.cand_pools = []
        for i in range(len(self.q)):
            pool = []
            for b in router_buckets[i]:
                pool.append(bucket_to_test_rows[int(b)])
            self.cand_pools.append(np.unique(np.concatenate(pool)) if pool else np.arange(self.N_test, dtype=np.int64))
        print(f"[Dataset] Pre-compute done in {time.time()-t0:.1f}s")
        avg_pool = np.mean([len(p) for p in self.cand_pools])
        print(f"[Dataset] Avg cand pool size: {avg_pool:.1f}, n_neg per sample: {self.n_neg}")

    def __len__(self): return self.q.shape[0]

    def __getitem__(self, idx):
        q = self.q[idx]
        target_emb = self.tgt_emb[self.tgt_row[idx]]
        pool = self.cand_pools[idx]
        n_actual = min(self.n_neg, len(pool))
        if n_actual > 0:
            neg_rows = self.rng.choice(pool, size=n_actual, replace=False)
        else:
            neg_rows = np.array([], dtype=np.int64)
        if n_actual < self.n_neg:
            extra = self.rng.integers(0, self.N_test, size=self.n_neg - n_actual)
            neg_rows = np.concatenate([neg_rows, extra])
        neg_embs = self.test_emb[neg_rows]
        all_cand = torch.cat([target_emb.unsqueeze(0), neg_embs], dim=0)
        return q, all_cand


def train_listwise(model, train_loader, val_loader, epochs, lr, device, verbose=True):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log = []
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0; total_n = 0
        t0 = time.time()
        for q, cands in train_loader:
            q = q.to(device, non_blocking=True)
            cands = cands.to(device, non_blocking=True)
            scores = model(q, cands)
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=device)
            loss = F.cross_entropy(scores, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * q.size(0)
            total_n += q.size(0)
        train_loss = total_loss / total_n
        model.eval()
        val_hit = 0; val_n = 0
        with torch.no_grad():
            for q, cands in val_loader:
                q = q.to(device, non_blocking=True)
                cands = cands.to(device, non_blocking=True)
                scores = model(q, cands)
                val_hit += (scores.argmax(dim=-1) == 0).sum().item()
                val_n += q.size(0)
        val_acc = val_hit / val_n
        scheduler.step()
        log.append({"epoch": ep, "train_loss": train_loss, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0]})
        if verbose:
            print(f"[CE-Listwise Ep {ep}] loss={train_loss:.4f} val_top1={val_acc:.4f} ({time.time()-t0:.1f}s)", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--text-buckets", required=True)
    ap.add_argument("--image-emb", required=True)
    ap.add_argument("--train-image-emb", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-train-queries", type=int, default=20000)
    ap.add_argument("--n-cand", type=int, default=100)
    ap.add_argument("--top-B", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-split", type=float, default=0.1)
    args = ap.parse_args()

    cfg = load_config(args.config); set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "07c_train_ce_listwise.log"))
    device = str(get_device(cfg["data"]["device"]))

    asg = torch.load(args.assignment, map_location='cpu', weights_only=False)
    hard = asg['hard_assignment']
    K = int(asg['K']); M = int(asg['M'])

    tb = torch.load(args.text_buckets, map_location='cpu', weights_only=False)
    train_q = tb['train_features']
    n_q = min(args.n_train_queries, train_q.shape[0])
    train_q = train_q[:n_q]
    data_dir = Path(args.text_buckets).parent
    raw = torch.load(f"{data_dir}/text_raw.pt", weights_only=False)
    train_target_row = raw['train_target_trainrow'][:n_q]
    train_img = torch.load(args.train_image_emb, weights_only=False)['features']
    test_img = torch.load(args.image_emb, weights_only=False)['features']
    logger.info(f"train_q={train_q.shape}, train_target={train_img.shape}, test_pool={test_img.shape}")
    logger.info(f"K={K}, M={M}, top_B={args.top_B}, n_cand={args.n_cand}")

    # Router predict top-B 桶
    logger.info(f"Computing top-{args.top_B} buckets per query...")
    router = load_router(args.router, hidden_dim=cfg["router"]["hidden_dim"],
        n_layers=cfg["router"]["n_layers"], dropout=cfg["router"]["dropout"]).eval().to(device)
    with torch.no_grad():
        rb_chunks = []
        for s in range(0, n_q, 4096):
            e = min(s + 4096, n_q)
            logits = router(train_q[s:e].to(device))
            rb_chunks.append(logits.topk(args.top_B, dim=-1).indices.cpu())
        router_buckets = torch.cat(rb_chunks, dim=0).numpy()
    del router; torch.cuda.empty_cache()
    logger.info(f"Router buckets: {router_buckets.shape}")

    # Bucket → test rows
    logger.info(f"Building test bucket index...")
    hard_np = hard.numpy()
    bucket_to_rows = [[] for _ in range(K)]
    for r in range(test_img.shape[0]):
        for b in hard_np[r]:
            bucket_to_rows[int(b)].append(r)
    bucket_to_rows = [np.array(x, dtype=np.int64) for x in bucket_to_rows]

    ds = CEDatasetListwise(train_q, train_target_row, train_img, test_img,
                           router_buckets, bucket_to_rows, n_cand=args.n_cand, seed=cfg["seed"])
    val_size = int(len(ds) * args.val_split)
    train_size = len(ds) - val_size
    train_set, val_set = random_split(ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = CrossEncoderRerank(embed_dim=cfg["data"]["embed_dim"],
        hidden_dim=args.hidden_dim, n_layers=args.n_layers, dropout=args.dropout)
    logger.info(f"Training listwise on {device}...")
    train_log = train_listwise(model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device, verbose=True)
    save_cross_encoder(model, train_log, args.output)
    logger.info(f"Saved to {args.output}")
    print(f"Final val_top1 = {train_log[-1]['val_acc']:.4f}")


if __name__ == "__main__":
    main()
