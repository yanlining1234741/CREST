"""CE V6: Train cand = full router top-B 桶, 跟 eval 分布完全一致.
关键改动: __getitem__ 返回完整桶 cand, 不再 random sample."""
import argparse, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cross_encoder import CECrossAttention, save_cross_encoder
from src.router import load_router
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger


class CEDatasetFullBucket(Dataset):
    """每 sample 返回 (q, [target + 完整桶 cand 截断到 n_cand_max], label=0)"""
    def __init__(self, train_q, train_target_row, train_img, test_img,
                 router_buckets, bucket_to_test_rows, n_cand_max=500, seed=42):
        self.q = train_q
        self.tgt_row = train_target_row.long().numpy()
        self.tgt_emb = train_img
        self.test_emb = test_img
        self.N_test = test_img.shape[0]
        self.n_cand_max = n_cand_max
        self.rng = np.random.default_rng(seed)

        print(f"[Dataset] Building cand pools (full bucket)...", flush=True)
        t0 = time.time()
        self.cand_pools = []
        for i in range(len(self.q)):
            pool = []
            for b in router_buckets[i]:
                pool.append(bucket_to_test_rows[int(b)])
            full = np.unique(np.concatenate(pool)) if pool else np.arange(self.N_test, dtype=np.int64)
            # 截断到 n_cand_max - 1 (留 1 个位置给 target)
            if len(full) > n_cand_max - 1:
                # 随机采, 保留多样性
                full = self.rng.choice(full, size=n_cand_max - 1, replace=False)
            self.cand_pools.append(full.astype(np.int64))
        avg = np.mean([len(p) for p in self.cand_pools])
        print(f"[Dataset] Done in {time.time()-t0:.1f}s, avg cand: {avg:.1f} (max {n_cand_max-1})", flush=True)

    def __len__(self): return self.q.shape[0]

    def __getitem__(self, idx):
        q = self.q[idx]
        target_emb = self.tgt_emb[self.tgt_row[idx]]
        cands_rows = self.cand_pools[idx]
        # 如果 cand 数 < n_cand_max - 1, padding 随机
        if len(cands_rows) < self.n_cand_max - 1:
            pad = self.rng.integers(0, self.N_test, size=self.n_cand_max - 1 - len(cands_rows))
            cands_rows = np.concatenate([cands_rows, pad])
        cand_embs = self.test_emb[cands_rows]
        # target 在 idx 0
        all_cand = torch.cat([target_emb.unsqueeze(0), cand_embs], dim=0)
        return q, all_cand


def train_loop(model, train_loader, val_loader, epochs, lr, device, verbose=True):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log = []
    for ep in range(1, epochs+1):
        model.train()
        tot_loss = 0; tot_n = 0
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
            tot_loss += loss.item() * q.size(0); tot_n += q.size(0)
        train_loss = tot_loss / tot_n
        model.eval()
        hit = 0; n = 0
        with torch.no_grad():
            for q, cands in val_loader:
                q = q.to(device, non_blocking=True)
                cands = cands.to(device, non_blocking=True)
                scores = model(q, cands)
                hit += (scores.argmax(dim=-1) == 0).sum().item()
                n += q.size(0)
        val_acc = hit / n
        alpha_val = torch.sigmoid(model.alpha).item() if hasattr(model, 'alpha') else 0
        delta_val = 0.3 * np.tanh(model.alpha.item()) if hasattr(model, 'alpha') else 0
        sched.step()
        log.append({"epoch": ep, "train_loss": train_loss, "val_acc": val_acc,
                    "alpha_raw": model.alpha.item() if hasattr(model,'alpha') else 0,
                    "delta": delta_val, "lr": sched.get_last_lr()[0]})
        if verbose:
            print(f"[CE-FB Ep {ep}] loss={train_loss:.4f} val_top1={val_acc:.4f} delta={delta_val:+.3f} ({time.time()-t0:.1f}s)", flush=True)
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
    ap.add_argument("--n-cand", type=int, default=500)
    ap.add_argument("--top-B", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-split", type=float, default=0.1)
    args = ap.parse_args()

    cfg = load_config(args.config); set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "07e_train_ce_fullbucket.log"))
    device = str(get_device(cfg["data"]["device"]))

    asg = torch.load(args.assignment, map_location='cpu', weights_only=False)
    hard = asg['hard_assignment']
    K = int(asg['K'])

    tb = torch.load(args.text_buckets, map_location='cpu', weights_only=False)
    train_q = tb['train_features']
    n_q = min(args.n_train_queries, train_q.shape[0])
    train_q = train_q[:n_q]
    raw = torch.load(f"{Path(args.text_buckets).parent}/text_raw.pt", weights_only=False)
    train_target_row = raw['train_target_trainrow'][:n_q]
    train_img = torch.load(args.train_image_emb, weights_only=False)['features']
    test_img = torch.load(args.image_emb, weights_only=False)['features']
    logger.info(f"train_q={train_q.shape}, train_target={train_img.shape}, test_pool={test_img.shape}")
    logger.info(f"K={K}, top_B={args.top_B}, n_cand={args.n_cand}")

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

    hard_np = hard.numpy()
    bucket_to_rows = [[] for _ in range(K)]
    for r in range(test_img.shape[0]):
        for b in hard_np[r]:
            bucket_to_rows[int(b)].append(r)
    bucket_to_rows = [np.array(x, dtype=np.int64) for x in bucket_to_rows]

    ds = CEDatasetFullBucket(train_q, train_target_row, train_img, test_img,
                              router_buckets, bucket_to_rows, n_cand_max=args.n_cand, seed=cfg["seed"])
    val_size = int(len(ds) * args.val_split)
    train_size = len(ds) - val_size
    train_set, val_set = random_split(ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = CECrossAttention(embed_dim=cfg["data"]["embed_dim"],
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout)
    logger.info(f"Training v6 (full bucket) on {device}...")
    train_log = train_loop(model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device, verbose=True)
    save_cross_encoder(model, train_log, args.output)
    logger.info(f"Saved to {args.output}")
    print(f"Final val_top1 = {train_log[-1]['val_acc']:.4f}, delta = {train_log[-1]['delta']:+.3f}")


if __name__ == "__main__":
    main()
