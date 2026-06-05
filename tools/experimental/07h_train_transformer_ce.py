"""标准 transformer cross-encoder 训练.
严格对齐文档 (生成式检索.md):
  - 2 层 transformer cross-encoder
  - bucket-local hard negative (从 target 桶采 k 个负样本)
  - pairwise loss: score(q,v_pos) > score(q,v_neg)
"""
import argparse, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cross_encoder import TransformerCrossEncoder, save_cross_encoder
from src.router import load_router
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger


class BucketLocalHardNegDataset(Dataset):
    """文档原版: 每个 query, v_pos + 从 target 所在桶采 k 个 hard neg.
    
    文档伪代码:
      target_bucket = pi(v_pos)
      candidates_in_bucket = pi^{-1}(target_bucket)
      hard_negs = sample(candidates_in_bucket - {v_pos}, k=7)
    """
    def __init__(self, train_q, train_target_row, train_img,
                 target_bucket_assignment, bucket_to_rows,
                 n_neg=7, seed=42):
        self.q = train_q
        self.tgt_row = train_target_row.long().numpy()
        self.tgt_emb = train_img       # train target pool (v_pos 来自这里)
        self.tgt_bucket = target_bucket_assignment   # (N_train_img, M) 每个 train img 的桶
        self.bucket_to_rows = bucket_to_rows          # bucket → train img rows
        self.n_neg = n_neg
        self.rng = np.random.default_rng(seed)
        self.N_train_img = train_img.shape[0]
        print(f"[Dataset] N_q={len(self.q)}, n_neg={n_neg} (bucket-local)", flush=True)

    def __len__(self): return self.q.shape[0]

    def __getitem__(self, idx):
        q = self.q[idx]
        v_pos_row = self.tgt_row[idx]
        v_pos = self.tgt_emb[v_pos_row]

        # v_pos 所在的桶 (取第一个桶)
        target_buckets = self.tgt_bucket[v_pos_row]   # (M,) 该 img 的 M 个桶
        # 从这些桶里采 hard neg
        pool = []
        for b in target_buckets:
            pool.append(self.bucket_to_rows[int(b)])
        pool = np.unique(np.concatenate(pool)) if pool else np.arange(self.N_train_img)
        # 排除 v_pos 自己
        pool = pool[pool != v_pos_row]

        n_actual = min(self.n_neg, len(pool))
        if n_actual > 0:
            neg_rows = self.rng.choice(pool, size=n_actual, replace=False)
        else:
            neg_rows = np.array([], dtype=np.int64)
        if n_actual < self.n_neg:
            extra = self.rng.integers(0, self.N_train_img, size=self.n_neg - n_actual)
            neg_rows = np.concatenate([neg_rows, extra])

        neg_embs = self.tgt_emb[neg_rows]                       # (n_neg, D)
        # 返回 [v_pos, neg_1, ..., neg_k], label=0
        all_cand = torch.cat([v_pos.unsqueeze(0), neg_embs], dim=0)  # (n_neg+1, D)
        return q, all_cand


def train(model, train_loader, val_loader, epochs, lr, device, loss_type='listwise'):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log = []
    for ep in range(1, epochs+1):
        model.train(); tot=0; n=0; t0=time.time()
        for q, cands in train_loader:
            q = q.to(device); cands = cands.to(device)
            scores = model(q, cands)   # (B, n_cand)
            if loss_type == 'listwise':
                # listwise: v_pos (idx 0) 应该分最高
                loss = F.cross_entropy(scores, torch.zeros(scores.size(0), dtype=torch.long, device=device))
            else:
                # pairwise margin: score(pos) - score(neg) > margin
                pos = scores[:, 0:1]              # (B,1)
                neg = scores[:, 1:]               # (B, n_neg)
                loss = F.relu(0.2 - (pos - neg)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()*q.size(0); n += q.size(0)
        model.eval(); hit=0; vn=0
        with torch.no_grad():
            for q, cands in val_loader:
                q=q.to(device); cands=cands.to(device)
                hit += (model(q,cands).argmax(-1)==0).sum().item(); vn += q.size(0)
        sched.step()
        log.append({"epoch":ep, "train_loss":tot/n, "val_acc":hit/vn})
        print(f"[TransCE Ep {ep}] loss={tot/n:.4f} val_top1={hit/vn:.4f} ({time.time()-t0:.1f}s)", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--text-buckets", required=True)
    ap.add_argument("--train-image-emb", required=True)
    ap.add_argument("--train-assignment", required=True, help="train image 的桶分配")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-train-queries", type=int, default=80000)
    ap.add_argument("--n-neg", type=int, default=7, help="文档默认 7")
    ap.add_argument("--epochs", type=int, default=15, help="文档说 15 epoch 收敛")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=2, help="文档: 2 层")
    ap.add_argument("--loss-type", type=str, default="listwise", choices=["listwise","pairwise"])
    args = ap.parse_args()

    cfg = load_config(args.config); set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir/"07h_train_transformer_ce.log"))
    device = str(get_device(cfg["data"]["device"]))

    tb = torch.load(args.text_buckets, map_location='cpu', weights_only=False)
    train_q = tb['train_features']
    n_q = min(args.n_train_queries, train_q.shape[0]); train_q = train_q[:n_q]
    raw = torch.load(f"{Path(args.text_buckets).parent}/text_raw.pt", weights_only=False)
    train_target_row = raw['train_target_trainrow'][:n_q]
    train_img = torch.load(args.train_image_emb, weights_only=False)['features']

    # train image 的桶分配 (从 train_assignment 读)
    train_asg = torch.load(args.train_assignment, map_location='cpu', weights_only=False)
    train_bucket = train_asg['hard_assignment'].numpy()   # (N_train_img, M)
    K = int(train_asg['K']); M = int(train_asg['M'])
    logger.info(f"train_q={train_q.shape}, train_img={train_img.shape}, K={K}, M={M}")

    # bucket → train img rows
    b2r = [[] for _ in range(K)]
    for r in range(train_img.shape[0]):
        for b in train_bucket[r]:
            b2r[int(b)].append(r)
    b2r = [np.array(x, dtype=np.int64) for x in b2r]

    ds = BucketLocalHardNegDataset(train_q, train_target_row, train_img,
                                    train_bucket, b2r, n_neg=args.n_neg, seed=cfg["seed"])
    vs = int(len(ds)*0.1); ts = len(ds)-vs
    tr, va = random_split(ds, [ts, vs], generator=torch.Generator().manual_seed(cfg["seed"]))
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = TransformerCrossEncoder(embed_dim=768, n_heads=args.n_heads, n_layers=args.n_layers)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"TransformerCrossEncoder: {n_params/1e6:.2f}M params, loss={args.loss_type}")
    print(f"模型参数量: {n_params/1e6:.2f}M (文档预期 3-10M)")

    tlog = train(model, tl, vl, args.epochs, args.lr, device, loss_type=args.loss_type)
    save_cross_encoder(model, tlog, args.output)
    logger.info(f"Saved {args.output}")
    print(f"Final val_top1={tlog[-1]['val_acc']:.4f}")


if __name__=="__main__": main()
