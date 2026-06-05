"""Bi-linear scorer 训练: score = q^T W v + b, W 初始化为 I.
文档建议的中间方案, 验证 bucket-local hard neg 能否涨点."""
import argparse, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cross_encoder import BiLinearScorer, save_cross_encoder
from src.router import load_router
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger


class CEDatasetListwise(Dataset):
    def __init__(self, train_q, train_target_row, train_img, test_img,
                 router_buckets, bucket_to_test_rows, n_cand=100, seed=42):
        self.q = train_q
        self.tgt_row = train_target_row.long().numpy()
        self.tgt_emb = train_img
        self.test_emb = test_img
        self.N_test = test_img.shape[0]
        self.n_cand = n_cand; self.n_neg = n_cand - 1
        self.rng = np.random.default_rng(seed)
        print(f"[Dataset] Pre-computing cand pool for {len(self.q)} queries...", flush=True)
        t0 = time.time()
        self.cand_pools = []
        for i in range(len(self.q)):
            pool = []
            for b in router_buckets[i]:
                pool.append(bucket_to_test_rows[int(b)])
            self.cand_pools.append(np.unique(np.concatenate(pool)) if pool else np.arange(self.N_test, dtype=np.int64))
        print(f"[Dataset] done {time.time()-t0:.1f}s", flush=True)

    def __len__(self): return self.q.shape[0]
    def __getitem__(self, idx):
        q = self.q[idx]
        target_emb = self.tgt_emb[self.tgt_row[idx]]
        pool = self.cand_pools[idx]
        n_actual = min(self.n_neg, len(pool))
        neg_rows = self.rng.choice(pool, size=n_actual, replace=False) if n_actual>0 else np.array([],dtype=np.int64)
        if n_actual < self.n_neg:
            extra = self.rng.integers(0, self.N_test, size=self.n_neg - n_actual)
            neg_rows = np.concatenate([neg_rows, extra])
        neg_embs = self.test_emb[neg_rows]
        all_cand = torch.cat([target_emb.unsqueeze(0), neg_embs], dim=0)
        return q, all_cand


def train(model, train_loader, val_loader, epochs, lr, device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log = []
    for ep in range(1, epochs+1):
        model.train(); tot=0; n=0; t0=time.time()
        for q, cands in train_loader:
            q=q.to(device); cands=cands.to(device)
            scores = model(q, cands)
            loss = F.cross_entropy(scores, torch.zeros(scores.size(0),dtype=torch.long,device=device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tot+=loss.item()*q.size(0); n+=q.size(0)
        model.eval(); hit=0; vn=0
        with torch.no_grad():
            for q,cands in val_loader:
                q=q.to(device); cands=cands.to(device)
                hit+=(model(q,cands).argmax(-1)==0).sum().item(); vn+=q.size(0)
        sched.step()
        log.append({"epoch":ep,"train_loss":tot/n,"val_acc":hit/vn})
        print(f"[BiLinear Ep {ep}] loss={tot/n:.4f} val_top1={hit/vn:.4f} ({time.time()-t0:.1f}s)",flush=True)
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
    ap.add_argument("--n-train-queries", type=int, default=30000)
    ap.add_argument("--n-cand", type=int, default=100)
    ap.add_argument("--top-B", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--low-rank", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config); set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir/"07g_train_bilinear.log"))
    device = str(get_device(cfg["data"]["device"]))

    asg = torch.load(args.assignment, map_location='cpu', weights_only=False)
    hard = asg['hard_assignment']; K=int(asg['K'])
    tb = torch.load(args.text_buckets, map_location='cpu', weights_only=False)
    train_q = tb['train_features']
    n_q = min(args.n_train_queries, train_q.shape[0]); train_q = train_q[:n_q]
    raw = torch.load(f"{Path(args.text_buckets).parent}/text_raw.pt", weights_only=False)
    train_target_row = raw['train_target_trainrow'][:n_q]
    train_img = torch.load(args.train_image_emb, weights_only=False)['features']
    test_img = torch.load(args.image_emb, weights_only=False)['features']
    logger.info(f"train_q={train_q.shape}, K={K}")

    router = load_router(args.router, hidden_dim=cfg["router"]["hidden_dim"],
        n_layers=cfg["router"]["n_layers"], dropout=cfg["router"]["dropout"]).eval().to(device)
    with torch.no_grad():
        rb=[]
        for s in range(0,n_q,4096):
            e=min(s+4096,n_q)
            rb.append(router(train_q[s:e].to(device)).topk(args.top_B,dim=-1).indices.cpu())
        router_buckets=torch.cat(rb,dim=0).numpy()
    del router; torch.cuda.empty_cache()

    hard_np=hard.numpy()
    b2r=[[] for _ in range(K)]
    for r in range(test_img.shape[0]):
        for b in hard_np[r]: b2r[int(b)].append(r)
    b2r=[np.array(x,dtype=np.int64) for x in b2r]

    ds = CEDatasetListwise(train_q, train_target_row, train_img, test_img,
                           router_buckets, b2r, n_cand=args.n_cand, seed=cfg["seed"])
    vs=int(len(ds)*0.1); ts=len(ds)-vs
    tr,va=random_split(ds,[ts,vs],generator=torch.Generator().manual_seed(cfg["seed"]))
    tl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    vl=DataLoader(va,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True)

    model = BiLinearScorer(embed_dim=768, init_identity=True, low_rank=args.low_rank)
    logger.info(f"Training BiLinear (low_rank={args.low_rank}) on {device}...")
    tlog = train(model, tl, vl, args.epochs, args.lr, device)
    save_cross_encoder(model, tlog, args.output)
    logger.info(f"Saved {args.output}")
    print(f"Final val_top1={tlog[-1]['val_acc']:.4f}")


if __name__=="__main__": main()
