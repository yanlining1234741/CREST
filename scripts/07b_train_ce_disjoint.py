"""CE disjoint: train on train_image_pool, eval against test_pool.
Hard negatives sampled from test_pool buckets (where eval happens).
"""
import argparse, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn.functional as F
import numpy as np
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cross_encoder import CrossEncoderRerank, train_cross_encoder, save_cross_encoder
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger


class CEDatasetDisjoint(Dataset):
    """
    每个 sample: (query feat, target feat_train, [n_hn 个 hard neg from test pool])
    
    Hard neg 怎么采: target 在 train_pool 找最近 centroid -> 桶 -> 从 test_pool 该桶里随机采
    这样 hard neg 跟 eval 时桶内 cand 同分布
    """
    def __init__(self, train_query_feat, train_target_trainrow,
                 train_image_feat, test_image_feat, hard_assignment_test,
                 centroids, n_neg=15, hn_ratio=0.7, seed=42):
        self.q = train_query_feat
        self.tgt_row = train_target_trainrow.long().numpy()
        self.tgt_emb = train_image_feat
        self.test_emb = test_image_feat
        self.N_test = test_image_feat.shape[0]
        self.n_neg = n_neg
        self.n_hn = int(n_neg * hn_ratio)
        self.n_rand = n_neg - self.n_hn

        # 每个 train target -> test 桶 (通过最近 centroid)
        print(f"[Dataset] Mapping {train_image_feat.shape[0]} train targets to test buckets...", flush=True)
        t0 = time.time()
        cent_n = F.normalize(centroids, dim=-1)
        train_n = F.normalize(train_image_feat, dim=-1)
        M = hard_assignment_test.shape[1]
        # 最近 M 个 centroid
        sim = train_n @ cent_n.t()
        train_target_buckets = sim.topk(M, dim=-1).indices.numpy()  # (N_train, M)
        print(f"[Dataset] Mapping done in {time.time()-t0:.1f}s")

        # 建 bucket -> test_rows 反向索引
        print(f"[Dataset] Building test pool reverse index...", flush=True)
        ha_np = hard_assignment_test.numpy()
        K = int(ha_np.max()) + 1
        bucket_to_rows = [[] for _ in range(K)]
        for r in range(self.N_test):
            for b in ha_np[r]:
                bucket_to_rows[int(b)].append(r)
        bucket_to_rows = [np.array(x, dtype=np.int64) for x in bucket_to_rows]
        self.bucket_to_rows = bucket_to_rows
        self.train_target_buckets = train_target_buckets
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        q = self.q[idx]
        tgt_train_row = self.tgt_row[idx]
        tgt = self.tgt_emb[tgt_train_row]  # train target embedding

        # Hard negatives: 从 target 所在桶 (test) 随机采
        buckets = self.train_target_buckets[tgt_train_row]
        candidates = []
        for b in buckets:
            candidates.extend(self.bucket_to_rows[int(b)])
        candidates = np.unique(candidates) if candidates else np.arange(self.N_test)
        n_hn_actual = min(self.n_hn, len(candidates))
        hn_idx = self.rng.choice(candidates, size=n_hn_actual, replace=False) if n_hn_actual>0 else np.array([], dtype=np.int64)

        # Random negatives: 全 test pool
        rn_idx = self.rng.integers(0, self.N_test, size=self.n_neg - n_hn_actual)
        all_neg_idx = np.concatenate([hn_idx, rn_idx])
        negs = self.test_emb[all_neg_idx]  # (n_neg, D)

        return q, tgt, negs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--text-buckets", required=True)
    ap.add_argument("--image-emb", required=True)
    ap.add_argument("--train-image-emb", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-train-queries", type=int, default=100000)
    ap.add_argument("--n-negatives", type=int, default=15)
    ap.add_argument("--hn-ratio", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-split", type=float, default=0.1)
    args = ap.parse_args()

    cfg = load_config(args.config); set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "07b_train_ce_disjoint.log"))

    asg = torch.load(args.assignment, map_location='cpu', weights_only=False)
    hard = asg['hard_assignment']
    cent = asg['centroids']
    K = int(asg['K']); M = int(asg['M'])

    tb = torch.load(args.text_buckets, map_location='cpu', weights_only=False)
    train_q = tb['train_features']
    n_q = min(args.n_train_queries, train_q.shape[0])
    train_q = train_q[:n_q]
    train_target_row = tb['train_buckets'][:n_q]  # NOT used here, we need train_target_trainrow
    
    # Actually need train_target_trainrow from text_raw.pt
    data_dir = Path(args.text_buckets).parent
    raw = torch.load(f"{data_dir}/text_raw.pt", weights_only=False)
    train_target_trainrow = raw['train_target_trainrow'][:n_q]

    train_img = torch.load(args.train_image_emb, weights_only=False)['features']
    test_img = torch.load(args.image_emb, weights_only=False)['features']
    logger.info(f"train_q={train_q.shape}, train_target_pool={train_img.shape}, test_pool={test_img.shape}")
    logger.info(f"K={K}, M={M}")

    ds = CEDatasetDisjoint(train_q, train_target_trainrow, train_img, test_img,
                           hard, cent, n_neg=args.n_negatives, hn_ratio=args.hn_ratio,
                           seed=cfg["seed"])
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
    device = str(get_device(cfg["data"]["device"]))
    logger.info(f"Training on {device}...")

    train_log = train_cross_encoder(model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=1e-5,
        temperature=args.temperature, device=device, verbose=True)
    save_cross_encoder(model, train_log, args.output)
    logger.info(f"Saved to {args.output}")
    print(f"Final val_acc = {train_log[-1]['val_acc']:.4f}")


if __name__ == "__main__":
    main()
