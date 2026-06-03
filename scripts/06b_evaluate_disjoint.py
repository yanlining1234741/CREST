"""Disjoint eval, 支持单/多 target + CE rerank."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.router import load_router
from src.cross_encoder import CrossEncoderRerank
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--text-buckets", required=True)
    ap.add_argument("--image-emb", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rerank-mode", default="flat", choices=["flat", "cross_encoder"])
    ap.add_argument("--cross-encoder", type=str, default=None)
    ap.add_argument("--ce-hidden-dim", type=int, default=512)
    ap.add_argument("--ce-n-layers", type=int, default=3)
    ap.add_argument("--ce-batch", type=int, default=4096)
    args = ap.parse_args()

    cfg = load_config(args.config); set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "06b_evaluate_disjoint.log"))
    device = str(get_device(cfg["data"]["device"]))
    top_B_list = tuple(cfg["eval"]["top_B"])
    top_K_list = tuple(cfg["eval"]["top_K"])

    tb = torch.load(args.text_buckets, map_location="cpu", weights_only=False)
    test_feat = tb["test_features"].to(device)
    K = int(tb["K"])

    multi_mode = 'test_target_multi' in tb
    if multi_mode:
        tt = tb['test_target_multi'].numpy()
        Q = tt.shape[0]
        test_tgt_sets = [set(int(x) for x in row if x >= 0) for row in tt]
    else:
        test_tgt = tb['test_target_row'].numpy()
        Q = len(test_tgt)
        test_tgt_sets = [{int(test_tgt[i])} for i in range(Q)]

    asg = torch.load(args.assignment, map_location='cpu', weights_only=False)
    hard = asg['hard_assignment']
    M = int(asg['M'])

    img = torch.load(args.image_emb, map_location='cpu', weights_only=False)
    pool_feat = img['features'].to(device)
    N_pool = pool_feat.shape[0]
    logger.info(f"Q={Q}, pool={N_pool}, K={K}, M={M}, mode={args.rerank_mode}")

    router = load_router(args.router, hidden_dim=cfg["router"]["hidden_dim"],
        n_layers=cfg["router"]["n_layers"], dropout=cfg["router"]["dropout"]).eval().to(device)

    # CE load
    ce = None
    if args.rerank_mode == "cross_encoder":
        assert args.cross_encoder is not None
        ce = CrossEncoderRerank(embed_dim=cfg["data"]["embed_dim"],
            hidden_dim=args.ce_hidden_dim, n_layers=args.ce_n_layers).to(device)
        ck = torch.load(args.cross_encoder, map_location=device, weights_only=False)
        ce.load_state_dict(ck.get('state_dict', ck))
        ce.eval()
        logger.info(f"CE loaded from {args.cross_encoder}")

    # reverse index
    hard_np = hard.numpy()
    flat_b = hard_np.flatten()
    flat_r = np.arange(N_pool).repeat(M).astype(np.int64)
    order = flat_b.argsort(kind='stable')
    sorted_b = flat_b[order]; sorted_r = flat_r[order]
    offsets = np.zeros(K+1, dtype=np.int64)
    uniq, cnt = np.unique(sorted_b, return_counts=True)
    offsets[uniq+1] = cnt
    offsets = offsets.cumsum()

    with torch.no_grad():
        logits = router(test_feat)
        topB = logits.topk(max(top_B_list), dim=-1).indices.cpu().numpy()

    target_bucket_sets = []
    for q in range(Q):
        bset = set()
        for tgt_row in test_tgt_sets[q]:
            bset.update(int(b) for b in hard_np[tgt_row])
        target_bucket_sets.append(bset)

    router_hit = {b: 0 for b in top_B_list}
    cand_count = {b: 0 for b in top_B_list}
    recall_hit = {(b, k): 0 for b in top_B_list for k in top_K_list}

    logger.info(f"Eval {Q} queries, rerank={args.rerank_mode}...")
    for q in tqdm(range(Q)):
        tgt_set = test_tgt_sets[q]
        tbset = target_bucket_sets[q]
        qf = test_feat[q:q+1]
        for b in top_B_list:
            sel = topB[q, :b]
            if any(int(s) in tbset for s in sel):
                router_hit[b] += 1
            cand_list = []
            for bk in sel:
                s, e = offsets[bk], offsets[bk+1]
                if e > s: cand_list.append(sorted_r[s:e])
            if not cand_list: continue
            cand = np.unique(np.concatenate(cand_list))
            cand_count[b] += len(cand)
            ct = torch.from_numpy(cand).to(device)
            cf = pool_feat[ct]

            with torch.no_grad():
                if args.rerank_mode == "cross_encoder" and ce is not None:
                    nc = cf.shape[0]
                    q_exp = qf.expand(nc, -1)
                    scores = []
                    for i in range(0, nc, args.ce_batch):
                        s_i = ce(q_exp[i:i+args.ce_batch], cf[i:i+args.ce_batch])
                        scores.append(s_i)
                    sim = torch.cat(scores, dim=0)
                else:
                    sim = (qf @ cf.t()).squeeze(0)
            ranked = cand[sim.argsort(descending=True).cpu().numpy()]
            for k in top_K_list:
                if set(ranked[:k].tolist()) & tgt_set:
                    recall_hit[(b, k)] += 1

    out = {}
    for b in top_B_list:
        out[f"router_recall@{b}"] = router_hit[b]/Q
        out[f"candidates@B={b}"] = cand_count[b]/Q
        for k in top_K_list:
            out[f"recall@{k}|B={b}"] = recall_hit[(b,k)]/Q
    out["n_queries"] = Q; out["pool_size"] = N_pool
    out["multi_target"] = multi_mode
    out["rerank_mode"] = args.rerank_mode
    with open(args.output, "w") as f: json.dump(out, f, indent=2)
    logger.info(f"Saved {args.output}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
