"""
CE Filter eval — 容量约束版 (capped)。
= 06g_evaluate_ce_filter.py, 只改 2 处:
  (1) M 用 hard.shape[1] (容量约束 hard 是 [N, max_m] 含 -1 padding)
  (2) 反向索引过滤 -1
其余完全相同。不影响原 06g。
"""
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
    ap.add_argument("--cross-encoder", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-filters", type=str, default="50,100,150,200")
    ap.add_argument("--ce-batch", type=int, default=4096)
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    ensure_dir(cfg["output_dir"])
    device = str(get_device(cfg["data"]["device"]))
    top_B_list = tuple(cfg["eval"]["top_B"])
    top_K_list = tuple(cfg["eval"]["top_K"])

    tb = torch.load(args.text_buckets, map_location="cpu", weights_only=False)
    test_feat = tb["test_features"].to(device); K = int(tb["K"])
    multi = 'test_target_multi' in tb
    if multi:
        tt = tb['test_target_multi'].numpy(); Q = tt.shape[0]
        tgt_sets = [set(int(x) for x in row if x >= 0) for row in tt]
    else:
        ttt = tb['test_target_row'].numpy(); Q = len(ttt)
        tgt_sets = [{int(ttt[i])} for i in range(Q)]

    asg = torch.load(args.assignment, map_location='cpu', weights_only=False)
    hard = asg['hard_assignment']
    M = hard.shape[1]                          # ★改1: 用真实列数, 不用 asg['M']
    img = torch.load(args.image_emb, map_location='cpu', weights_only=False)
    pool_feat = img['features'].to(device); N_pool = pool_feat.shape[0]
    print(f"Q={Q}, pool={N_pool}, K={K}, hard_cols={M}, mode=ce_filter_CAPPED")

    router = load_router(args.router, hidden_dim=cfg["router"]["hidden_dim"],
        n_layers=cfg["router"]["n_layers"], dropout=cfg["router"]["dropout"]).eval().to(device)
    ce = CrossEncoderRerank(embed_dim=768, hidden_dim=512, n_layers=3).to(device)
    ck = torch.load(args.cross_encoder, map_location=device, weights_only=False)
    ce.load_state_dict(ck.get('state_dict', ck)); ce.eval()
    print(f"CE loaded from {args.cross_encoder}")

    hard_np = hard.numpy()
    flat_b = hard_np.flatten()
    flat_r = np.arange(N_pool).repeat(M).astype(np.int64)
    valid = flat_b >= 0                        # ★改2: 过滤 -1 padding
    flat_b = flat_b[valid]; flat_r = flat_r[valid]
    order = flat_b.argsort(kind='stable')
    sorted_b = flat_b[order]; sorted_r = flat_r[order]
    offsets = np.zeros(K+1, dtype=np.int64)
    uniq, cnt = np.unique(sorted_b, return_counts=True)
    offsets[uniq+1] = cnt; offsets = offsets.cumsum()

    with torch.no_grad():
        logits = router(test_feat)
        topB_idx = logits.topk(max(top_B_list), dim=-1).indices.cpu().numpy()

    n_filters = [int(x) for x in args.n_filters.split(",")]
    all_results = {}

    for N_filter in n_filters:
        print(f"\n>>> N_filter = {N_filter}")
        recall_hit = {(b, k): 0 for b in top_B_list for k in top_K_list}
        cc_before = {b: 0 for b in top_B_list}
        cc_after = {b: 0 for b in top_B_list}

        for q in tqdm(range(Q), desc=f"N={N_filter}"):
            ts = tgt_sets[q]; qf = test_feat[q:q+1]
            for b in top_B_list:
                sel = topB_idx[q, :b]
                cand_list = []
                for bk in sel:
                    s, e = offsets[bk], offsets[bk+1]
                    if e > s: cand_list.append(sorted_r[s:e])
                if not cand_list: continue
                cand = np.unique(np.concatenate(cand_list))
                cc_before[b] += len(cand)
                ct = torch.from_numpy(cand).to(device)
                cf = pool_feat[ct]

                with torch.no_grad():
                    if cf.shape[0] <= N_filter:
                        cand_filtered = cand
                        cf_filtered = cf
                    else:
                        ce_chunks = []
                        for i in range(0, cf.shape[0], args.ce_batch):
                            end = min(i + args.ce_batch, cf.shape[0])
                            q_exp = qf.expand(end - i, -1)
                            ce_chunks.append(ce(q_exp, cf[i:end]))
                        ce_s = torch.cat(ce_chunks)
                        keep_idx = ce_s.topk(N_filter).indices.cpu().numpy()
                        cand_filtered = cand[keep_idx]
                        cf_filtered = pool_feat[torch.from_numpy(cand_filtered).to(device)]

                    cc_after[b] += len(cand_filtered)
                    cos_s = (qf @ cf_filtered.t()).squeeze(0)

                ranked = cand_filtered[cos_s.argsort(descending=True).cpu().numpy()]
                for k in top_K_list:
                    if set(ranked[:k].tolist()) & ts:
                        recall_hit[(b, k)] += 1

        all_results[f"N_filter={N_filter}"] = {}
        for b in top_B_list:
            all_results[f"N_filter={N_filter}"][f"candidates_bucket@B={b}"] = cc_before[b]/Q
            all_results[f"N_filter={N_filter}"][f"candidates_cosine@B={b}"] = cc_after[b]/Q
            for k in top_K_list:
                all_results[f"N_filter={N_filter}"][f"recall@{k}|B={b}"] = recall_hit[(b,k)]/Q
    all_results["n_queries"] = Q
    all_results["pool_size"] = N_pool
    all_results["multi_target"] = multi
    all_results["rerank_mode"] = "ce_filter+cosine_CAPPED"
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"CE Filter (CAPPED) Results — N_filter sweep")
    print(f"{'='*80}")
    for B in top_B_list:
        bucket_cand = all_results[f'N_filter={n_filters[0]}'][f'candidates_bucket@B={B}']
        print(f"\nB={B} (桶内 cand ≈ {bucket_cand:.0f})")
        print(f"{'N_filter':>9} {'cosine算':>9} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
        for N in n_filters:
            v = all_results[f"N_filter={N}"]
            cos_cand = v[f'candidates_cosine@B={B}']
            r1 = v[f'recall@1|B={B}']*100
            r5 = v[f'recall@5|B={B}']*100
            r10 = v[f'recall@10|B={B}']*100
            print(f"{N:>9} {cos_cand:>9.0f} {r1:>8.2f} {r5:>8.2f} {r10:>8.2f}")


if __name__ == "__main__":
    main()
