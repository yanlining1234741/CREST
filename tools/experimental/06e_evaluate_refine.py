"""Eval with Query Refinement CE."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.router import load_router
from src.cross_encoder import CEQueryRefinement
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
    ap.add_argument("--hidden-dim", type=int, default=512)
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
    hard = asg['hard_assignment']; M = int(asg['M'])
    img = torch.load(args.image_emb, map_location='cpu', weights_only=False)
    pool_feat = img['features'].to(device); N_pool = pool_feat.shape[0]
    print(f"Q={Q}, pool={N_pool}, K={K}, M={M}, mode=refine")

    router = load_router(args.router, hidden_dim=cfg["router"]["hidden_dim"],
        n_layers=cfg["router"]["n_layers"], dropout=cfg["router"]["dropout"]).eval().to(device)
    ce = CEQueryRefinement(embed_dim=cfg["data"]["embed_dim"], hidden_dim=args.hidden_dim).to(device)
    ck = torch.load(args.cross_encoder, map_location=device, weights_only=False)
    ce.load_state_dict(ck.get('state_dict', ck)); ce.eval()
    scale_val = 0.1 * torch.sigmoid(ce.scale).item()
    print(f"Refine scale = {scale_val:.4f}")

    hard_np = hard.numpy()
    flat_b = hard_np.flatten()
    flat_r = np.arange(N_pool).repeat(M).astype(np.int64)
    order = flat_b.argsort(kind='stable')
    sorted_b = flat_b[order]; sorted_r = flat_r[order]
    offsets = np.zeros(K+1, dtype=np.int64)
    uniq, cnt = np.unique(sorted_b, return_counts=True)
    offsets[uniq+1] = cnt; offsets = offsets.cumsum()

    with torch.no_grad():
        logits = router(test_feat)
        topB = logits.topk(max(top_B_list), dim=-1).indices.cpu().numpy()

    tgt_bsets = []
    for q in range(Q):
        bs = set()
        for tr in tgt_sets[q]:
            bs.update(int(b) for b in hard_np[tr])
        tgt_bsets.append(bs)

    rh = {b: 0 for b in top_B_list}
    cc = {b: 0 for b in top_B_list}
    recall_hit = {(b, k): 0 for b in top_B_list for k in top_K_list}

    for q in tqdm(range(Q)):
        ts = tgt_sets[q]; tbs = tgt_bsets[q]
        qf = test_feat[q:q+1]
        for b in top_B_list:
            sel = topB[q, :b]
            if any(int(s) in tbs for s in sel): rh[b] += 1
            cand_list = []
            for bk in sel:
                s, e = offsets[bk], offsets[bk+1]
                if e > s: cand_list.append(sorted_r[s:e])
            if not cand_list: continue
            cand = np.unique(np.concatenate(cand_list))
            cc[b] += len(cand)
            ct = torch.from_numpy(cand).to(device)
            cf = pool_feat[ct].unsqueeze(0)  # (1, n_cand, D)
            with torch.no_grad():
                sim = ce(qf, cf).squeeze(0)
            ranked = cand[sim.argsort(descending=True).cpu().numpy()]
            for k in top_K_list:
                if set(ranked[:k].tolist()) & ts:
                    recall_hit[(b, k)] += 1

    out = {}
    for b in top_B_list:
        out[f"router_recall@{b}"] = rh[b]/Q
        out[f"candidates@B={b}"] = cc[b]/Q
        for k in top_K_list:
            out[f"recall@{k}|B={b}"] = recall_hit[(b,k)]/Q
    out["n_queries"] = Q; out["pool_size"] = N_pool
    out["multi_target"] = multi; out["rerank_mode"] = "refine"
    out["scale"] = scale_val
    with open(args.output, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
