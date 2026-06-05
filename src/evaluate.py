"""完整 end-to-end 评估. OPTIMIZED for VN 542K pool."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .router import RouterMLP


@torch.no_grad()
def evaluate_end_to_end(
    router: RouterMLP,
    text_features: torch.Tensor,
    text_image_ids: torch.Tensor,
    image_features: torch.Tensor,
    image_ids: torch.Tensor,
    hard_assignment: torch.Tensor,
    top_B_list: Tuple[int, ...] = (1, 3, 5, 10),
    top_K_list: Tuple[int, ...] = (1, 5, 10),
    device: str = "cuda",
    candidate_pool_size: int = 5000,
    show_progress: bool = True,
    router_prior_lambda: float = 0.0,
    router_prior_agg: str = "max",
    rerank_mode: str = "flat",
    cross_encoder=None,
    ce_batch_size: int = 4096,
    ce_alpha: float = 0.3,
) -> Dict[str, float]:
    """优化版: 用 numpy 向量化替代 Python set / list."""
    router = router.eval().to(device)
    if cross_encoder is not None:
        cross_encoder = cross_encoder.eval().to(device)

    pool_size = min(candidate_pool_size, image_features.shape[0])
    pool_features = image_features[:pool_size].to(device)
    pool_ids = image_ids[:pool_size]
    pool_assign = hard_assignment[:pool_size]
    K = router.K

    print(f"[Eval] Building reverse index for K={K}, pool_size={pool_size}...", flush=True)
    
    pool_assign_np = pool_assign.numpy()
    N, M = pool_assign_np.shape
    flat_buckets = pool_assign_np.flatten()
    flat_rows = np.arange(N).repeat(M).astype(np.int64)
    sort_order = flat_buckets.argsort(kind='stable')
    sorted_buckets = flat_buckets[sort_order]
    sorted_rows = flat_rows[sort_order]
    
    bucket_offsets = np.zeros(K + 1, dtype=np.int64)
    unique_b, counts = np.unique(sorted_buckets, return_counts=True)
    bucket_offsets[unique_b + 1] = counts
    bucket_offsets = bucket_offsets.cumsum()
    
    print(f"[Eval] Reverse index built. sorted_rows size={len(sorted_rows)}", flush=True)

    pool_id_set = set(pool_ids.tolist())
    valid_query_mask = torch.tensor(
        [iid.item() in pool_id_set for iid in text_image_ids]
    )
    text_features = text_features[valid_query_mask]
    text_image_ids = text_image_ids[valid_query_mask]
    Q = text_features.shape[0]
    if Q == 0:
        raise ValueError("No queries left")

    pool_id_to_row = {iid.item(): i for i, iid in enumerate(pool_ids)}

    max_B = max(top_B_list)

    text_features_dev = text_features.to(device)
    logits = router(text_features_dev)
    topB_buckets = logits.topk(max_B, dim=-1).indices.cpu().numpy()

    if router_prior_lambda > 0:
        log_probs_all = F.log_softmax(logits, dim=-1).cpu().numpy()
    else:
        log_probs_all = None

    router_hit = {b: 0 for b in top_B_list}
    cand_count = {b: 0 for b in top_B_list}
    recall_hit = {(b, k): 0 for b in top_B_list for k in top_K_list}

    target_rows = np.array([pool_id_to_row[iid.item()] for iid in text_image_ids], dtype=np.int64)
    target_buckets_arr = pool_assign_np[target_rows]

    print(f"[Eval] Starting evaluation on {Q} queries...", flush=True)
    
    it = range(Q)
    if show_progress:
        it = tqdm(it, desc=f"Eval (mode={rerank_mode})")

    for q_idx in it:
        target_row = target_rows[q_idx]
        target_buckets_set = set(target_buckets_arr[q_idx].tolist())
        q_feat = text_features_dev[q_idx:q_idx + 1]

        for b in top_B_list:
            selected = topB_buckets[q_idx, :b]

            if any(int(s) in target_buckets_set for s in selected):
                router_hit[b] += 1

            cand_rows_list = []
            for bk in selected:
                start, end = bucket_offsets[bk], bucket_offsets[bk + 1]
                if end > start:
                    cand_rows_list.append(sorted_rows[start:end])
            
            if not cand_rows_list:
                continue
            
            cand_rows_np = np.unique(np.concatenate(cand_rows_list))
            cand_count[b] += len(cand_rows_np)

            cand_rows_t = torch.from_numpy(cand_rows_np).to(device)
            cand_feat = pool_features[cand_rows_t]

            cos_sim = (q_feat @ cand_feat.t()).squeeze(0)
            if rerank_mode == "cross_encoder" and cross_encoder is not None:
                n_cand = cand_feat.shape[0]
                q_expand = q_feat.expand(n_cand, -1)
                scores = []
                for i in range(0, n_cand, ce_batch_size):
                    s = cross_encoder(q_expand[i:i+ce_batch_size], cand_feat[i:i+ce_batch_size])
                    scores.append(s)
                sim = torch.cat(scores, dim=0)
            elif rerank_mode == "residual" and cross_encoder is not None:
                n_cand = cand_feat.shape[0]
                q_expand = q_feat.expand(n_cand, -1)
                ce_scores = []
                for i in range(0, n_cand, ce_batch_size):
                    ce_scores.append(
                        cross_encoder(q_expand[i:i+ce_batch_size], cand_feat[i:i+ce_batch_size])
                    )
                ce_sc = torch.cat(ce_scores, dim=0)
                ce_z = (ce_sc - ce_sc.mean()) / ce_sc.std().clamp_min(1e-6)
                sim = cos_sim + ce_alpha * ce_z
            else:
                sim = cos_sim

                if router_prior_lambda > 0:
                    cand_buckets_np = pool_assign_np[cand_rows_np]
                    q_log_probs_np = log_probs_all[q_idx]
                    cand_log_probs_per_view = q_log_probs_np[cand_buckets_np]
                    if router_prior_agg == "max":
                        cand_log_probs = cand_log_probs_per_view.max(axis=-1)
                    elif router_prior_agg == "mean":
                        cand_log_probs = cand_log_probs_per_view.mean(axis=-1)
                    else:
                        cand_log_probs = np.log(np.exp(cand_log_probs_per_view).sum(axis=-1))
                    cand_log_probs_t = torch.from_numpy(cand_log_probs).float().to(device)
                    sim = sim + router_prior_lambda * cand_log_probs_t

            order = sim.argsort(descending=True).cpu().numpy()
            ranked_rows_np = cand_rows_np[order]
            
            matches = np.where(ranked_rows_np == target_row)[0]
            if len(matches) > 0:
                rank = int(matches[0]) + 1
            else:
                rank = -1
            
            for k in top_K_list:
                if 0 < rank <= k:
                    recall_hit[(b, k)] += 1

    out = {}
    for b in top_B_list:
        out[f"router_recall@{b}"] = router_hit[b] / Q
        out[f"candidates@B={b}"] = cand_count[b] / Q
        for k in top_K_list:
            out[f"recall@{k}|B={b}"] = recall_hit[(b, k)] / Q
    out["n_queries_evaluated"] = Q
    out["pool_size"] = pool_size
    out["router_prior_lambda"] = router_prior_lambda
    out["rerank_mode"] = rerank_mode
    if rerank_mode == "residual":
        out["ce_alpha"] = ce_alpha
    return out


def format_eval_table(metrics, top_B_list, top_K_list):
    lines = []
    lines.append(f"# Evaluated on {metrics['n_queries_evaluated']} queries, pool size = {metrics['pool_size']}")
    lines.append(f"# rerank_mode = {metrics.get('rerank_mode', 'flat')}")
    lines.append("")
    lines.append(f"{'B':>4} {'router_R@B':>12} {'avg_cands':>10} " +
                 " ".join(f"{'R@'+str(k)+'|B':>10}" for k in top_K_list))
    lines.append("-" * 60)
    for b in top_B_list:
        row = (f"{b:>4} {metrics[f'router_recall@{b}']:>12.4f} "
               f"{metrics[f'candidates@B={b}']:>10.1f} ")
        row += " ".join(f"{metrics[f'recall@{k}|B={b}']:>10.4f}" for k in top_K_list)
        lines.append(row)
    return "\n".join(lines)
