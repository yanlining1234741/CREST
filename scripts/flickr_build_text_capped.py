"""容量约束专用 build_text。
train target 的桶标签 = 它在 pool 容量约束 assignment 里的实际桶 (跟 GENIUS 对齐:
train target 用它在全 pool 索引里的实际 ID/桶)。

关键: train_buckets 的 -1 padding 用该行第一个有效桶填充,
因为 router 的 multi_label_ce 用 gather, -1 会被当成桶 K-1 (静默错误)。
重复有效桶不改变 "any-of-M 命中" 语义。
"""
import sys, argparse
import torch
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--assignment", required=True)
    args = ap.parse_args()

    raw = torch.load(f"{args.data}/text_raw.pt", weights_only=False)
    trimg = torch.load(f"{args.data}/train_image_embeddings.pt", weights_only=False)['features']
    pool = torch.load(f"{args.data}/image_embeddings.pt", weights_only=False)['features']
    asg = torch.load(args.assignment, map_location='cpu', weights_only=False)
    hard = asg['hard_assignment']   # [N_pool, max_m] 容量约束 (含 -1)
    K = int(asg['K']); M = int(asg['M'])

    # train target 是 pool 子集: 匹配每个 train target → pool 行号
    print("匹配 train target → pool 行号...")
    tn = F.normalize(trimg, dim=-1)
    pn = F.normalize(pool, dim=-1)
    train_pool_row = torch.empty(trimg.shape[0], dtype=torch.long)
    min_match = 1.0
    BS = 2000
    for i in range(0, trimg.shape[0], BS):
        sim = tn[i:i+BS] @ pn.t()
        mx, idx = sim.max(dim=1)
        train_pool_row[i:i+BS] = idx
        min_match = min(min_match, mx.min().item())
    print(f"  最差匹配相似度: {min_match:.4f} (应≈1.0, 否则 train 不在 pool)")
    assert min_match > 0.99, "train target 没有都在 pool 里!"

    # train target 的桶 = pool 里它那行的容量约束桶 (含 -1)
    train_img_buckets = hard[train_pool_row]   # [n_train_target, max_m]

    # ★ -1 padding 用该行第一个有效桶填充 (防 gather 静默错误)
    tib = train_img_buckets.clone()
    for r in range(tib.shape[0]):
        row = tib[r]
        valid = row[row >= 0]
        if len(valid) == 0:
            raise ValueError(f"train target {r} 没有有效桶")
        first_valid = valid[0]
        row[row < 0] = first_valid   # -1 → 第一个有效桶 (重复, 不改变命中语义)
        tib[r] = row
    assert (tib >= 0).all(), "还有 -1 残留"

    # train caption → 桶
    trct = raw['train_target_trainrow']
    train_cap_buckets = tib[trct]   # [n_train_cap, max_m]

    out_dict = {
        "train_features": raw['train_features'],
        "train_buckets": train_cap_buckets,
        "test_features": raw['test_features'],
        "K": K, "M": M,
    }
    if 'test_target_multi' in raw:
        out_dict['test_target_multi'] = raw['test_target_multi']
        n_avg = (raw['test_target_multi'] >= 0).sum(dim=1).float().mean().item()
        print(f"test multi-target: {raw['test_target_multi'].shape[0]} q, avg {n_avg:.2f} pos")
    else:
        out_dict['test_target_row'] = raw['test_target']
        print("test single-target")

    torch.save(out_dict, f"{args.data}/text_with_buckets.pt")
    print(f"train cap buckets {tuple(train_cap_buckets.shape)} (无-1, padding已填有效桶)")
    print(f"saved text_with_buckets.pt (K={K})")


if __name__ == "__main__":
    main()
