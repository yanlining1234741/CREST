"""Sinkhorn 后: train 图分桶, 合并 train+test caption. 支持单/多 target."""
import sys, json, argparse
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--centroids-from", required=True)
    args=ap.parse_args()

    raw=torch.load(f"{args.data}/text_raw.pt", weights_only=False)
    trimg=torch.load(f"{args.data}/train_image_embeddings.pt", weights_only=False)['features']
    asg=torch.load(args.assignment, map_location='cpu', weights_only=False)
    cent=asg['centroids']
    M=asg['M']; K=asg['K']

    # train target -> 桶
    tn=torch.nn.functional.normalize(trimg,dim=-1)
    cn=torch.nn.functional.normalize(cent,dim=-1)
    sim=tn @ cn.t()
    train_img_buckets = sim.topk(M, dim=-1).indices

    # train caption -> 桶 (via train_target_trainrow)
    trct=raw['train_target_trainrow']
    train_cap_buckets = train_img_buckets[trct]

    # test target: 支持单/多
    out_dict = {
        "train_features": raw['train_features'],
        "train_buckets": train_cap_buckets,
        "test_features": raw['test_features'],
        "K": K, "M": M,
    }
    if 'test_target_multi' in raw:
        # 多 target: (n_test, max_pos) with -1 padding
        out_dict['test_target_multi'] = raw['test_target_multi']
        n_test = raw['test_target_multi'].shape[0]
        n_avg = (raw['test_target_multi'] >= 0).sum(dim=1).float().mean().item()
        print(f"test multi-target mode: {n_test} queries, avg {n_avg:.2f} pos/query")
    else:
        # 单 target
        out_dict['test_target_row'] = raw['test_target']
        print(f"test single-target mode")
    
    torch.save(out_dict, f"{args.data}/text_with_buckets.pt")
    print(f"train cap {raw['train_features'].shape} buckets {train_cap_buckets.shape}")
    print(f"test cap {raw['test_features'].shape}")
    print("saved text_with_buckets.pt")

if __name__=="__main__": main()
