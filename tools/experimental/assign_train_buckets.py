"""用现有 Sinkhorn centroids 给 train image 分桶 (top-M 最近 centroid).
不重训 Sinkhorn, 直接 assign."""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignment", required=True, help="含 centroids 的 test assignment")
    ap.add_argument("--train-image-emb", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    asg = torch.load(args.assignment, map_location='cpu', weights_only=False)
    centroids = asg['centroids']     # (K, D)
    K = int(asg['K']); M = int(asg['M'])
    print(f"centroids: {centroids.shape}, K={K}, M={M}")

    train_img = torch.load(args.train_image_emb, weights_only=False)['features']  # (N, D)
    print(f"train_img: {train_img.shape}")

    # L2 normalize (跟 Sinkhorn 时一致)
    train_norm = train_img / train_img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cent_norm = centroids / centroids.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # 每个 train img 取 top-M 最近 centroid
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_norm = train_norm.to(device); cent_norm = cent_norm.to(device)
    hard_list = []
    B = 8192
    for s in range(0, train_norm.shape[0], B):
        e = min(s+B, train_norm.shape[0])
        sim = train_norm[s:e] @ cent_norm.t()    # (b, K)
        topM = sim.topk(M, dim=-1).indices.cpu()  # (b, M)
        hard_list.append(topM)
    hard = torch.cat(hard_list, dim=0)   # (N, M)
    print(f"train hard_assignment: {hard.shape}")

    out = {'hard_assignment': hard, 'centroids': centroids, 'K': K, 'M': M,
           'note': 'train image bucket assignment via nearest centroids'}
    torch.save(out, args.output)
    print(f"Saved {args.output}")

if __name__=="__main__": main()
