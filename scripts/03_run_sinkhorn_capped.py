"""03 容量约束版: 每桶硬卡 [L,U], K 自适应 (数据大多桶/小少桶)。

K 自适应: K = clip(round(pool*M / S), k_min, k_max), S=(L+U)/2
不影响原 03_run_sinkhorn.py。

用法:
  python scripts/03_run_sinkhorn_capped.py --config configs/nights_task4.yaml \
    --input outputs/nights_task4/query_aware_emb.pt \
    --L 300 --U 1500 --M 8 \
    --output outputs/nights_task4/assignment_capped.pt
"""
import argparse, json, sys, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.query_aware import load_query_aware
from src.sinkhorn import run_sinkhorn_kmeans_capped
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--U", type=int, required=True)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--K", type=int, default=None, help="不传则自适应")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "03_capped.log"))

    qa = load_query_aware(args.input)
    Z = qa.features
    N = Z.shape[0]
    L, U, M = args.L, args.U, args.M

    # K 自适应
    if args.K is not None:
        K = args.K
    else:
        S = (L + U) / 2
        k_raw = round(N * M / S)
        k_min = math.ceil(N * M / U)
        k_max = math.floor(N * M / L)
        K = max(k_min, min(k_raw, k_max))  # 落在可行范围
    print(f"pool={N}, M={M}, [L,U]=[{L},{U}] -> K={K} (目标桶大小≈{N*M/K:.0f})")
    logger.info(f"capped: pool={N}, M={M}, L={L}, U={U}, K={K}")

    device = str(get_device(cfg["data"]["device"]))
    result = run_sinkhorn_kmeans_capped(
        Z, K=K, M=M, L=L, U=U,
        epsilon=cfg["sinkhorn"]["epsilon"],
        n_sinkhorn_iters=cfg["sinkhorn"]["n_sinkhorn_iters"],
        n_em_iters=cfg["sinkhorn"]["n_em_iters"],
        init=cfg["sinkhorn"]["init"],
        seed=cfg["seed"], device=device, verbose=True,
    )

    output = args.output or str(out_dir / f"assignment_capped_L{L}_U{U}.pt")
    torch.save({
        "hard_assignment": result.hard_assignment,
        "centroids": result.centroids,
        "K": result.K, "M": result.M,
        "bucket_stats": result.bucket_stats,
        "image_ids": qa.image_ids,
        "alpha": qa.alpha,
        "L": L, "U": U,
    }, output)
    logger.info(f"stats: {json.dumps(result.bucket_stats, indent=2)}")
    print(f"\n✓ saved: {output}")
    print(f"桶统计: {json.dumps(result.bucket_stats, indent=2)}")


if __name__ == "__main__":
    main()
