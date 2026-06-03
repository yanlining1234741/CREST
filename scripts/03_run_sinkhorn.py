"""03: Sinkhorn-Balanced K-Means 桶分配 (Stage 2)。

用法:
    python scripts/03_run_sinkhorn.py --config configs/coco_siglip2.yaml \\
        --input outputs/query_aware_emb_alpha0.5.pt --K 256 --M 1

输出:
    outputs/assignment_K<K>_M<M>.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.query_aware import load_query_aware
from src.sinkhorn import run_sinkhorn_kmeans, save_assignment
from src.utils import (ensure_dir, get_device, load_config,
                       set_seed, setup_logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True,
                        help="Stage 1 输出的 .pt 文件")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--n_em_iters", type=int, default=None)
    parser.add_argument("--init", type=str, default=None,
                        choices=["kmeans++", "random"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save_soft", action="store_true",
                        help="是否保存 soft assignment (体积大)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "03_run_sinkhorn.log"))

    K = args.K if args.K is not None else cfg["sinkhorn"]["K"]
    M = args.M if args.M is not None else cfg["sinkhorn"]["M"]
    eps = args.epsilon if args.epsilon is not None else cfg["sinkhorn"]["epsilon"]
    n_em = args.n_em_iters if args.n_em_iters is not None else cfg["sinkhorn"]["n_em_iters"]
    init = args.init or cfg["sinkhorn"]["init"]

    output = args.output or str(out_dir / f"assignment_K{K}_M{M}.pt")

    logger.info(f"Loading query-aware features: {args.input}")
    qa = load_query_aware(args.input)
    Z = qa.features
    logger.info(f"N={Z.shape[0]}, D={Z.shape[1]}, alpha={qa.alpha}")

    device = str(get_device(cfg["data"]["device"]))
    logger.info(f"Running Sinkhorn-kmeans on {device}: "
                f"K={K}, M={M}, eps={eps}, n_em={n_em}, init={init}")

    result = run_sinkhorn_kmeans(
        Z, K=K, M=M, epsilon=eps,
        n_sinkhorn_iters=cfg["sinkhorn"]["n_sinkhorn_iters"],
        n_em_iters=n_em,
        init=init,
        seed=cfg["seed"],
        device=device,
        verbose=True,
    )

    # 附加 image_ids 以便下游对齐
    blob = {
        "hard_assignment": result.hard_assignment,
        "centroids": result.centroids,
        "K": result.K,
        "M": result.M,
        "bucket_stats": result.bucket_stats,
        "image_ids": qa.image_ids,
        "alpha": qa.alpha,
    }
    if args.save_soft:
        blob["soft_assignment"] = result.soft_assignment
    torch.save(blob, output)

    logger.info(f"Bucket stats:\n{json.dumps(result.bucket_stats, indent=2)}")
    logger.info(f"✓ Saved to {output}")

    # 健康检查
    std_ratio = result.bucket_stats["std_over_mean"]
    if std_ratio > 0.2:
        logger.warning(f"⚠ Bucket size imbalance: std/mean = {std_ratio:.3f}, "
                       f"consider more EM iters or smaller epsilon")
    else:
        logger.info(f"✓ Buckets balanced: std/mean = {std_ratio:.3f}")


if __name__ == "__main__":
    main()
