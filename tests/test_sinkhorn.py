"""测试 Sinkhorn 实现。

跑法:
    pytest tests/test_sinkhorn.py -v
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sinkhorn import (compute_bucket_stats, kmeanspp_init,
                          run_sinkhorn_kmeans, sinkhorn_normalize)
from src.utils import l2_normalize, set_seed


def test_sinkhorn_normalize_row_sum():
    """Sinkhorn 后行和应近似为 1。"""
    set_seed(0)
    N, K = 100, 10
    log_alpha = torch.randn(N, K)
    log_P = sinkhorn_normalize(log_alpha, n_iters=5, col_target=N / K)
    P = log_P.exp()
    row_sums = P.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(N), atol=1e-3), \
        f"row sums not close to 1: range [{row_sums.min()}, {row_sums.max()}]"


def test_sinkhorn_normalize_col_sum():
    """Sinkhorn 后列和应近似为 N/K。"""
    set_seed(0)
    N, K = 100, 10
    log_alpha = torch.randn(N, K)
    log_P = sinkhorn_normalize(log_alpha, n_iters=5, col_target=N / K)
    P = log_P.exp()
    col_sums = P.sum(dim=0)
    target = N / K
    assert torch.allclose(col_sums, torch.full((K,), target), atol=0.5), \
        f"col sums not close to {target}: range [{col_sums.min()}, {col_sums.max()}]"


def test_kmeanspp_init_returns_K_unit_vectors():
    set_seed(0)
    N, D, K = 500, 64, 16
    Z = l2_normalize(torch.randn(N, D))
    C = kmeanspp_init(Z, K, seed=0)
    assert C.shape == (K, D)
    norms = C.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(K), atol=1e-5)


def test_sinkhorn_kmeans_balanced_buckets():
    """主测试: 在随机数据上,Sinkhorn-kmeans 应产生近似均衡的桶。"""
    set_seed(0)
    N, D, K = 1000, 128, 10
    # 用合成的"语义簇"数据
    centers = l2_normalize(torch.randn(K, D))
    Z = []
    for k in range(K):
        c = centers[k]
        cluster = c + 0.1 * torch.randn(N // K, D)
        Z.append(cluster)
    Z = l2_normalize(torch.cat(Z, dim=0))

    result = run_sinkhorn_kmeans(
        Z, K=K, M=1, epsilon=0.05, n_em_iters=15,
        device="cpu", verbose=False,
    )

    stats = result.bucket_stats
    # 均衡性: std/mean < 0.2
    assert stats["std_over_mean"] < 0.2, \
        f"buckets imbalanced: std/mean = {stats['std_over_mean']}"
    # 无空桶
    assert stats["n_empty"] == 0


def test_sinkhorn_kmeans_multiview():
    """multi-view: M=2 时,每个桶大小应约为 N*M/K。"""
    set_seed(0)
    N, D, K, M = 1000, 64, 10, 2
    Z = l2_normalize(torch.randn(N, D))

    result = run_sinkhorn_kmeans(
        Z, K=K, M=M, epsilon=0.05, n_em_iters=15,
        device="cpu", verbose=False,
    )

    # hard_assignment 形状应为 [N, M]
    assert result.hard_assignment.shape == (N, M)
    # 每行应有 M 个不同的桶 (大多数情况)
    expected_size = N * M / K
    assert abs(result.bucket_stats["mean"] - expected_size) < 5
