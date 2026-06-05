"""Stage 2: Sinkhorn-Balanced K-means Bucket Assignment.

数学:
    给定 N 个 item 的 query-aware 表征 Z [N, D],求分配矩阵 P [N, K]
    使得:
        行约束: sum_k P[i,k] = M             (每 item 分到 M 个桶,M=1 是单视角)
        列约束: sum_i P[i,k] = N*M/K          (每桶大小严格均衡)
        目标: minimize <P, C> + epsilon * H(P)  其中 C[i,k] = -<z_i, c_k>

    用 Sinkhorn iterations 求解 (entropy-regularized OT)。

EM 外循环:
    E-step: 固定 centroids,Sinkhorn 求 P
    M-step: 固定 P,更新 centroids = weighted mean

输出: assignment.pt
    {
        'soft_assignment': Tensor [N, K]   软分配
        'hard_assignment': Tensor [N, M],  硬分配 (M 个桶 ID)
        'centroids': Tensor [K, D],         桶中心
        'K': int, 'M': int,
        'bucket_stats': Dict,
    }
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math

import torch
from tqdm import tqdm

from .utils import l2_normalize


@dataclass
class AssignmentResult:
    soft_assignment: torch.Tensor       # [N, K]
    hard_assignment: torch.Tensor       # [N, M]
    centroids: torch.Tensor             # [K, D]
    K: int
    M: int
    bucket_stats: Dict


# ---------------------------------------------------------------------------
# Sinkhorn 核心
# ---------------------------------------------------------------------------

def sinkhorn_normalize(
    log_alpha: torch.Tensor,
    n_iters: int = 5,
    col_target: float = 1.0,
) -> torch.Tensor:
    """Sinkhorn iterations in log space.

    log_alpha: [N, K] 初始 log-similarity (== -cost / eps)
    n_iters: 迭代次数, 3-5 通常足够
    col_target: 列和的目标值 (= N*M/K)

    返回归一化后的 log_P, [N, K]
    满足: exp(log_P).sum(dim=1) ≈ 1
         exp(log_P).sum(dim=0) ≈ col_target
    """
    log_col_target = math.log(col_target)
    log_P = log_alpha.clone()
    for _ in range(n_iters):
        # 行归一化: 每行和 = 1
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
        # 列归一化: 每列和 = col_target
        log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True) + log_col_target
    return log_P


# ---------------------------------------------------------------------------
# K-means++ 初始化
# ---------------------------------------------------------------------------

def kmeanspp_init(
    Z: torch.Tensor, K: int, seed: int = 42,
) -> torch.Tensor:
    """K-means++ 初始化 K 个 centroid。"""
    device = Z.device
    N, D = Z.shape
    rng = torch.Generator(device=device).manual_seed(seed)
    centroids = torch.empty(K, D, device=device)

    # 第一个随机选
    first = torch.randint(N, (1,), generator=rng, device=device).item()
    centroids[0] = Z[first]

    # 后续按距离加权采样
    closest_dist_sq = torch.full((N,), float("inf"), device=device)
    for k in range(1, K):
        diff = Z - centroids[k - 1]
        new_dist_sq = (diff * diff).sum(dim=1)
        closest_dist_sq = torch.minimum(closest_dist_sq, new_dist_sq)
        probs = closest_dist_sq / (closest_dist_sq.sum() + 1e-10)
        next_idx = torch.multinomial(probs, 1, generator=rng).item()
        centroids[k] = Z[next_idx]

    return l2_normalize(centroids)


# ---------------------------------------------------------------------------
# 主 EM 循环
# ---------------------------------------------------------------------------

def run_sinkhorn_kmeans(
    Z: torch.Tensor,
    K: int,
    M: int = 1,
    epsilon: float = 0.05,
    n_sinkhorn_iters: int = 5,
    n_em_iters: int = 20,
    init: str = "kmeans++",
    seed: int = 42,
    device: str = "cuda",
    verbose: bool = True,
) -> AssignmentResult:
    """主入口:Sinkhorn-balanced k-means。

    Args:
        Z: [N, D] L2-normalized item 表征
        K: 桶数量
        M: 每 item 分到几个桶 (multi-view)
        epsilon: entropy regularization
        n_sinkhorn_iters: 内层 Sinkhorn 迭代数
        n_em_iters: 外层 EM 迭代数
    """
    Z = Z.to(device)
    N, D = Z.shape
    assert N >= K, f"need N >= K, got N={N}, K={K}"

    if verbose:
        print(f"[Sinkhorn] N={N}, K={K}, M={M}, eps={epsilon}, "
              f"target_bucket_size={N*M/K:.1f}")

    # 初始化
    if init == "kmeans++":
        centroids = kmeanspp_init(Z, K, seed=seed)
    elif init == "random":
        rng = torch.Generator(device=device).manual_seed(seed)
        idx = torch.randperm(N, generator=rng, device=device)[:K]
        centroids = l2_normalize(Z[idx].clone())
    else:
        raise ValueError(f"Unknown init: {init}")

    col_target = N * M / K
    prev_obj = float("inf")

    it = range(n_em_iters)
    if verbose:
        it = tqdm(it, desc="Sinkhorn EM")

    for em_iter in it:
        # ---- E-step: 计算分配 ----
        sim = Z @ centroids.t()                   # [N, K] cosine sim (already normalized)
        log_alpha = sim / epsilon
        log_P = sinkhorn_normalize(
            log_alpha, n_iters=n_sinkhorn_iters, col_target=col_target,
        )
        P = log_P.exp()

        # ---- M-step: 更新 centroid ----
        # c_k = sum_i P[i,k] * z_i / sum_i P[i,k]
        weight = P.sum(dim=0, keepdim=True)       # [1, K]
        new_centroids = (P.t() @ Z) / (weight.t() + 1e-10)
        new_centroids = l2_normalize(new_centroids)

        # 目标函数 (诊断)
        obj = -(P * sim).sum().item() + epsilon * (P * (log_P)).sum().item()

        # 早停
        if abs(prev_obj - obj) < 1e-6 * abs(prev_obj):
            if verbose:
                print(f"[Sinkhorn] converged at iter {em_iter}, obj={obj:.4f}")
            break
        prev_obj = obj

        centroids = new_centroids

    # 最终硬分配: 取 top-M 个桶
    sim_final = Z @ centroids.t()
    if M == 1:
        hard = sim_final.argmax(dim=1, keepdim=True)  # [N, 1]
    else:
        hard = sim_final.topk(M, dim=1).indices       # [N, M]

    # 统计
    stats = compute_bucket_stats(hard, K)
    if verbose:
        print(f"[Sinkhorn] bucket size: "
              f"mean={stats['mean']:.1f}, std={stats['std']:.1f}, "
              f"min={stats['min']}, max={stats['max']}")

    return AssignmentResult(
        soft_assignment=P.cpu(),
        hard_assignment=hard.cpu(),
        centroids=centroids.cpu(),
        K=K,
        M=M,
        bucket_stats=stats,
    )


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------

def compute_bucket_stats(hard_assignment: torch.Tensor, K: int) -> Dict:
    """统计桶大小分布。

    hard_assignment: [N, M],每行是该 item 所属的 M 个桶 ID
    """
    flat = hard_assignment.flatten()
    counts = torch.bincount(flat, minlength=K).float()
    return {
        "mean": counts.mean().item(),
        "std": counts.std().item(),
        "min": int(counts.min().item()),
        "max": int(counts.max().item()),
        "n_empty": int((counts == 0).sum().item()),
        "std_over_mean": (counts.std() / (counts.mean() + 1e-10)).item(),
    }


def save_assignment(result: AssignmentResult, path: str,
                    save_soft: bool = False) -> None:
    """保存 Stage 2 结果。soft assignment 比较大,默认不存。"""
    blob = {
        "hard_assignment": result.hard_assignment,
        "centroids": result.centroids,
        "K": result.K,
        "M": result.M,
        "bucket_stats": result.bucket_stats,
    }
    if save_soft:
        blob["soft_assignment"] = result.soft_assignment
    torch.save(blob, path)


def load_assignment(path: str) -> AssignmentResult:
    """加载 Stage 2 结果。"""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return AssignmentResult(
        soft_assignment=blob.get("soft_assignment", torch.empty(0)),
        hard_assignment=blob["hard_assignment"],
        centroids=blob["centroids"],
        K=blob["K"],
        M=blob["M"],
        bucket_stats=blob["bucket_stats"],
    )


# ===========================================================================
# 容量约束版 (capacity-constrained): 每桶候选数硬卡 [L, U]
# 新增函数, 不影响上面原有的 run_sinkhorn_kmeans
# ===========================================================================

def capacity_constrained_assign(
    sim: torch.Tensor,   # [N, K] 候选到各桶中心的相似度 (越大越近)
    M: int,              # 每候选分到 M 个桶
    L: int,              # 每桶下界
    U: int,              # 每桶上界
    verbose: bool = True,
):
    """带容量上下界的硬分配。

    思路:
      1. 上界: 贪心分配, 每候选想去最近的 M 个桶, 但桶满 U 就拒绝,
         候选改去次近的未满桶。保证每桶 <= U。
      2. 下界: 分配完后, 不足 L 的桶, 从"离它最近且能匀出"的候选里拉,
         给候选追加一个桶 (该候选的桶数 > M, 容忍少量超配以满足下界)。

    返回:
      hard: [N, M_eff] long, 每候选所属桶 (下界修正可能让个别候选 > M 个桶,
            用 -1 padding 对齐到 max 桶数)
      counts: [K] 每桶最终大小
    """
    N, K = sim.shape
    device = sim.device

    # 候选对每个桶的偏好排序 (从最近到最远)
    # 为省内存, 只取前 max(M*4, 一定数量) 个候选桶备选
    n_pref = min(K, max(M * 8, 64))
    pref = sim.topk(n_pref, dim=1).indices  # [N, n_pref] 每候选的候选桶 (按近到远)

    bucket_members = [[] for _ in range(K)]  # 每桶的候选 list
    cand_buckets = [[] for _ in range(N)]    # 每候选的桶 list
    counts = torch.zeros(K, dtype=torch.long, device=device)

    # ---- 上界贪心: 每候选填到 M 个桶, 跳过已满 (U) 的桶 ----
    pref_cpu = pref.cpu().numpy()
    for i in range(N):
        assigned = 0
        for k in pref_cpu[i]:
            k = int(k)
            if counts[k] < U:
                bucket_members[k].append(i)
                cand_buckets[i].append(k)
                counts[k] += 1
                assigned += 1
                if assigned >= M:
                    break
        # 若所有备选桶都满 (极端), 强行放进最空的桶 (保证候选至少 1 个桶)
        if assigned == 0:
            k = int(counts.argmin())
            bucket_members[k].append(i)
            cand_buckets[i].append(k)
            counts[k] += 1

    if verbose:
        print(f"[capped] 上界后: min={int(counts.min())} max={int(counts.max())} "
              f"(U={U}), 空桶={int((counts==0).sum())}")

    # ---- 下界修正: 不足 L 的桶, 拉最近的候选补足 ----
    sim_cpu = sim.cpu()
    for k in range(K):
        if counts[k] >= L:
            continue
        # 该桶最想要的候选 (按相似度排序), 还没在这桶里的
        order = sim_cpu[:, k].argsort(descending=True)
        ptr = 0
        while counts[k] < L and ptr < N:
            i = int(order[ptr]); ptr += 1
            if k in cand_buckets[i]:
                continue
            # 给候选 i 追加桶 k (容忍 i 的桶数 > M)
            bucket_members[k].append(i)
            cand_buckets[i].append(k)
            counts[k] += 1

    if verbose:
        print(f"[capped] 下界后: min={int(counts.min())} max={int(counts.max())} "
              f"(L={L},U={U}), 空桶={int((counts==0).sum())}")
        n_over = sum(1 for b in cand_buckets if len(b) > M)
        print(f"[capped] 因下界超配(桶数>M)的候选: {n_over}/{N} "
              f"({100*n_over/N:.1f}%)")

    # ---- 对齐成 [N, max_m] 张量, -1 padding ----
    max_m = max(len(b) for b in cand_buckets)
    hard = torch.full((N, max_m), -1, dtype=torch.long)
    for i in range(N):
        hard[i, :len(cand_buckets[i])] = torch.tensor(cand_buckets[i], dtype=torch.long)

    return hard, counts.cpu()


def run_sinkhorn_kmeans_capped(
    Z: torch.Tensor,
    K: int,
    M: int,
    L: int,
    U: int,
    epsilon: float = 0.01,
    n_sinkhorn_iters: int = 5,
    n_em_iters: int = 30,
    init: str = "kmeans++",
    seed: int = 42,
    device: str = "cuda",
    verbose: bool = True,
) -> AssignmentResult:
    """容量约束版: 先跑标准 Sinkhorn EM 得 centroids, 再做容量约束硬分配。

    每桶最终候选数保证 ∈ [L, U] (除非 K 设置本身不可行)。
    """
    Z = Z.to(device)
    N, D = Z.shape
    assert N >= K, f"need N >= K, got N={N}, K={K}"

    # 可行性检查: K 必须满足 N*M/U <= K <= N*M/L
    k_min = math.ceil(N * M / U)
    k_max = math.floor(N * M / L)
    if verbose:
        print(f"[capped] N={N}, K={K}, M={M}, [L,U]=[{L},{U}]")
        print(f"[capped] 可行 K 范围: [{k_min}, {k_max}] (要每桶∈[{L},{U}])")
        if K < k_min:
            print(f"[capped] ⚠ K={K} < {k_min}: 桶不够, 无法保证每桶<=U "
                  f"(总量{N*M}需要至少{k_min}桶)")
        if K > k_max:
            print(f"[capped] ⚠ K={K} > {k_max}: 桶太多, 无法保证每桶>=L "
                  f"(总量{N*M}最多撑{k_max}桶)")

    # ---- 标准 Sinkhorn EM 得 centroids (复用原逻辑) ----
    if init == "kmeans++":
        centroids = kmeanspp_init(Z, K, seed=seed)
    else:
        rng = torch.Generator(device=device).manual_seed(seed)
        idx = torch.randperm(N, generator=rng, device=device)[:K]
        centroids = l2_normalize(Z[idx].clone())

    col_target = N * M / K
    prev_obj = float("inf")
    it = tqdm(range(n_em_iters), desc="Sinkhorn EM (capped)") if verbose else range(n_em_iters)
    for em_iter in it:
        sim = Z @ centroids.t()
        log_alpha = sim / epsilon
        log_P = sinkhorn_normalize(log_alpha, n_iters=n_sinkhorn_iters, col_target=col_target)
        P = log_P.exp()
        weight = P.sum(dim=0, keepdim=True)
        new_centroids = l2_normalize((P.t() @ Z) / (weight.t() + 1e-10))
        obj = -(P * sim).sum().item() + epsilon * (P * log_P).sum().item()
        if abs(prev_obj - obj) < 1e-6 * abs(prev_obj):
            if verbose:
                print(f"[capped] EM converged at iter {em_iter}")
            break
        prev_obj = obj
        centroids = new_centroids

    # ---- 容量约束硬分配 (替换原 topk) ----
    sim_final = Z @ centroids.t()
    hard, counts = capacity_constrained_assign(sim_final, M, L, U, verbose=verbose)

    # 统计 (注意 hard 有 -1 padding, 要过滤)
    flat = hard.flatten()
    flat = flat[flat >= 0]
    counts_t = torch.bincount(flat, minlength=K).float()
    stats = {
        "mean": counts_t.mean().item(),
        "std": counts_t.std().item(),
        "min": int(counts_t.min().item()),
        "max": int(counts_t.max().item()),
        "n_empty": int((counts_t == 0).sum().item()),
        "std_over_mean": (counts_t.std() / (counts_t.mean() + 1e-10)).item(),
        "L": L, "U": U,
        "within_bounds": bool(((counts_t >= L) & (counts_t <= U)).all().item()),
    }
    if verbose:
        print(f"[capped] 最终桶大小: mean={stats['mean']:.1f} "
              f"min={stats['min']} max={stats['max']} "
              f"全部∈[{L},{U}]: {stats['within_bounds']}")

    return AssignmentResult(
        soft_assignment=torch.empty(0),
        hard_assignment=hard.cpu(),
        centroids=centroids.cpu(),
        K=K, M=M,
        bucket_stats=stats,
    )
