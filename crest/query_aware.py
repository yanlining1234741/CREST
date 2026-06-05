"""Stage 1: Query-Aware Item Representation.

对每个 item v_i,设它对应的 caption 集合 {q_j^{(i)}},则:

    z_tilde_i = alpha * z_v_i + (1 - alpha) * mean_j(z_q_j)

然后 L2 归一化。alpha 控制"image 视角"和"query 视角"的混合比例:
    alpha = 1.0 → 纯 image (baseline,等价于不做 query-aware)
    alpha = 0.5 → 平衡
    alpha = 0.0 → 纯 caption mean

输出: query_aware_emb.pt
    {
        'features': Tensor [N, D],     新的 item 表征
        'image_ids': Tensor [N],       对齐 input
        'alpha': float,                 用了哪个 alpha
        'n_captions_per_image': List[int],  每张图有几个 caption (诊断用)
    }
"""
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from tqdm import tqdm

from .data import Embeddings, build_image_to_captions_map
from .utils import l2_normalize


@dataclass
class QueryAwareResult:
    features: torch.Tensor              # [N, D]
    image_ids: torch.Tensor             # [N]
    alpha: float
    n_captions_per_image: list          # [N]
    diagnostic: Dict[str, float]


def build_query_aware_embeddings(
    emb: Embeddings,
    alpha: float = 0.5,
    l2_norm: bool = True,
    show_progress: bool = True,
) -> QueryAwareResult:
    """构造 query-aware item 表征。

    Args:
        emb: 加载好的 Embeddings 对象
        alpha: item embedding 的权重, 1-alpha 是 caption mean 的权重
        l2_norm: 混合后是否重新 L2 归一化
        show_progress: 是否显示进度条
    """
    assert 0.0 <= alpha <= 1.0, f"alpha must be in [0, 1], got {alpha}"

    img2caps = build_image_to_captions_map(emb)

    N, D = emb.image_features.shape
    new_features = torch.empty(N, D)
    n_caps_list = []

    it = range(N)
    if show_progress:
        it = tqdm(it, desc=f"Stage1 alpha={alpha}")

    for i in it:
        img_id = emb.image_ids[i].item()
        z_v = emb.image_features[i]
        caps = img2caps.get(img_id, None)

        if caps is None or len(caps) == 0:
            # 没 caption 的 item 退化为纯 image
            new_features[i] = z_v
            n_caps_list.append(0)
        else:
            caption_mean = caps.mean(dim=0)
            # 注意: caption mean 之后通常长度不再为 1
            mixed = alpha * z_v + (1.0 - alpha) * caption_mean
            new_features[i] = mixed
            n_caps_list.append(len(caps))

    if l2_norm:
        new_features = l2_normalize(new_features)

    # 诊断: 新表征下,正确 caption 和 item 的 cosine sim 应该比原 image 更高
    diag = _compute_diagnostic(emb, new_features, alpha)

    return QueryAwareResult(
        features=new_features,
        image_ids=emb.image_ids.clone(),
        alpha=alpha,
        n_captions_per_image=n_caps_list,
        diagnostic=diag,
    )


def _compute_diagnostic(
    emb: Embeddings,
    new_features: torch.Tensor,
    alpha: float,
    n_sample: int = 1000,
) -> Dict[str, float]:
    """诊断:
        1. 配对 (caption, new_item) 的平均 cos sim
        2. 配对 (caption, raw_image) 的平均 cos sim
        差值 > 0 说明 query-aware 起作用了
    """
    n_sample = min(n_sample, emb.n_images)
    rng = torch.Generator().manual_seed(42)
    sample_idx = torch.randperm(emb.n_images, generator=rng)[:n_sample]

    new_sims, raw_sims = [], []
    img2caps = build_image_to_captions_map(emb)

    for i in sample_idx.tolist():
        img_id = emb.image_ids[i].item()
        caps = img2caps.get(img_id, None)
        if caps is None or len(caps) == 0:
            continue
        new_sim = (caps @ new_features[i]).mean().item()
        raw_sim = (caps @ emb.image_features[i]).mean().item()
        new_sims.append(new_sim)
        raw_sims.append(raw_sim)

    return {
        "alpha": alpha,
        "n_sampled": len(new_sims),
        "paired_sim_new": sum(new_sims) / max(1, len(new_sims)),
        "paired_sim_raw_image": sum(raw_sims) / max(1, len(raw_sims)),
        "delta": (sum(new_sims) - sum(raw_sims)) / max(1, len(new_sims)),
    }


def save_query_aware(result: QueryAwareResult, path: str) -> None:
    """保存 Stage 1 结果。"""
    torch.save(
        {
            "features": result.features,
            "image_ids": result.image_ids,
            "alpha": result.alpha,
            "n_captions_per_image": result.n_captions_per_image,
            "diagnostic": result.diagnostic,
        },
        path,
    )


def load_query_aware(path: str) -> QueryAwareResult:
    """加载 Stage 1 结果。"""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return QueryAwareResult(
        features=blob["features"],
        image_ids=blob["image_ids"],
        alpha=blob["alpha"],
        n_captions_per_image=blob["n_captions_per_image"],
        diagnostic=blob["diagnostic"],
    )
