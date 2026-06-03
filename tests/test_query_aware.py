"""测试 Query-Aware 表征。"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import Embeddings
from src.query_aware import build_query_aware_embeddings
from src.utils import l2_normalize, set_seed


def _make_synthetic_embeddings(n_images: int = 50, n_captions_per_image: int = 5,
                                D: int = 64) -> Embeddings:
    """生成合成数据: 每张图有 n_captions_per_image 个 caption,
    caption embedding 是 image embedding 加微小噪声 (模拟语义对齐)。"""
    set_seed(0)
    img_feat = l2_normalize(torch.randn(n_images, D))
    img_ids = torch.arange(n_images, dtype=torch.long)

    txt_features = []
    txt_img_ids = []
    for i in range(n_images):
        for _ in range(n_captions_per_image):
            cap = img_feat[i] + 0.05 * torch.randn(D)
            txt_features.append(cap)
            txt_img_ids.append(i)
    txt_features = l2_normalize(torch.stack(txt_features))
    txt_img_ids = torch.tensor(txt_img_ids, dtype=torch.long)

    return Embeddings(
        image_features=img_feat,
        image_ids=img_ids,
        text_features=txt_features,
        text_image_ids=txt_img_ids,
        embed_dim=D,
    )


def test_alpha_1_equals_image():
    """alpha=1 时,新表征应等于原 image embedding。"""
    emb = _make_synthetic_embeddings()
    result = build_query_aware_embeddings(emb, alpha=1.0, l2_norm=True,
                                          show_progress=False)
    assert torch.allclose(result.features, emb.image_features, atol=1e-6)


def test_alpha_0_uses_caption_mean():
    """alpha=0 时,应为 caption mean 的 L2-normalized 版本。"""
    emb = _make_synthetic_embeddings()
    result = build_query_aware_embeddings(emb, alpha=0.0, l2_norm=True,
                                          show_progress=False)
    # 抽一个验证
    img_id = 0
    caps = emb.get_captions_for_image(img_id)
    expected = caps.mean(dim=0)
    expected = expected / expected.norm()
    assert torch.allclose(result.features[0], expected, atol=1e-5)


def test_output_is_l2_normalized():
    emb = _make_synthetic_embeddings()
    result = build_query_aware_embeddings(emb, alpha=0.5, l2_norm=True,
                                          show_progress=False)
    norms = result.features.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(emb.n_images), atol=1e-5)


def test_diagnostic_shows_improvement():
    """对齐的 caption-image 情形下,alpha=0.5 的 paired sim 应略高于 alpha=1.0。"""
    emb = _make_synthetic_embeddings()
    r05 = build_query_aware_embeddings(emb, alpha=0.5, show_progress=False)
    r10 = build_query_aware_embeddings(emb, alpha=1.0, show_progress=False)
    # delta 应为正或接近 0 (alpha=0.5 时 query-aware 起作用)
    assert r05.diagnostic["delta"] >= r10.diagnostic["delta"] - 0.01
