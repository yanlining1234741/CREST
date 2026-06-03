"""数据加载:读取预计算的 SigLIP2 embedding,构建 image ↔ caption 映射。

数据格式约定:
    image_emb_path: 一个 torch.save() 的 dict
        {
            'features': Tensor [N, D],
            'image_ids': List[int] (长度 N)
        }
    text_emb_path: 同样的 dict
        {
            'features': Tensor [M, D],     M ≈ 5N (COCO 每图 5 caption)
            'image_ids': List[int]         每条 caption 对应的 image_id
        }

如果你的格式不同,只改这里。
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import torch
from .utils import l2_normalize


@dataclass
class Embeddings:
    """统一的 embedding 容器。"""
    image_features: torch.Tensor      # [N, D]
    image_ids: torch.Tensor           # [N], int64
    text_features: torch.Tensor       # [M, D]
    text_image_ids: torch.Tensor      # [M], 每条 caption 对应的 image_id
    embed_dim: int

    @property
    def n_images(self) -> int:
        return self.image_features.shape[0]

    @property
    def n_captions(self) -> int:
        return self.text_features.shape[0]

    def get_captions_for_image(self, image_id: int) -> torch.Tensor:
        """返回某 image_id 对应的所有 caption features。"""
        mask = self.text_image_ids == image_id
        return self.text_features[mask]


def load_embeddings(
    image_emb_path: str,
    text_emb_path: str,
    expected_dim: int = 1152,
    normalize: bool = True,
) -> Embeddings:
    """加载预计算 embedding。"""
    img_blob = torch.load(image_emb_path, map_location="cpu", weights_only=False)
    txt_blob = torch.load(text_emb_path, map_location="cpu", weights_only=False)

    img_feat = _coerce_tensor(img_blob, ["features", "image_features", "embeddings"])
    img_ids = _coerce_long_tensor(img_blob, ["image_ids", "ids"])

    txt_feat = _coerce_tensor(txt_blob, ["features", "text_features", "embeddings"])
    txt_img_ids = _coerce_long_tensor(txt_blob, ["image_ids", "img_ids"])

    # 校验
    assert img_feat.shape[1] == expected_dim, (
        f"image embedding dim {img_feat.shape[1]} != expected {expected_dim}"
    )
    assert txt_feat.shape[1] == expected_dim, (
        f"text embedding dim {txt_feat.shape[1]} != expected {expected_dim}"
    )
    assert img_feat.shape[0] == img_ids.shape[0]
    assert txt_feat.shape[0] == txt_img_ids.shape[0]

    if normalize:
        img_feat = l2_normalize(img_feat)
        txt_feat = l2_normalize(txt_feat)

    return Embeddings(
        image_features=img_feat,
        image_ids=img_ids,
        text_features=txt_feat,
        text_image_ids=txt_img_ids,
        embed_dim=expected_dim,
    )


def _coerce_tensor(blob: Dict, keys: List[str]) -> torch.Tensor:
    for k in keys:
        if k in blob:
            t = blob[k]
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            return t.float()
    raise KeyError(f"None of {keys} found in blob with keys {list(blob.keys())}")


def _coerce_long_tensor(blob: Dict, keys: List[str]) -> torch.Tensor:
    for k in keys:
        if k in blob:
            t = blob[k]
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            return t.long()
    raise KeyError(f"None of {keys} found in blob with keys {list(blob.keys())}")


def build_image_to_captions_map(emb: Embeddings) -> Dict[int, torch.Tensor]:
    """构建 image_id → [num_captions, D] 的映射,加速 Stage 1。"""
    mapping: Dict[int, List[int]] = {}
    for idx, img_id in enumerate(emb.text_image_ids.tolist()):
        mapping.setdefault(img_id, []).append(idx)

    return {
        img_id: emb.text_features[indices]
        for img_id, indices in mapping.items()
    }


def diagnostic_stats(emb: Embeddings) -> Dict[str, float]:
    """打印诊断统计,确认数据加载无误。"""
    img_norms = emb.image_features.norm(dim=-1)
    txt_norms = emb.text_features.norm(dim=-1)

    # 抽样配对的 text-image cosine sim (应该明显高于随机对)
    n_sample = min(1000, emb.n_images)
    rng = torch.Generator().manual_seed(0)
    sample_idx = torch.randperm(emb.n_images, generator=rng)[:n_sample]
    paired_sims, random_sims = [], []
    for i in sample_idx.tolist():
        img_id = emb.image_ids[i].item()
        img_feat = emb.image_features[i]
        caps = emb.get_captions_for_image(img_id)
        if len(caps) == 0:
            continue
        paired_sims.append((caps @ img_feat).mean().item())
        # 随机对照
        rand_caps = emb.text_features[torch.randint(emb.n_captions, (5,))]
        random_sims.append((rand_caps @ img_feat).mean().item())

    return {
        "n_images": emb.n_images,
        "n_captions": emb.n_captions,
        "captions_per_image_avg": emb.n_captions / max(1, emb.n_images),
        "img_norm_mean": img_norms.mean().item(),
        "img_norm_std": img_norms.std().item(),
        "txt_norm_mean": txt_norms.mean().item(),
        "txt_norm_std": txt_norms.std().item(),
        "paired_text_image_sim": sum(paired_sims) / max(1, len(paired_sims)),
        "random_text_image_sim": sum(random_sims) / max(1, len(random_sims)),
    }
