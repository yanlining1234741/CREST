"""Stage 3: Generative Router (query → bucket).

简单可靠的设计:在 text embedding 上接一个 MLP 分类器,输出 K 路 logits。

为什么这样? 一开始不需要 autoregressive 生成,因为:
    1. 单层 bucket K ≤ 1024 路 softmax 工程上 trivial
    2. 后续要扩到 hierarchical (K1 × K2) 再上 autoregressive

输出: router.pt
    {
        'state_dict': nn.Module 参数
        'config': {hidden_dim, n_layers, K, embed_dim, ...}
        'train_log': List[Dict]  每个 epoch 的训练/验证指标
    }
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class RouterMLP(nn.Module):
    """简单 MLP router:text emb → K-way logits。"""

    def __init__(self, embed_dim: int, K: int, hidden_dim: int = 512,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = embed_dim
        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, K))
        self.net = nn.Sequential(*layers)
        self.K = K
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

class RouterDataset(Dataset):
    """(query_embedding, target_bucket_ids) 对。

    如果 M > 1 (multi-view),一个 query 对应多个 ground-truth 桶,
    用 multi-label cross-entropy。
    """

    def __init__(
        self,
        text_features: torch.Tensor,        # [n_captions, D]
        text_image_ids: torch.Tensor,       # [n_captions]
        image_ids: torch.Tensor,            # [N], 对齐 hard_assignment
        hard_assignment: torch.Tensor,      # [N, M]
    ):
        # image_id → 行索引
        img_id_to_row = {iid.item(): i for i, iid in enumerate(image_ids)}
        valid_mask = torch.tensor(
            [iid.item() in img_id_to_row for iid in text_image_ids]
        )
        self.text_features = text_features[valid_mask]
        ids = text_image_ids[valid_mask]
        row_idx = torch.tensor(
            [img_id_to_row[iid.item()] for iid in ids], dtype=torch.long,
        )
        self.target_buckets = hard_assignment[row_idx]  # [n_captions, M]
        self.K = int(hard_assignment.max().item()) + 1

    def __len__(self) -> int:
        return self.text_features.shape[0]

    def __getitem__(self, idx: int):
        return self.text_features[idx], self.target_buckets[idx]


def multi_label_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """对 M-view 用 multi-label CE:any one of M is correct。

    logits: [B, K]
    targets: [B, M]   M 个 ground-truth 桶
    损失: -log(sum_m softmax(logits)[targets[m]])
    """
    log_probs = F.log_softmax(logits, dim=-1)              # [B, K]
    # gather: [B, M]
    selected = log_probs.gather(1, targets)
    # logsumexp 表示 "any-of-M 命中"
    loss = -torch.logsumexp(selected, dim=-1).mean()
    return loss


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_router(
    model: RouterMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    top_B_list: Tuple[int, ...] = (1, 3, 5, 10),
    verbose: bool = True,
) -> List[Dict]:
    """训练 router,返回每个 epoch 的日志。"""
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    log = []

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_losses = []
        it = train_loader
        if verbose:
            it = tqdm(it, desc=f"Epoch {epoch}/{epochs} train", leave=False)
        for x, y in it:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = multi_label_ce(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        scheduler.step()

        # ---- val ----
        val_metrics = evaluate_router(model, val_loader, top_B_list, device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = sum(train_losses) / max(1, len(train_losses))
        val_metrics["lr"] = scheduler.get_last_lr()[0]
        log.append(val_metrics)

        if verbose:
            recall_str = " ".join(
                f"R@{b}={val_metrics[f'recall@{b}']:.4f}" for b in top_B_list
            )
            print(f"[Epoch {epoch}] loss={val_metrics['train_loss']:.4f} "
                  f"val {recall_str}")

    return log


@torch.no_grad()
def evaluate_router(
    model: RouterMLP,
    loader: DataLoader,
    top_B_list: Tuple[int, ...],
    device: str = "cuda",
) -> Dict[str, float]:
    """计算 router top-B recall。"""
    model.eval()
    max_B = max(top_B_list)
    counts = {b: 0 for b in top_B_list}
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)                              # [B, M]
        logits = model(x)
        topB = logits.topk(max_B, dim=-1).indices     # [B, max_B]
        # any-of-M hits any-of-B
        for b in top_B_list:
            topb = topB[:, :b].unsqueeze(2)            # [B, b, 1]
            y_exp = y.unsqueeze(1)                     # [B, 1, M]
            hit = (topb == y_exp).any(dim=2).any(dim=1)
            counts[b] += hit.sum().item()
        total += x.shape[0]
    return {f"recall@{b}": counts[b] / max(1, total) for b in top_B_list}


def save_router(model: RouterMLP, log: List[Dict], path: str) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "embed_dim": model.embed_dim,
            "K": model.K,
        },
        "train_log": log,
    }, path)


def load_router(path: str, hidden_dim: int = 512, n_layers: int = 2,
                dropout: float = 0.1) -> RouterMLP:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = RouterMLP(
        embed_dim=cfg["embed_dim"], K=cfg["K"],
        hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
    )
    model.load_state_dict(blob["state_dict"])
    return model


# ============ Disjoint pool 支持 (Flickr/MSCOCO) ============
class RouterDatasetDisjoint(Dataset):
    """直接吃 (query_feature, bucket_label) 对, 不走 image_id 查找。
    用于 disjoint pool: train query 的 target image 不在 eval pool 里,
    桶 label 已由外部预先算好 (train 图映射到 eval-pool centroid)。
    """
    def __init__(self, text_features, target_buckets, K):
        self.text_features = text_features      # [Q, D]
        self.target_buckets = target_buckets    # [Q, M]
        self.K = K

    def __len__(self):
        return self.text_features.shape[0]

    def __getitem__(self, idx):
        return self.text_features[idx], self.target_buckets[idx]
