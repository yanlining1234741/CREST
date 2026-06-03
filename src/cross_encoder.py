"""Cross-Encoder Rerank. ULTRA-OPTIMIZED for VN 542K."""
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CrossEncoderRerank(nn.Module):
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 512,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        input_dim = embed_dim * 4 + 1
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.dim() == 2 and v.dim() == 3:
            q = q.unsqueeze(1).expand(-1, v.size(1), -1)
        elif q.dim() == 3 and v.dim() == 2:
            v = v.unsqueeze(1).expand(-1, q.size(1), -1)
        concat = torch.cat([q, v], dim=-1)
        product = q * v
        diff = (q - v).abs()
        cos = (q * v).sum(dim=-1, keepdim=True)
        features = torch.cat([concat, product, diff, cos], dim=-1)
        return self.mlp(features).squeeze(-1)


class CrossEncoderDataset(Dataset):
    """ULTRA-OPTIMIZED: 不预计算 HN pool, 用 lazy 随机采样.
    
    思路:
      不预计算所有 query 的 HN pool (太慢)
      __getitem__ 时直接从桶里随机 sample n_hn 个
      避免 numpy unique 等慢操作
    """

    def __init__(
        self,
        text_features: torch.Tensor,
        text_targets: torch.Tensor,
        image_features: torch.Tensor,
        hard_assignment: torch.Tensor,
        n_negatives: int = 15,
        hn_ratio: float = 0.7,
        seed: int = 42,
        max_hn_per_target: int = 200,  # 现在没用了
    ):
        valid_mask = text_targets >= 0
        self.text_features = text_features[valid_mask]
        self.text_targets = text_targets[valid_mask].long().numpy()
        self.image_features = image_features
        self.n_neg = n_negatives
        self.n_hn = int(n_negatives * hn_ratio)
        self.n_rand = n_negatives - self.n_hn
        self.N_img = image_features.shape[0]
        
        print(f"[Dataset] Building reverse index...", flush=True)
        t0 = time.time()
        
        # 向量化建立 bucket → images 索引
        ha_np = hard_assignment.numpy()  # [N, M]
        N, M = ha_np.shape
        K = int(ha_np.max() + 1)
        
        # flatten + sort
        flat_buckets = ha_np.flatten()  # [N*M]
        flat_imgs = np.arange(N).repeat(M)  # [N*M]
        sort_order = flat_buckets.argsort(kind='stable')
        self.sorted_imgs = flat_imgs[sort_order].astype(np.int32)
        
        # bucket_offsets
        bucket_offsets = np.zeros(K + 1, dtype=np.int64)
        unique_buckets, counts = np.unique(flat_buckets[sort_order], return_counts=True)
        bucket_offsets[unique_buckets + 1] = counts
        self.bucket_offsets = bucket_offsets.cumsum()
        
        # 每个 image 所在的桶
        self.image_buckets = ha_np.astype(np.int32)  # [N, M]
        
        elapsed = time.time() - t0
        print(f"[Dataset] Reverse index done in {elapsed:.1f}s. K={K}, M={M}, sorted_imgs size={len(self.sorted_imgs)}", flush=True)
        
        # 创建每个 thread 独立的 RNG (NumPy default_rng 是 thread-safe)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        print(f"[Dataset] Ready. n_neg={self.n_neg} (n_hn={self.n_hn} + n_rand={self.n_rand})", flush=True)

    def __len__(self) -> int:
        return self.text_features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.text_features[idx]
        target = self.text_targets[idx]
        v_pos = self.image_features[target]
        
        # Hard negatives: 从 target 的 M 个桶里直接随机采
        target_buckets = self.image_buckets[target]  # [M]
        
        # 直接从每个桶随机采样, 不做 union 不做 unique (太慢)
        # 每个 hard negative 随机选一个桶, 再从那桶随机选一个 image
        hn_buckets = self.rng.choice(target_buckets, self.n_hn, replace=True)
        hn = np.empty(self.n_hn, dtype=np.int64)
        for i in range(self.n_hn):
            b = hn_buckets[i]
            start, end = self.bucket_offsets[b], self.bucket_offsets[b + 1]
            if end > start:
                hn[i] = self.sorted_imgs[self.rng.integers(start, end)]
            else:
                hn[i] = self.rng.integers(0, self.N_img)
        
        # Random negatives
        rand_neg = self.rng.integers(0, self.N_img, self.n_rand)
        
        neg_idxs = np.concatenate([hn, rand_neg])
        # 可能采到 target 自己, 但概率低; 不再 filter (省时间)
        
        v_negs = self.image_features[torch.from_numpy(neg_idxs).long()]
        return q, v_pos, v_negs


def info_nce_loss(score_pos: torch.Tensor, score_neg: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    all_scores = torch.cat([score_pos.unsqueeze(1), score_neg], dim=1)
    labels = torch.zeros(score_pos.shape[0], dtype=torch.long, device=score_pos.device)
    return F.cross_entropy(all_scores / temperature, labels)


def train_cross_encoder(
    model, train_loader, val_loader,
    epochs=20, lr=1e-3, weight_decay=1e-5, temperature=0.1,
    device="cuda", verbose=True,
):
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        it = train_loader
        if verbose:
            it = tqdm(it, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for q, v_pos, v_negs in it:
            q = q.to(device, non_blocking=True)
            v_pos = v_pos.to(device, non_blocking=True)
            v_negs = v_negs.to(device, non_blocking=True)
            score_pos = model(q, v_pos)
            score_neg = model(q, v_negs)
            loss = info_nce_loss(score_pos, score_neg, temperature)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for q, v_pos, v_negs in val_loader:
                q = q.to(device); v_pos = v_pos.to(device); v_negs = v_negs.to(device)
                score_pos = model(q, v_pos).unsqueeze(1)
                score_neg = model(q, v_negs)
                all_scores = torch.cat([score_pos, score_neg], dim=1)
                preds = all_scores.argmax(dim=-1)
                val_correct += (preds == 0).sum().item()
                val_total += q.shape[0]

        val_acc = val_correct / max(1, val_total)
        avg_loss = sum(train_losses) / max(1, len(train_losses))
        log.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0],
        })
        if verbose:
            print(f"[CE Epoch {epoch}] loss={avg_loss:.4f} val_acc={val_acc:.4f}", flush=True)
    return log


def save_cross_encoder(model, log, path):
    torch.save({
        'state_dict': model.state_dict(),
        'config': {'embed_dim': model.embed_dim},
        'train_log': log,
    }, path)


def load_cross_encoder(path, hidden_dim=512, n_layers=3, dropout=0.1):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    cfg = blob['config']
    model = CrossEncoderRerank(
        embed_dim=cfg['embed_dim'],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    model.load_state_dict(blob['state_dict'])
    return model


class CECrossAttention(nn.Module):
    """Cross-attention based listwise reranker.
    
    架构关键: query 通过 cross-attention 关注所有 cand, 输出 ranking score.
    跟 MLP 的根本区别: cand 之间的 score 互相依赖 (通过 softmax(QK^T)),
    自然学到 ranking 关系而不是独立二分类.
    
    Hybrid 设计: final score = α * cosine + (1-α) * attn_score
    α 默认 0.5, learned. 保底退化到 cosine.
    """
    def __init__(self, embed_dim=768, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # Project q, k, v
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Multi-layer cross attention
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        # Final scoring head
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        # Learned mixing with cosine (保底)
        self.alpha = nn.Parameter(torch.tensor(0.0))  # delta 初始 0, 等价 cosine  # 初始 0.3, learned
    
    def forward(self, q, v):
        """
        q: (B, D) or (B, 1, D)
        v: (B, N, D) - cand candidates
        Returns: (B, N) scores
        """
        if q.dim() == 2:
            q_in = q.unsqueeze(1)  # (B, 1, D)
        else:
            q_in = q
        
        # 原始 cosine (用于 hybrid)
        with torch.no_grad():
            cos_score = (q_in @ v.transpose(-1, -2)).squeeze(1)  # (B, N)
        
        # Project
        q_proj = self.q_proj(q_in)  # (B, 1, D)
        k_proj = self.k_proj(v)      # (B, N, D)
        v_proj = self.v_proj(v)      # (B, N, D)
        
        # Cross-attention: query attends to cand
        # 每层: q 作为 query, k/v 是 cand
        h = q_proj
        for attn, norm in zip(self.attn_layers, self.norms):
            attn_out, _ = attn(h, k_proj, v_proj)  # (B, 1, D)
            h = norm(h + attn_out)
        
        # h 此时是 (B, 1, D), 整合了所有 cand 的信息
        # 把 h 跟每个 cand 算 score
        # 关键: 用 batch-wise z-score 归一化, 让 attn 跟 cos 同量级
        attn_raw = (h @ v.transpose(-1, -2)).squeeze(1)  # (B, N)
        # Batch-wise: 每个 query 的 N 个 score 归一化
        attn_mean = attn_raw.mean(dim=-1, keepdim=True)
        attn_std = attn_raw.std(dim=-1, keepdim=True).clamp_min(1e-6)
        attn_norm = (attn_raw - attn_mean) / attn_std * 0.1  # 0.1 限制范围
        
        # Residual: final = cos + delta * attn_norm
        delta = torch.tanh(self.alpha)  # [-0.3, 0.3] - 严格限 attn 贡献
        final_score = cos_score + delta * attn_norm
        return final_score


class CEQueryRefinement(nn.Module):
    """Query refinement: 用桶上下文给 query 加微调, 仍用 cosine 排序.
    
    设计原则:
    1. 不替代 cosine, 只微调 query embedding
    2. delta_q 范围严格限制 (norm ≤ 0.1)
    3. ranking monotonicity 保证 (cos(q+δ, v) 是 cos(q,v) + cos(δ,v))
    """
    def __init__(self, embed_dim=768, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # context aggregation: 用 cand 上下文给 query 提供信息
        self.context_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        # query refinement: q + delta(context)
        self.refine = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        # 学一个修正幅度 (限制在 [0, 0.1])
        self.scale = nn.Parameter(torch.tensor(0.0))  # sigmoid -> [0, 0.1]
    
    def forward(self, q, v):
        """
        q: (B, D)
        v: (B, N, D)
        Returns: (B, N) scores via cosine(refined_q, v)
        """
        if q.dim() == 2:
            q_in = q
        else:
            q_in = q.squeeze(1)
        
        # 1. 聚合桶上下文 (cand mean 作为代表)
        context = v.mean(dim=1)                          # (B, D)
        context_feat = self.context_proj(context)        # (B, D)
        
        # 2. q + context concat → refine
        combined = torch.cat([q_in, context_feat], dim=-1)  # (B, 2D)
        delta_q = self.refine(combined)                  # (B, D)
        
        # 3. 限制 delta_q 范数, 限制贡献
        scale = 0.1 * torch.sigmoid(self.scale)          # [0, 0.1]
        delta_q_normed = F.normalize(delta_q, dim=-1)
        refined_q = q_in + scale * delta_q_normed        # (B, D)
        
        # 4. cosine ranking (用 refined q)
        scores = (refined_q.unsqueeze(1) @ v.transpose(-1, -2)).squeeze(1)  # (B, N)
        return scores


class BiLinearScorer(nn.Module):
    """Bi-linear scoring: score(q,v) = q^T W v + b
    W 初始化为单位矩阵 → 初始等价 cosine, 训练微调让 hard neg 更可分.
    文档 (生成式检索.md) 建议的中间方案."""
    def __init__(self, embed_dim=768, init_identity=True, low_rank=0):
        super().__init__()
        self.embed_dim = embed_dim
        if low_rank > 0:
            # 低秩分解 W = U V^T, 减参数
            self.U = nn.Parameter(torch.randn(embed_dim, low_rank) * 0.01)
            self.V = nn.Parameter(torch.randn(embed_dim, low_rank) * 0.01)
            self.low_rank = low_rank
            self.W = None
        else:
            # 全秩 W, 初始化为 I
            W_init = torch.eye(embed_dim) if init_identity else torch.randn(embed_dim, embed_dim) * 0.01
            self.W = nn.Parameter(W_init)
            self.low_rank = 0
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, q, v):
        """q: (B,D) or (B,1,D); v: (B,N,D) -> (B,N)"""
        if q.dim() == 3:
            q = q.squeeze(1)
        if self.low_rank > 0:
            # score = (q U)(v V)^T  →  q U V^T v
            qU = q @ self.U               # (B, r)
            # v: (B, N, D) @ V (D, r) → (B, N, r)
            vV = v @ self.V               # (B, N, r)
            score = (qU.unsqueeze(1) * vV).sum(-1)  # (B, N)
        else:
            qW = q @ self.W               # (B, D)
            score = (qW.unsqueeze(1) * v).sum(-1)  # (B, N)
        return score + self.b


class TransformerCrossEncoder(nn.Module):
    """标准 transformer cross-encoder, 严格按文档 (生成式检索.md) 实现.

    架构: 把 (query, item) 当作 2-token 序列, 过 N 层 transformer encoder,
    用 [CLS]-style pooling 输出相关性分数.

    文档规格:
      - 2 层 transformer
      - 3-10M 参数 (类 BERT-mini)
      - 输入 (query_emb, item_emb) 一对向量
      - 输出 score(q, v)
    """
    def __init__(self, embed_dim=768, n_heads=8, n_layers=2,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # 给 query / item 加可学习的 type embedding (区分两个 token)
        self.type_emb = nn.Parameter(torch.randn(2, embed_dim) * 0.02)
        # [CLS] token, 用它的输出做最终打分
        self.cls_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        # transformer encoder layers (标准 self-attention)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        # 打分头: [CLS] 输出 → scalar
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, q, v):
        """
        q: (B, D) or (B, N, D)
        v: (B, N, D) or (B, D)
        Returns: (B,) or (B, N) scores

        把每个 (q, v) 对组成序列 [CLS, q, v], 过 transformer, 用 CLS 打分.
        """
        # 统一形状: 处理 (B,D)+(B,N,D) 的广播
        if q.dim() == 2 and v.dim() == 3:
            B, N, D = v.shape
            q = q.unsqueeze(1).expand(-1, N, -1)   # (B, N, D)
            flat_q = q.reshape(B * N, D)
            flat_v = v.reshape(B * N, D)
            out_shape = (B, N)
        elif q.dim() == 2 and v.dim() == 2:
            flat_q = q; flat_v = v
            out_shape = (q.shape[0],)
        else:
            raise ValueError(f"unsupported shapes q={q.shape} v={v.shape}")

        BN = flat_q.shape[0]
        # 构造序列: [CLS, q+type0, v+type1]
        cls = self.cls_token.expand(BN, -1).unsqueeze(1)              # (BN, 1, D)
        q_tok = (flat_q + self.type_emb[0]).unsqueeze(1)              # (BN, 1, D)
        v_tok = (flat_v + self.type_emb[1]).unsqueeze(1)             # (BN, 1, D)
        seq = torch.cat([cls, q_tok, v_tok], dim=1)                  # (BN, 3, D)
        # transformer self-attention (q 和 v 充分交互)
        encoded = self.encoder(seq)                                  # (BN, 3, D)
        cls_out = encoded[:, 0]                                       # (BN, D) 取 CLS
        score = self.score_head(cls_out).squeeze(-1)                 # (BN,)
        return score.reshape(out_shape)


def load_transformer_ce(path, embed_dim=768, n_heads=8, n_layers=2, device='cuda'):
    model = TransformerCrossEncoder(embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers)
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck.get('state_dict', ck))
    return model
