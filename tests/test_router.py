"""测试 Router。"""
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router import RouterMLP, evaluate_router, multi_label_ce, train_router
from src.utils import set_seed, l2_normalize


def test_routermlp_forward_shape():
    model = RouterMLP(embed_dim=64, K=10, hidden_dim=32, n_layers=2)
    x = torch.randn(8, 64)
    logits = model(x)
    assert logits.shape == (8, 10)


def test_multi_label_ce_correctness():
    """multi_label_ce: 当 M=1 时,应等价于普通 CE。"""
    logits = torch.randn(4, 5)
    targets_single = torch.tensor([0, 1, 2, 3]).unsqueeze(1)  # [4, 1]
    our_loss = multi_label_ce(logits, targets_single)
    standard_loss = torch.nn.functional.cross_entropy(logits, targets_single.squeeze(1))
    assert torch.allclose(our_loss, standard_loss, atol=1e-5)


def test_router_can_learn_trivial_mapping():
    """Sanity check: 在 trivial 数据 (text=onehot, bucket=arg) 上,
    router 应该能学到 100% 准确率。"""
    set_seed(0)
    D, K = 16, 4
    # 每个桶对应一个 one-hot text
    n_per_bucket = 100
    xs, ys = [], []
    for k in range(K):
        for _ in range(n_per_bucket):
            x = torch.zeros(D)
            x[k] = 1.0
            x = x + 0.05 * torch.randn(D)
            xs.append(l2_normalize(x))
            ys.append(torch.tensor([k]))
    X = torch.stack(xs)
    Y = torch.stack(ys)

    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = RouterMLP(embed_dim=D, K=K, hidden_dim=32, n_layers=2, dropout=0.0)
    log = train_router(model, loader, loader, epochs=20,
                       lr=1e-2, weight_decay=0.0,
                       device="cpu", top_B_list=(1,), verbose=False)

    final_recall = log[-1]["recall@1"]
    assert final_recall > 0.9, f"router cannot learn trivial mapping, got R@1={final_recall}"
