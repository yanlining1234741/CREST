# CREST (implementation notes)

**CREST: Collapse-Free Routing with Equipartitioned Semantic Buckets for Cross-Modal Retrieval**

跨模态检索中的**共享语义桶分配**实验框架。目标:把 N 个 item 分配到 K 个桶,
使得 query 能准确路由到目标 item 所在的桶,且粒度尽可能细。

---

## 一、核心思路

### 问题
传统 RVQ 在 image embedding 空间做 k-means → 桶按"视觉相似性"划分。
但 router 是从 **text query** 预测桶,文本视角和视觉视角不一致 → router recall 受限。

### 三层解药
1. **Stage 1 - Query-Aware 表征**:把每个 item 的 caption embedding 混入 item embedding,
   让聚类反映"query 视角的等价类"。
2. **Stage 2 - Sinkhorn 平衡分配**:用 entropy-regularized OT 求解桶分配,
   强制每个桶大小 ≈ N/K,避免 codebook collapse。
3. **Stage 3 - Router 训练 + Top-B**:训练 query → bucket 分类器,
   推理时取 top-B,recall-budget Pareto 前沿就出来了。

### 关键设计:**模块化诊断**
每个 Stage 输出一个独立文件,你可以:
- 单独运行任何一个 Stage
- 替换某个 Stage 的输出,看下游怎么变
- 比较不同配置下的中间产物 → 定位是哪一个模块影响了最终 router recall

---

## 二、目录结构

```
crest/
├── README.md                          # 本文件
├── configs/
│   └── coco_siglip2.yaml              # 默认配置
├── src/
│   ├── data.py                        # 加载 SigLIP2 embedding
│   ├── query_aware.py                 # Stage 1: query-aware 表征
│   ├── sinkhorn.py                    # Stage 2: Sinkhorn 平衡分配
│   ├── router.py                      # Stage 3: router 网络 + 训练
│   ├── evaluate.py                    # 评估指标
│   └── utils.py                       # 通用工具
├── scripts/
│   ├── 01_prepare_data.py             # 验证数据文件
│   ├── 02_build_query_aware.py        # 跑 Stage 1
│   ├── 03_run_sinkhorn.py             # 跑 Stage 2
│   ├── 04_train_router.py             # 跑 Stage 3
│   ├── 05_evaluate.py                 # 完整 end-to-end 评估
│   ├── ablate.py                      # 消融:扫 α / K / M / B
│   └── run_pipeline.sh                # 一键跑完整 pipeline
├── tests/
│   ├── test_query_aware.py
│   ├── test_sinkhorn.py
│   └── test_router.py
└── outputs/                           # 所有中间产物 + 日志
```

---

## 三、数据格式约定

需要你提供两个 `.pt` 文件 (基于 SigLIP2-SO400M,d=1152):

**`image_embeddings.pt`**:
```python
{
    'features': torch.Tensor [N, 1152],  # L2-normalized
    'image_ids': List[int],              # 长度 N
}
```

**`text_embeddings.pt`** (COCO 每张图 5 个 caption):
```python
{
    'features': torch.Tensor [M, 1152],  # L2-normalized, M ≈ 5N
    'image_ids': List[int],              # 每条 caption 对应的 image_id
    'caption_ids': List[int],            # 可选
}
```

如果你的格式不同,改 `src/data.py` 里的 `load_embeddings()` 函数即可。

---

## 四、运行步骤

### Step 0: 环境
```bash
pip install torch>=2.0 numpy scikit-learn tqdm pyyaml
```

### Step 1: 准备配置
编辑 `configs/coco_siglip2.yaml`,填入你的 embedding 文件路径。

### Step 2: 验证数据
```bash
python scripts/01_prepare_data.py --config configs/coco_siglip2.yaml
```
输出: `outputs/data_stats.txt`,确认 embedding 加载正确、维度对、归一化对。

### Step 3: 构造 query-aware 表征 (Stage 1)
```bash
python scripts/02_build_query_aware.py --config configs/coco_siglip2.yaml --alpha 0.5
```
输出: `outputs/query_aware_emb_alpha0.5.pt`
- α=1.0 → 纯 image (baseline)
- α=0.5 → 平衡
- α=0.0 → 纯 caption 平均

### Step 4: Sinkhorn 桶分配 (Stage 2)
```bash
python scripts/03_run_sinkhorn.py --config configs/coco_siglip2.yaml \
    --input outputs/query_aware_emb_alpha0.5.pt \
    --K 256 --M 1
```
输出: `outputs/assignment_K256_M1.pt` (包含 assignment, centroids, bucket stats)

### Step 5: 训练 router (Stage 3)
```bash
python scripts/04_train_router.py --config configs/coco_siglip2.yaml \
    --assignment outputs/assignment_K256_M1.pt \
    --epochs 30
```
输出: `outputs/router_K256.pt` + 训练日志

### Step 6: 完整评估
```bash
python scripts/05_evaluate.py --config configs/coco_siglip2.yaml \
    --assignment outputs/assignment_K256_M1.pt \
    --router outputs/router_K256.pt
```
输出: `outputs/eval_K256.json`,包含:
- Router Recall@{1,3,5,10}  ← **核心指标**
- 每 query 候选数 (bucket size × B)
- 桶大小分布 (mean/std/min/max)
- End-to-end Recall@{1,5,10}

### Step 7: 一键跑完整 pipeline
```bash
bash scripts/run_pipeline.sh
```

---

## 五、消融实验 (定位模块影响)

```bash
# 扫 α (Stage 1 的影响)
python scripts/ablate.py --sweep alpha --values 1.0,0.7,0.5,0.3,0.0

# 扫 K (粒度的影响)
python scripts/ablate.py --sweep K --values 64,128,256,512,1024

# 扫 M (multi-view 的影响)
python scripts/ablate.py --sweep M --values 1,2,3,4

# 扫 B (top-B beam search 的影响)
python scripts/ablate.py --sweep B --values 1,3,5,10,20

# 关闭 Sinkhorn,只用 k-means (验证 Sinkhorn 的影响)
python scripts/ablate.py --sweep sinkhorn --values on,off
```

每个消融生成 `outputs/ablate_<param>.csv`,直接读表对比即可。

---

## 六、如何定位"哪个模块影响最大"

读 `outputs/<run>/diagnostic.json`,每个 Stage 都有自己的诊断指标:

| Stage | 关键诊断指标 | 健康范围 (COCO 5K) |
|---|---|---|
| 1 (query-aware) | text-image 在新空间的 cosine sim | 0.85+ |
| 2 (Sinkhorn) | 桶大小 std / mean | < 0.1 |
| 2 (Sinkhorn) | query co-occurrence in same bucket | > 0.6 |
| 3 (router) | training accuracy | > 90% |
| 3 (router) | val router recall@1 | > 85% |
| End-to-end | Recall@1 | > 60% |

诊断逻辑:
- 如果 Stage 1 sim 低 → 你的 caption 数据有问题
- 如果 Stage 2 std/mean 高 → Sinkhorn 没收敛,加迭代次数
- 如果 Stage 2 co-occurrence 低 → α 调小,让 query 视角主导
- 如果 Stage 3 train acc 高但 val recall 低 → 过拟合,加 dropout
- 如果以上都正常但 R@1 低 → 桶太大,K 增大或 M 增大

---

## 七、单元测试

```bash
pytest tests/                       # 跑全部
pytest tests/test_sinkhorn.py -v    # 单独测 Sinkhorn 实现
```

测试覆盖:
- Sinkhorn 迭代后的列和确实接近 N/K
- Query-aware 混合后向量仍 L2 归一化
- Router 在合成数据上能学到 trivial 映射 (sanity check)

---

## 八、下一步扩展

- Flickr30K: 改配置文件即可
- 层级化 (K1 × K2): 在 Stage 2 后串两次 Sinkhorn
- 你的 RVQ baseline: 用 `scripts/05_evaluate.py --baseline rvq` 直接对比
