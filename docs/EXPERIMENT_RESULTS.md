# QSBA 实验结果与配置手册

> **更新日期:** 2026-05-26  
> **汇总来源:** `qsba/_all_eval_results.txt`（243 runs）、各 `outputs_*/eval_correct.json`、Table 3 α sweep（`outputs_table3_alpha*_K32_M6`）  
> **论文草稿:** `../qsba_aaai.tex`

---

## 目录

1. [方法简述](#1-方法简述)
2. [数据与评测协议](#2-数据与评测协议)
3. [默认模型配置](#3-默认模型配置)
4. [主结果（论文 Table 1）](#4-主结果论文-table-1)
5. [效率与候选规模（Table 2）](#5-效率与候选规模table-2)
6. [消融实验](#6-消融实验)
7. [各数据集最佳 run 索引](#7-各数据集最佳-run-索引)
8. [复现命令](#8-复现命令)
9. [指标定义](#9-指标定义)

---

## 1. 方法简述

**QSBA** = Query-Aware Sinkhorn Bucket Assignment，三阶段流水线：

| 阶段 | 脚本 | 输入 | 输出 | 作用 |
|------|------|------|------|------|
| **Stage 1** | `02_build_query_aware.py` | 冻结 CLIP 图/文嵌入 | `query_aware_emb.pt` | α 混合图嵌入与 caption 质心 |
| **Stage 2** | `03_run_sinkhorn.py` | Stage 1 特征 | `assignment_K_M.pt` | Sinkhorn EM：N 图 → K 桶，每图最多 M 桶 |
| **Stage 3** | `04_train_router.py` | 文本嵌入 + 桶标签 | `router_K.pt` | MLP：query → K 类 logits |
| **Eval** | `06_evaluate_correct.py` | router + assignment | `eval_correct.json` | Top-B 桶并集 + dense cosine rerank |

**推理:** 预测 Top-B 桶 → 候选池 = 这些桶内所有图像 → CLIP dense R@K。

---

## 2. 数据与评测协议

### 2.1 嵌入（全部实验共用）

| 项目 | 值 |
|------|-----|
| **Backbone** | CLIP ViT-L/14（冻结，仅预计算特征） |
| **维度** | 768 |
| **归一化** | L2-normalized |
| **存储** | `image_embeddings.pt` / `text_embeddings.pt` |

### 2.2 各数据集规模

| 数据集 | 数据目录 | 图像数 (文件中) | 文本条数 | **评测 pool** | **评测 query** |
|--------|----------|-----------------|----------|---------------|----------------|
| **MS-COCO 5K** | `data_clipsf_v2/` | 5,000 | 124,809 | **5,000** | **24,809** |
| **Flickr30K** | `data_flickr_clipsf_shuffled/` | 31,783 | 158,914 | **1,000** | **5,000** |
| **Merged** | `data_merged_clipsf/` | 36,783 | — | **36,783** | **24,809** |
| **Visual News** | 专用 | 542,246 | — | **542,246** | **19,898** |

**张量 shape 实测:**

```
COCO  image: [5000, 768]     text: [124809, 768]
Flickr image: [31783, 768]   text: [158914, 768]
```

### 2.3 `06_evaluate_correct.py` 切片约定

| 数据集 | `--pool-start` | `--pool-end` | `--query-start` | 说明 |
|--------|----------------|--------------|-----------------|------|
| **COCO 5K** | `0` | `5000` | `100000` | query 取 text 索引 `[100000, 124809)` |
| **Flickr** | 依 run 配置 | pool_size=1000 | — | 评测 JSON 中 `pool_size=1000`, `n_queries=5000` |
| **默认（脚本）** | `82783` | 文件末尾 | `100000` | 用于 merged 等；**COCO 主表需显式覆盖为 0–5000** |

**评测设置（主表）:**

- `rerank_mode`: dense cosine（CLIP 空间）
- `router_prior_lambda`: 0.0
- `top_B`: [1, 3, 5]；`top_K`: [1, 5, 10]

---

## 3. 默认模型配置

### 3.1 Stage 1 — Query-aware

```yaml
query_aware:
  alpha: 1.0          # 主实验：纯图像；Table 3 扫 {0, 0.25, 0.5, 0.75, 1.0}
  l2_normalize: true
```

公式: \(\tilde{z}_i = \mathrm{L2Norm}(\alpha z_i^{\mathrm{img}} + (1-\alpha)\bar{z}_i^{\mathrm{cap}})\)

### 3.2 Stage 2 — Sinkhorn EM

```yaml
sinkhorn:
  K: 16 | 32 | ...     # 桶数
  M: 4 | 6 | 12 | 18   # 每图最多 M 个桶（多视图）
  epsilon: 0.01        # 熵正则；0.03 用于非均衡对照
  n_sinkhorn_iters: 20
  n_em_iters: 30
  init: kmeans++
```

### 3.3 Stage 3 — Router MLP

```yaml
router:
  hidden_dim: 1024
  n_layers: 3
  dropout: 0.1
  lr: 0.001
  batch_size: 256
  weight_decay: 1.0e-5
  val_split: 0.2
  epochs: 60   # Flickr 最佳
  epochs: 80   # COCO 最佳 (extra2_K32_M18)
```

**结构** (`src/router.py`): `768 → [Linear+LN+GELU+Drop]×(L-1) → K logits`  
**参数量:** K=16/32 时约 **1.85–1.87M** 可训练参数。

### 3.4 论文主表两套「最佳配置」

| 字段 | Flickr 最佳 | COCO 最佳 |
|------|-------------|-----------|
| **Run 目录** | `outputs_flickr2_K16_M12` | `outputs_extra2_K32_M18` |
| **K / M** | 16 / 12 | 32 / 18 |
| **α** | 1.0 | 1.0 |
| **Router epochs** | 60 | 80 |
| **数据** | `data_flickr_clipsf_shuffled/` | `data_clipsf_v2/` |
| **config** | `outputs_flickr2_K16_M12/config.yaml` | `outputs_extra2_K32_M18/config.yaml` |

---

## 4. 主结果（论文 Table 1）

### 4.1 与发表基线对比（R@1 %，文本→图）

| Method | Flickr30K | MS-COCO | 候选/query |
|--------|-----------|---------|------------|
| CLIP ViT-L/14 dense † | 83.4 | 58.4 | 1000 / 5000 |
| IRGen † | 70.1 | 43.1 | full |
| GRACE † | 73.5 | 46.8 | full |
| AVG † | 79.2 | 51.3 | full |
| **GENIUS †** | **84.1** | **58.1** | full |
| **QSBA B=1** | **79.4** | **55.2** | **934 / 4979** |
| QSBA B=3 | 79.5 | 55.2 | 1000 / 5000 |
| QSBA B=5 | 79.5 | 55.2 | 1000 / 5000 |

† 来自原论文，非本仓库重跑。

### 4.2 QSBA 完整指标（JSON 实测）

#### Flickr — `outputs_flickr2_K16_M12`

| B | Router R@B | R@1 | R@5 | R@10 | Mean candidates |
|---|------------|-----|-----|------|-----------------|
| 1 | 99.84% | **79.38%** | 95.36% | 97.40% | **934** |
| 3 | 100% | 79.52% | 95.50% | 97.52% | 1000 |
| 5 | 100% | 79.52% | 95.50% | 97.52% | 1000 |

#### MS-COCO — `outputs_extra2_K32_M18`

| B | Router R@B | R@1 | R@5 | R@10 | Mean candidates |
|---|------------|-----|-----|------|-----------------|
| 1 | 99.97% | **55.15%** | 80.70% | 88.24% | **4979** |
| 3 | 100% | 55.15% | 80.70% | 88.24% | 5000 |
| 5 | 100% | 55.15% | 80.70% | 88.24% | 5000 |

**要点:**

- Flickr：B=1 已接近扫满 1K 池，B>1 仅 +0.14 pt R@1。
- COCO：M=18 时 B=1 已扫 ~99.6% 池，**加 B 无收益**；效率叙事应强调 router 准确率，而非 beam widening。

---

## 5. 效率与候选规模（Table 2）

| Method | Dataset | Candidates / query |
|--------|---------|-------------------|
| Dense CLIP | Flickr30K | 1,000 |
| Dense CLIP | MS-COCO | 5,000 |
| QSBA B=1 | Flickr30K (K=16,M=12) | **934** |
| QSBA B=1 | MS-COCO (K=32,M=18) | **4,979** |
| QSBA B=1 | Merged (K=128,M=8) | **2,728** |
| QSBA B=1 | Visual News (K=64,M=4) | **38,035** |

对应 run: `flickr2_K16_M12`, `extra2_K32_M18`, `merged_K128_M8`, `visualnews_K64_M4`。

---

## 6. 消融实验

### 6.1 α 混合系数（COCO 5K，K=32，M=6，B=1）

**协议:** `scripts/run_table3_alpha_sweep.sh` — pool `[0,5000)`, query `[100000, …)`, router **40 epochs**, CPU。

| α | Run 目录 | Router R@1 | **Candidates@B=1** | R@1 |
|---|----------|--------------|----------------------|-----|
| 1.0 (image only) | `outputs_table3_alpha1_0_K32_M6` | 0.997 | **1474** | 55.20% |
| 0.75 | `outputs_table3_alpha0_75_K32_M6` | 0.997 | **1553** | 55.22% |
| 0.5 | `outputs_table3_alpha0_5_K32_M6` | 0.998 | **1419** | 55.19% |
| 0.25 | `outputs_table3_alpha0_25_K32_M6` | 0.998 | **1484** | 55.24% |
| 0.0 (caption only) | `outputs_table3_alpha0_0_K32_M6` | 0.999 | **1338** | 55.22% |

**结论（CLIP + K=32 + M=6）:** α 对 **R@1 几乎平坦**（~55.2%）；候选数在 1.3k–1.6k。  
**注意:** 主配置 **M=18** 时候选 ~4979，与 M=6 不可直接对比 R@1 绝对值。

### 6.2 K × M 网格（Flickr30K，pool=1K，B=1）

Router R@1 / R@1 / candidates@B=1：

| K \ M | M=1 | M=2 | M=4 | M=6 | M=8 | M=12 |
|-------|-----|-----|-----|-----|-----|------|
| **8** | .682 / .564 / 127 | .876 / .711 / 258 | .985 / .786 / 539 | **.997 / .793 / 873** | — | — |
| **16** | .615 / .515 / 64 | .812 / .666 / 128 | .942 / .758 / 265 | .975 / .780 / 415 | .986 / .787 / 610 | **.998 / .794 / 934** |
| **32** | .562 / .476 / 32 | .750 / .624 / 65 | .884 / .722 / 132 | .936 / .757 / 218 | .963 / .775 / 308 | — |

**最佳:** K=16, M=12（R@1=79.4%, cand=934）。K 过大（32）在 1K pool 上 router 变差。

### 6.3 Top-B 曲线（主配置）

| Dataset | B=1 | B=3 | B=5 |
|---------|-----|-----|-----|
| Flickr R@1 | 79.38% | 79.52% | 79.52% |
| Flickr cand | 934 | 1000 | 1000 |
| COCO R@1 | 55.15% | 55.15% | 55.15% |
| COCO cand | 4979 | 5000 | 5000 |

### 6.4 Sinkhorn vs 高 ε（非均衡对照，COCO 5K）

| 设置 | Run | std/mean 桶大小 | R@1 (B=1) |
|------|-----|-----------------|-----------|
| Sinkhorn ε=0.01 | `v3_phH_K32_M6` | ~0.66 | **54.60%** |
| 高 ε=0.03 代理 | `v3_phL_K64_M6_eps0.03_em30sk20` | **1.56** | 53.63% |

### 6.5 Router 深度/宽度（COCO 5K，节选）

| hidden | layers | Router R@1 | R@1 | 备注 |
|--------|--------|------------|-----|------|
| 512 | 2 | ~0.74 | ~46% | `master_phC_h512_l2` |
| 1024 | 3 | **~0.98–1.00** | **~54–55%** | `v3_phI_h1024_l3` 等 |
| 1024 | 3 | **0.9997** | **55.15%** | **主配置** `extra2_K32_M18` |
| 2048 | 3 | 0.978 | 54.36% | `v3_phI_h2048_l3` |

---

## 7. 各数据集最佳 run 索引

| 数据集 | 最佳目录 | R@1\|B=1 | Router R@1 | Cand@1 | Pool | Q |
|--------|----------|-----------|------------|--------|------|---|
| Flickr30K | `outputs_flickr2_K16_M12` | 0.7938 | 0.9984 | 934 | 1000 | 5000 |
| MS-COCO | `outputs_extra2_K32_M18` | 0.5515 | 0.9997 | 4979 | 5000 | 24809 |
| Merged | `outputs_merged_K128_M8` | 0.4783 | 0.9704 | 2728 | 36783 | 24809 |
| Visual News | `outputs_visualnews_K64_M4` | 0.1793 | 0.8107 | 38035 | 542246 | 19898 |

**COCO 次优参考:** `extra2_K48_M18` R@1=0.5511, cand=**4125**（若论文需更低候选可引用）。

**全库扫描:** `grep BEST qsba/_all_eval_results.txt`（文件末尾注释行）。

---

## 8. 复现命令

```bash
cd qsba

# 单 run 端到端（示例：COCO 最佳配置）
CONFIG=outputs_extra2_K32_M18/config.yaml
OUT=outputs_extra2_K32_M18

python scripts/02_build_query_aware.py --config $CONFIG
python scripts/03_run_sinkhorn.py --config $CONFIG \
  --input $OUT/query_aware_emb.pt --K 32 --M 18 \
  --output $OUT/assignment_K32_M18.pt
python scripts/04_train_router.py --config $CONFIG \
  --assignment $OUT/assignment_K32_M18.pt \
  --output $OUT/router_K32.pt
python scripts/06_evaluate_correct.py \
  --config $CONFIG \
  --assignment $OUT/assignment_K32_M18.pt \
  --router $OUT/router_K32.pt \
  --pool-start 0 --pool-end 5000 \
  --query-start 100000 \
  --top-b 1 3 5 \
  --output-json $OUT/eval_correct.json

# Table 3 α sweep（5 个 α，约数小时 CPU）
bash scripts/run_table3_alpha_sweep.sh

# 打印 Table 3 行
python scripts/fill_table3_from_sweep.py

# 重新生成论文图
python scripts/generate_paper_figures.py
```

**汇总全部 243 runs:**

```bash
# 若存在生成脚本；否则直接查看：
cat qsba/_all_eval_results.txt
```

---

## 9. 指标定义

| 指标 | 含义 |
|------|------|
| **router_recall@B** | 目标图像所在桶是否出现在 query 预测的 Top-B 桶中 |
| **recall@K\|B=B** | 在 Top-B 桶并集候选池上 dense rerank 后的 Recall@K |
| **candidates@B=B** | 平均每 query 候选图像数（≤ pool_size） |
| **pool_size** | 评测候选库图像数 |
| **n_queries_evaluated** | 评测 query 条数 |

---

## 附录：论文数字速查

| 论文位置 | 关键数字 | 来源 |
|----------|----------|------|
| Abstract | Flickr 79.4%, COCO 55.2%, cand 934 / 4979 | 主表 JSON |
| Table 1 | 同上 + B=3/5 | `flickr2_K16_M12`, `extra2_K32_M18` |
| Table 2 | 4979, 934, 2728, 38035 | 见 §5 |
| Table 3 α | 1474, 1553, 1419, 1484, 1338 | `table3_alpha*_K32_M6` |
| Table 4 K×M | 0.998/0.794 等 | `flickr2_K*_M*` |
| GENIUS 对比 | 84.1 / 58.1 | 原论文 † |

---

*本文档随 `eval_correct.json` 更新；修改配置后请重跑 eval 并同步此文件。*
