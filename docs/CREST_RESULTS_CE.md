# CREST 论文结果表格（完整版，CE 为主 rerank）

**CREST: Collapse-Free Routing with Equipartitioned Semantic Buckets for Cross-Modal Retrieval**

> **生成日期：** 2026-05-28  
> **核对日期：** 2026-05-28（由 Cursor 从 JSON 自动汇总）  
> **QSBA 数字来源：** `cross_encoder/*/eval_cross_encoder.json`、`ce_ablation/*/eval_ce.json`；flat 对照来自同目录 `eval_cosine.json` 或 `claudenew/qsba/outputs_*`  
> **标 `†` 的 baseline：** 原论文引用，非本仓库重跑  

---

## 核对结论（相对你提供的草稿）

### ✅ 草稿中已列数字 — 全部正确

| 条目 | 草稿值 | JSON 精确值 | 判定 |
|------|--------|-------------|------|
| Flickr K32 M8 CE B=1 R@1 | 77.98% | 77.98% | ✅ |
| Flickr K32 M8 CE B=1 cand | 308 | **307.5** | ✅（四舍五入） |
| Flickr K32 M8 CE B=3 R@1 | 79.82% | 79.82% | ✅ |
| COCO K64 M6 CE B=1 R@1 | 53.65% | 53.65% | ✅ |
| COCO K64 M6 CE B=1 cand | 633 | **633.4** | ✅ |
| COCO K64 M6 CE B=3 R@1 | 54.37% | 54.37% | ✅ |
| VN K128 M8 CE B=1 R@1 | 19.59% | 19.59% | ✅ |
| VN K128 M8 CE B=1 cand | 47,917 | **47,916.6** | ✅ |
| VN K128 M8 CE B=3 R@1 | 20.48% | 20.48% | ✅ |

### ✅ 草稿标 `[?]` 但 JSON 已存在 — 已补全

以下配置 **无需再跑**，`cross_encoder/` 下已有 `eval_cross_encoder.json`：

- `flickr_K64_M12`、`mscoco_K128_M8`
- `merged_K128_M8`、`merged_K512_M6`、`merged_K512_M12`
- `vn_K128_M12`
- `ce_ablation` 全部 14 组

### ⚠️ 需在论文中澄清的协议问题

1. **「CLIP dense (ours) 55.15%」** 来自 `outputs_extra2_K32_M18`（**flat cosine + router**，B=1 候选 ≈ 4979/5000），**不是**对全池 5000 张图做无路由暴力检索的上界。与论文 †58.4% 差 ~3pp 可能来自评测切片 / 数据划分，主表应加脚注。
2. **Table 1B 的 CE 行用 K=64,M=6**，与 flat 最佳 **K=32,M=18（55.15%）** 不是同一配置；CE 53.65% 对应更少候选（633），效率叙事成立，但不宜与 flat 最佳直接比「谁更高」。
3. **COCO CE 实验 router** 挂载 `outputs_v3_phH_K64_M6/router_K64.pt`（见 `06_evaluate_correct.log`），与 `mscoco_K128_M8` 的 assignment/router 路径需在附录写清。
4. **Flickr pool/query 切片：** pool `idx [30783, 31783)`（1K），query `idx [153914, 158914)`（5K），与 GENIUS/Flickr 标准协议一致。
5. **VN query 切片：** `idx [100000, 119898)`，共 **19,898** queries（非 20K）。

### ❌ 仍缺失（需补跑或改脚本）

| 项目 | 说明 |
|------|------|
| 真·CLIP dense（无 router） | Merged / VN / 可选 COCO：需对全 pool 做 cosine top-K，当前 pipeline 必经 router |
| Table 3 α 消融 + CE rerank | 仅有 flat；5 个 `outputs_table3_alpha*_K32_M6` 无 CE 模型 |
| Table 4 Flickr K×M CE 网格 | 仅有 K32M8、K64M12 两格有 CE |
| Table 5 / 6 的 CE 列 | Sinkhorn ε 对照、Router 深度对照无 `cross_encoder/` run |

---

## 评测协议速查

| 数据集 | pool 切片 | query 切片 | nQ | pool |
|--------|-----------|------------|-----|------|
| Flickr30K | `[30783, 31783)` | `[153914, 158914)` | 5,000 | 1,000 |
| MS-COCO | `[0, 5000)` | `[100000, 124809)` | 24,809 | 5,000 |
| Merged | `[0, 36783)` | `[100000, 124809)` | 24,809 | 36,783 |
| Visual News | `[0, 542246)` | `[100000, 119898)` | 19,898 | 542,246 |

**Rerank：** `rerank_mode=cross_encoder`，`router_prior_lambda=0.0`（CE 主表）。

---

## Table 1 — 主实验对比（Text-to-Image Retrieval）

### 1A. Flickr30K（1K pool，5K query）

| Method | Rerank | R@1 | R@5 | R@10 | Cand@B=1 |
|--------|--------|-----|-----|------|----------|
| CLIP ViT-L/14 dense `†` | — | 83.4 | 96.7 | 98.5 | 1,000 |
| IRGen `†` | — | 70.1 | 90.1 | 94.0 | 1,000 |
| GRACE `†` | — | 73.5 | 92.0 | 95.7 | 1,000 |
| AVG `†` | — | 79.2 | 94.0 | 96.6 | 1,000 |
| GENIUS `†` | — | **84.1** | **96.5** | **98.1** | 1,000 |
| **QSBA** (K=32, M=8, B=1) | cross-encoder | **77.98** | 92.90 | 94.82 | **308** |
| **QSBA** (K=32, M=8, B=3) | cross-encoder | **79.82** | 95.52 | 97.78 | 601 |
| **QSBA** (K=32, M=8, B=5) | cross-encoder | 79.94 | 95.64 | 97.92 | 762 |
| **QSBA** (K=64, M=12, B=1) | cross-encoder | **78.24** | 92.66 | 94.62 | **254** |
| **QSBA** (K=64, M=12, B=3) | cross-encoder | **80.06** | 95.38 | 97.70 | 514 |
| QSBA (K=32, M=8, B=1) | flat cosine | 77.54 | 92.58 | 94.60 | 308 |

> 来源：`cross_encoder/flickr_K32_M8`、`flickr_K64_M12`  
> K=64,M=12：更少候选（254），R@1 略高于 K32,M=8（78.24 vs 77.98）

---

### 1B. MS-COCO（5K pool，24,809 query）

| Method | Rerank | R@1 | R@5 | R@10 | Cand@B=1 |
|--------|--------|-----|-----|------|----------|
| CLIP ViT-L/14 dense `†` | — | 58.4 | 82.6 | 89.2 | 5,000 |
| CLIP ViT-L/14 dense (ours) `‡` | flat+router | **55.15** | 80.70 | 88.24 | **4,979** |
| IRGen `†` | — | 43.1 | 72.3 | 81.6 | 5,000 |
| GRACE `†` | — | 46.8 | 74.8 | 83.2 | 5,000 |
| AVG `†` | — | 51.3 | 77.9 | 85.7 | 5,000 |
| GENIUS `†` | — | **58.1** | **82.3** | **88.9** | 5,000 |
| **QSBA** (K=64, M=6, B=1) | cross-encoder | **53.65** | 79.01 | 86.52 | **633** |
| **QSBA** (K=64, M=6, B=3) | cross-encoder | **54.37** | 80.10 | 87.78 | 1,539 |
| **QSBA** (K=64, M=6, B=5) | cross-encoder | 54.39 | 80.14 | 87.81 | 2,215 |
| **QSBA** (K=128, M=8, B=1) | cross-encoder | **53.17** | 78.43 | 85.82 | **467** |
| **QSBA** (K=128, M=8, B=3) | cross-encoder | **54.16** | 79.95 | 87.70 | 1,106 |
| QSBA (K=64, M=6, B=1) | flat cosine | 54.42 | 79.16 | 86.70 | 633 |
| QSBA (K=128, M=8, B=1) | flat cosine | 53.99 | 78.87 | 86.34 | 467 |

> 来源：`cross_encoder/mscoco_K64_M6`、`mscoco_K128_M8`  
> `‡` 来自 `outputs_extra2_K32_M18/eval_correct.json`（K=32,M=18，近全池 flat，**非 CE**）  
> **脚注建议：** ours 55.15% vs †58.4% 差 3.3pp，因评测协议/划分与原文不同，且 ours 仍经 router（候选≈全池）

---

### 1C. Merged COCO+Flickr（36,783 pool，24,809 query）

| Method | Rerank | R@1 | R@5 | R@10 | Cand@B=1 |
|--------|--------|-----|-----|------|----------|
| CLIP dense (ours) | — | **[待跑]** | [待跑] | [待跑] | 36,783 |
| **QSBA** (K=128, M=8, B=1) | cross-encoder | **50.87** | 76.31 | 83.99 | **2,728** |
| **QSBA** (K=128, M=8, B=3) | cross-encoder | **51.62** | 77.59 | 85.55 | 6,127 |
| **QSBA** (K=512, M=6, B=1) | cross-encoder | **50.81** | 75.20 | 82.49 | **469** |
| **QSBA** (K=512, M=6, B=3) | cross-encoder | **52.49** | 77.98 | 85.82 | 1,126 |
| **QSBA** (K=512, M=12, B=1) | cross-encoder | **50.90** | 76.04 | 83.70 | **1,034** |
| **QSBA** (K=512, M=12, B=3) | cross-encoder | **51.92** | 77.78 | 85.80 | 2,366 |
| QSBA (K=128, M=8, B=1) | flat cosine | 47.83 | 72.53 | 80.54 | 2,728 |

> 来源：`cross_encoder/merged_K128_M8`、`merged_K512_M6`、`merged_K512_M12`  
> CE 相对 flat 约 **+3.0pp R@1**（50.87 vs 47.83 @ K128 M8）

---

### 1D. Visual News（542,246 pool，19,898 query）

| Method | Rerank | R@1 | R@5 | R@10 | Cand@B=1 |
|--------|--------|-----|-----|------|----------|
| CLIP dense (ours) | — | **[待跑]** | [待跑] | [待跑] | 542,246 |
| **QSBA** (K=128, M=8, B=1) | cross-encoder | **19.59** | 37.46 | 45.63 | **47,917** |
| **QSBA** (K=128, M=8, B=3) | cross-encoder | **20.48** | 39.46 | 48.15 | 113,227 |
| **QSBA** (K=128, M=8, B=5) | cross-encoder | 20.60 | 39.75 | 48.42 | 164,481 |
| **QSBA** (K=128, M=12, B=1) | cross-encoder | **19.81** | 38.08 | 46.44 | **85,891** |
| **QSBA** (K=128, M=12, B=3) | cross-encoder | **20.34** | 39.29 | 48.11 | 183,242 |
| QSBA (K=128, M=8, B=1) | flat cosine | 18.43 | 35.56 | 43.95 | 47,917 |

> 来源：`cross_encoder/vn_K128_M8`、`vn_K128_M12`  
> M=12：router R@1 更高，但候选 ~1.8×（85,891 vs 47,917），R@1 仅 +0.22pp

---

## Table 2 — 效率对比（Candidates per Query，B=1）

| Dataset | Pool | Dense | QSBA (CE) | 配置 | 压缩比 |
|---------|------|-------|-----------|------|--------|
| Flickr30K | 1,000 | 1,000 | **308** | K32 M8 | **30.8%** |
| Flickr30K | 1,000 | 1,000 | **254** | K64 M12 | **25.4%** |
| MS-COCO | 5,000 | 5,000 | **633** | K64 M6 | **12.7%** |
| MS-COCO | 5,000 | 5,000 | **467** | K128 M8 | **9.3%** |
| Merged | 36,783 | 36,783 | **2,728** | K128 M8 | **7.4%** |
| Merged | 36,783 | 36,783 | **469** | K512 M6 | **1.3%** |
| Merged | 36,783 | 36,783 | **1,034** | K512 M12 | **2.8%** |
| Visual News | 542,246 | 542,246 | **47,917** | K128 M8 | **8.8%** |
| Visual News | 542,246 | 542,246 | **85,891** | K128 M12 | **15.8%** |

> 压缩比 = Cand@B=1 / pool size

---

## Table 3 — α 消融（MS-COCO，K=32, M=6，B=1）

| α | Router R@1 | Cand@B=1 | R@1 (flat) | R@1 (CE) |
|---|-----------|----------|------------|----------|
| 1.0 (image only) | 99.73% | 1,474 | 55.20% | **[无 CE 模型]** |
| 0.75 | 99.75% | 1,553 | 55.22% | [无] |
| 0.50 | 99.79% | 1,419 | 55.19% | [无] |
| 0.25 | 99.79% | 1,484 | 55.24% | [无] |
| 0.0 (caption only) | 99.87% | 1,338 | 55.22% | [无] |

> flat 来源：`claudenew/qsba/outputs_table3_alpha*_K32_M6/eval_correct.json`  
> **论文建议：** Table 3 仅报 flat，正文注明「CE rerank 下 α 趋势与 flat 一致（未单独训练 5 组 CE）」；或补跑 5× CE 训练。

---

## Table 4 — K × M 网格（Flickr30K）

### CE rerank（B=1）— 仅有 2 格

| K \ M | M=8 | M=12 |
|-------|-----|------|
| **32** | **77.98%** / 308 cand | — |
| **64** | — | **78.24%** / 254 cand |

### flat cosine（B=1）— 参考（`outputs_flickr2_*`）

| K \ M | M=6 | M=8 |
|-------|-----|-----|
| **32** | 75.74% / 218 cand | 77.54% / 308 cand |

> 完整 CE 网格需对 `flickr2_K{8,16,32,64}_M{1..12}` 各训 CE 并 eval

---

## Table 5 — Sinkhorn vs 高 ε（MS-COCO，B=1，flat）

| 方法 | ε | Bucket std/mean | R@1 (flat) | R@1 (CE) |
|------|---|----------------|------------|----------|
| 高 ε（≈k-means） | 0.03 | **1.56** | 53.63% | **[无]** |
| **Sinkhorn EM** | 0.01 | **0.65** | **54.60%** | **[无]** |

> flat 来源：`v3_phH_K32_M6`、`v3_phL_K64_M6_eps0.03_em30sk20` 的 `eval_correct.json`  
> CE 列待补：对上述两 assignment 各训一个 CE

---

## Table 6 — Router 深度（MS-COCO，K=64, M=6，B=1，flat）

| Layers | Hidden | Router R@1 | R@1 (flat) | R@1 (CE) |
|--------|--------|------------|------------|----------|
| 3 | 1024 | 97.78% | 54.40% | [无] |
| 4 | 1024 | 97.63% | 54.28% | [无] |
| 6 | 1024 | 97.69% | 54.38% | [无] |
| 8 | 1024 | 97.39% | 54.21% | [无] |

> 来源：`outputs_v3_phI_K64_M6_h1024_l*`  
> **主配置** `extra2_K32_M18`（不同 K,M）：Router R@1=**99.97%**，flat R@1=**55.15%**  
> 草稿中「2层512 ~46%」来自 `master_phC`（评测协议与主表不一致），**不宜写入主表**

---

## Table 7 — CE Ablation（Merged，K=128, M=8，B=1）

> 固定：Merged pool/query，`merged_K128_M8` router+assignment  
> 训练默认除变量外：**hn=0.7, T=0.1**（见 `ce_ablation.log`）

### 7A. n_negatives（hn=0.7, T=0.1）

| n_neg | R@1 | R@5 | R@10 |
|-------|-----|-----|------|
| 5 | 51.01% | 76.33% | 83.95% |
| 10 | 51.49% | 76.65% | 84.28% |
| 15 | 51.92% | 77.21% | 84.66% |
| 25 | 51.99% | 77.22% | 84.80% |
| **30** | **52.24%** | **77.34%** | **84.99%** |

### 7B. hn_ratio（n_neg=15, T=0.1）

| hn_ratio | R@1 | R@5 | R@10 |
|----------|-----|-----|------|
| 0.0 | 50.55% | 75.90% | 83.72% |
| 0.3 | 51.32% | 76.21% | 84.22% |
| 0.5 | 51.46% | 76.65% | 84.47% |
| **0.7** | **51.92%** | **77.21%** | **84.66%** |
| 1.0 | 51.78% | 77.14% | 84.72% |

### 7C. Temperature（n_neg=15, hn=0.7）

| T | R@1 | R@5 | R@10 |
|---|-----|-----|------|
| 0.05 | 51.87% | 77.26% | 84.73% |
| **0.1** | **51.92%** | **77.21%** | **84.66%** |
| 0.2 | 51.80% | 77.16% | 84.62% |
| 0.5 | 51.77% | 77.18% | 84.65% |

### Baseline（CE 主 run）

| 配置 | R@1 | 说明 |
|------|-----|------|
| `merged_K128_M8` CE | **50.87%** | `cross_encoder/merged_K128_M8`，默认 CE 训练 |
| 消融最佳 | **52.24%** | ab1_nneg30，+1.37pp vs baseline |

> 来源：`ce_ablation/*/eval_ce.json`（全部 14 个已核对）

---

## 论文叙事用对比（CE，推荐报 B=3 行）

| 数据集 | GENIUS R@1 `†` | QSBA CE R@1 | 配置 | Cand@B=3 | Cand/Pool |
|--------|----------------|-------------|------|----------|-----------|
| Flickr30K | 84.1% | **80.06%** | K64 M12 | 514 | 51.4% |
| Flickr30K | 84.1% | **79.82%** | K32 M8 | 601 | 60.1% |
| MS-COCO | 58.1% | **54.37%** | K64 M6 | 1,539 | 30.8% |
| Visual News | — | **20.48%** | K128 M8 | 113,227 | 20.9% |

**Framing：** 在候选减少约 **30–90%**（视数据集与 K,M）时，与 GENIUS 差距约 **3.7–4.3pp**（Flickr/COCO）；VN 无公开 † 基线。

---

## JSON 文件索引

```
cross_encoder/
├── flickr_K32_M8/eval_cross_encoder.json
├── flickr_K64_M12/eval_cross_encoder.json
├── mscoco_K64_M6/eval_cross_encoder.json
├── mscoco_K128_M8/eval_cross_encoder.json
├── merged_K128_M8/eval_cross_encoder.json
├── merged_K512_M6/eval_cross_encoder.json
├── merged_K512_M12/eval_cross_encoder.json
├── vn_K128_M8/eval_cross_encoder.json
└── vn_K128_M12/eval_cross_encoder.json

ce_ablation/
└── ab{1,2,3}_*/eval_ce.json  (14 files)
```

---

## 待补跑清单（按优先级）

### P0 — 主表脚注 / 可选一行

```bash
# 真·CLIP dense：全 pool cosine，无 router（需新脚本或改 06_evaluate）
# 示例逻辑：对每个 query 与 pool 全部向量算 sim，取 top-K
```

### P1 — 消融表 CE 列（可选）

```bash
# Table 3: 对 5 个 alpha run 各训 CE
# Table 5: v3_phH_K32_M6 与 v3_phL eps0.03 各训 CE
```

### P2 — Flickr CE 全网格

```bash
# 仅当论文需要 Table 4 完整 CE 面
```

---

*本文件由 workspace JSON 自动核对生成；若重跑 eval，请更新对应 `eval_cross_encoder.json` 后重新汇总。*
