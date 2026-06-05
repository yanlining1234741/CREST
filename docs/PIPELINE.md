# Pipeline guide

CREST has **6 stages**. Each stage writes inspectable artifacts under `$CREST_DATA_ROOT/outputs/<dataset>/`.

```
                    ┌─────────────────────────────────────────────────────────┐
  Stage 0           │  Verify embeddings load (diagnostic stats)              │
  verify_data       └─────────────────────────────────────────────────────────┘
                                        │
                    ┌─────────────────────────────────────────────────────────┐
  Stage 1           │  Query-aware representation  (α=1.0, M-BEIR protocol)   │
  query_aware       │  → query_aware_emb_alpha1.0.pt                            │
                    └─────────────────────────────────────────────────────────┘
                                        │
                    ┌─────────────────────────────────────────────────────────┐
  Stage 2           │  Sinkhorn equipartitioned bucket assignment (K, M)        │
  assign_buckets    │  → assignment_K{K}_M{M}.pt                                │
                    │  → data/text_with_buckets.pt (train/test bucket labels)   │
                    └─────────────────────────────────────────────────────────┘
                                        │
                    ┌─────────────────────────────────────────────────────────┐
  Stage 3           │  Train query→bucket router (disjoint train queries)       │
  train_router      │  → router_K{K}.pt                                         │
                    └─────────────────────────────────────────────────────────┘
                                        │
                    ┌─────────────────────────────────────────────────────────┐
  Stage 4           │  Train cross-encoder reranker on router candidates        │
  train_cross_encoder │ → cross_encoder/{dataset}_K{K}_M{M}/cross_encoder.pt  │
                    └─────────────────────────────────────────────────────────┘
                                        │
                    ┌─────────────────────────────────────────────────────────┐
  Stage 5           │  End-to-end eval: router top-B → CE rerank → Recall@K     │
  evaluate          │  → cross_encoder/.../eval_cross_encoder.json            │
                    └─────────────────────────────────────────────────────────┘
```

## Paper hyperparameters

| Dataset | K | M | Reference R@1 (CE, B=1) |
|---------|---|---|-------------------------|
| Flickr30K | 64 | 8 | 77.98% |
| MS-COCO | 128 | 8 | 53.65% |
| Visual News | 512 | 6 | 19.59% |

## One-command reproduction

```bash
export CREST_DATA_ROOT=/path/to/mbeir_aligned
export CUDA_VISIBLE_DEVICES=0

# Full pipeline (all 6 stages)
python run.py train --dataset flickr
python run.py train --dataset mscoco
python run.py train --dataset visualnews_task3
```

Resume from interruption:

```bash
python run.py train --dataset mscoco --skip-if-exists
```

Run a single stage:

```bash
python run.py train --dataset flickr --stage 3   # router only
```

## Manual stage commands

Equivalent to `python run.py train`:

```bash
python stages/pipeline.py --dataset flickr --stage all
python stages/pipeline.py --dataset mscoco --stage 2 --skip-if-exists
```

## Stage scripts (low-level)

| Stage | Script | Core module |
|-------|--------|-------------|
| 0 | `scripts/01_prepare_data.py` | `crest/data.py` |
| 1 | `scripts/02_build_query_aware.py` | `crest/query_aware.py` |
| 2 | `scripts/03_run_sinkhorn.py` + `scripts/flickr_build_text.py` | `crest/sinkhorn.py` |
| 3 | `scripts/04b_train_router_disjoint.py` | `crest/router.py` |
| 4 | `scripts/07b_train_ce_disjoint.py` | `crest/cross_encoder.py` |
| 5 | `scripts/06b_evaluate_disjoint.py` | `crest/evaluate.py` |

## Efficiency benchmark (VN QPS vs corpus size)

```bash
python run.py benchmark --task qps
# or
python benchmarks/13_qps_vs_size.py
python benchmarks/19_qps_vs_size_all_methods.py
```

Outputs: `results/efficiency/qps_vs_size_three_methods.png`

## Experimental scripts

Advanced ablations and capped-bucket variants are in `tools/experimental/` (not required for main paper Table 1).
