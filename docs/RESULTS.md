# Expected results

Compare your `eval_cross_encoder.json` against published numbers after running stage 5.

## Main table (cross-encoder rerank, B=1)

| Dataset | R@1 | R@5 | R@10 | Avg candidates |
|---------|-----|-----|------|----------------|
| Flickr30K (K=64,M=8) | **77.98** | 92.90 | 94.82 | ~308 |
| MS-COCO (K=128,M=8) | **53.65** | — | — | ~633 |
| Visual News (K=512,M=6) | **19.59** | — | — | ~47,917 |

Full tables: [CREST_RESULTS_CE.md](CREST_RESULTS_CE.md)

## Verify your run

```bash
# After stage 5
python tools/check_results.py --dataset flickr
```

Or manually:

```bash
cat $CREST_DATA_ROOT/cross_encoder/flickr_K64_M8/eval_cross_encoder.json | python -m json.tool | head -30
```

Look for `recall_at_1` under `rerank_mode: cross_encoder` and `top_B: 1`.

## Pre-computed reference JSON (no GPU)

Shipped in this repo:

```
results/paper/
├── flickr_K64_M8/eval_cross_encoder.json
├── mscoco_K128_M8/eval_cross_encoder.json
└── vn_task3_K512_M6/eval_cross_encoder.json
```

## Efficiency (VN, V100 batch=1)

| Method | 537K corpus QPS |
|--------|-----------------|
| CREST router+CE | ~291 (3-trial avg) |
| GENIUS (VN) | ~6.2 |
| GRACE (COCO ref.) | ~0.22 |

See `results/efficiency/latency_table_vn_three.txt` and `qps_vs_size_three_methods.png`.
