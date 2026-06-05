# Stages

| Stage | Name | Input | Output |
|-------|------|-------|--------|
| **0** | `verify_data` | embeddings in `data/` | `outputs/*/data_stats` |
| **1** | `query_aware` | image + text embeddings | `query_aware_emb_alpha1.0.pt` |
| **2** | `assign_buckets` | query-aware emb | `assignment_K*_M*.pt`, `text_with_buckets.pt` |
| **3** | `train_router` | text_with_buckets | `router_K*.pt` |
| **4** | `train_cross_encoder` | router candidates | `cross_encoder/*/cross_encoder.pt` |
| **5** | `evaluate` | router + CE | `eval_cross_encoder.json` |

**Entry point:** `python stages/pipeline.py --dataset <name> --stage all`

Or use the top-level CLI: `python run.py train --dataset flickr`

See [docs/PIPELINE.md](../docs/PIPELINE.md) for the full guide.
