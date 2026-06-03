# Data layout (not included in git)

QSBA expects **pre-extracted CLIP-SF (768-d) embeddings** aligned with the **M-BEIR / GENIUS** evaluation protocol.

## Directory structure

Place data under `QSBA_DATA_ROOT` (default: sibling folder `../mbeir_aligned`):

```
mbeir_aligned/
├── data/
│   ├── flickr/
│   │   └── image_embeddings.pt      # pool + queries (Flickr slice)
│   ├── mscoco/
│   │   ├── image_embeddings.pt
│   │   └── text_embeddings.pt
│   └── visualnews_task3/
│       ├── image_embeddings.pt      # image queries
│       └── text_embeddings.pt       # text corpus pool
├── outputs/                         # sinkhorn + router (created by pipeline)
└── cross_encoder/                   # CE checkpoints (created by training)
```

## Embedding format

Each `.pt` file:

```python
{
    "features": FloatTensor [N, 768],   # L2-normalized
    "image_ids": List[int],             # optional metadata
}
```

## How to obtain embeddings

1. Follow [GENIUS](https://github.com/...) / M-BEIR instructions to extract CLIP-SF features on:
   - Flickr30K (pool 1K, queries 5K — Karpathy split)
   - MS-COCO (pool 5K, queries 24,809)
   - Visual News task3 (pool ~537K, queries 19,898)
2. Or contact the authors for pre-computed `mbeir_aligned` archives.

## Generate configs after data is ready

```bash
export QSBA_DATA_ROOT=/path/to/mbeir_aligned
python scripts/setup_mbeir_configs.py
```
