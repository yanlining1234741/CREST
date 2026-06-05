# Data preparation

CREST uses **pre-extracted CLIP-SF embeddings** (768-d, L2-normalized) aligned with the **M-BEIR / GENIUS** evaluation protocol.

## Option A: Download our release archive (recommended)

Contact the authors or download from the project page:

| Archive | Contents | Size |
|---------|----------|------|
| `CREST-data-critical-*.tar.gz` | Flickr / COCO / VN embeddings + GRACE baseline | ~13 GB |

After download:

```bash
tar -xzf CREST-data-critical-*.tar.gz
export CREST_DATA_ROOT=/your/path/mbeir_aligned
mkdir -p $CREST_DATA_ROOT/data
cp -r CREST-data-critical-*/mbeir_embeddings/* $CREST_DATA_ROOT/data/
```

## Option B: Extract embeddings yourself

Follow the [GENIUS](https://github.com/) / M-BEIR protocol to extract CLIP-SF features. Required layout:

```
$CREST_DATA_ROOT/
├── data/
│   ├── flickr/
│   │   ├── image_embeddings.pt      # pool (1K) + metadata
│   │   ├── train_image_embeddings.pt
│   │   └── text_raw.pt              # train/test query captions + targets
│   ├── mscoco/
│   │   ├── image_embeddings.pt      # 5K pool
│   │   ├── train_image_embeddings.pt
│   │   └── text_raw.pt
│   └── visualnews_task3/
│       ├── image_embeddings.pt      # image queries
│       └── text_embeddings.pt       # text corpus pool (~537K)
├── outputs/                         # created by pipeline
└── cross_encoder/                   # created by stage 4
```

### Embedding file format

```python
{
    "features": FloatTensor [N, 768],   # L2-normalized
    "image_ids": List[int],             # optional
    # text_raw.pt additionally has:
    "train_features", "test_features",
    "train_target_trainrow", "test_target_row" or "test_target_multi",
}
```

## Evaluation protocol (M-BEIR slices)

| Dataset | Pool | Queries | Query index slice |
|---------|------|---------|-------------------|
| Flickr30K | 1,000 | 5,000 | `[153914, 158914)` |
| MS-COCO | 5,000 | 24,809 | `[100000, 124809)` |
| Visual News | 537,568 | 19,898 | `[100000, 119898)` |

## After data is in place

```bash
python run.py setup --data-root $CREST_DATA_ROOT
python run.py verify --dataset flickr
```
