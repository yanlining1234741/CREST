# Data layout (not included in git)

CREST expects **pre-extracted CLIP-SF (768-d) embeddings** aligned with the **M-BEIR / GENIUS** evaluation protocol.

## Directory structure

Place data under `CREST_DATA_ROOT` (default: sibling folder `../mbeir_aligned`):

```
mbeir_aligned/
├── data/
│   ├── flickr/
│   ├── mscoco/
│   └── visualnews_task3/
├── outputs/
└── cross_encoder/
```

## Generate configs after data is ready

```bash
export CREST_DATA_ROOT=/path/to/mbeir_aligned
python scripts/setup_mbeir_configs.py
```
