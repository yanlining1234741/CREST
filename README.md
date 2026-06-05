# CREST

**CREST: Collapse-Free Routing with Equipartitioned Semantic Buckets for Cross-Modal Retrieval**

Official PyTorch implementation for scalable text-to-image retrieval via query-aware semantic buckets, Sinkhorn equipartitioned assignment, learned routing, and cross-encoder reranking.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CREST_DATA_ROOT=/path/to/mbeir_aligned

python run.py setup
python run.py verify --dataset flickr
python run.py train --dataset flickr          # full 6-stage pipeline
python tools/check_results.py --dataset flickr  # compare R@1 vs paper
```

## Repository structure

```
CREST/
├── run.py                 # Top-level CLI (setup / train / verify / benchmark)
├── crest/                 # Core library
│   ├── query_aware.py     # Stage 1
│   ├── sinkhorn.py        # Stage 2 — equipartitioned assignment
│   ├── router.py          # Stage 3 — collapse-free routing
│   ├── cross_encoder.py   # Stage 4 — reranking
│   ├── evaluate.py        # Stage 5 — metrics
│   ├── datasets.py        # Paper hyperparameters (K, M per dataset)
│   └── paths.py           # CREST_DATA_ROOT resolution
├── stages/
│   ├── pipeline.py        # 6-stage orchestrator
│   └── README.md
├── scripts/               # Stage implementation scripts (called by pipeline)
├── configs/datasets/      # Generated YAML configs
├── tools/
│   ├── setup_configs.py
│   ├── check_results.py
│   └── experimental/      # Ablations & legacy variants
├── benchmarks/            # VN QPS vs corpus size
├── docs/                  # Full documentation
│   ├── INSTALL.md
│   ├── DATA.md            # How to obtain embeddings
│   ├── PIPELINE.md        # Stage-by-stage guide
│   └── RESULTS.md         # Expected metrics
├── tests/
└── results/               # Published JSON & figures
```

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/INSTALL.md](docs/INSTALL.md) | Environment setup |
| [docs/DATA.md](docs/DATA.md) | Download / prepare embeddings |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Train & evaluate (6 stages) |
| [docs/RESULTS.md](docs/RESULTS.md) | Expected numbers & verification |
| [docs/CREST_RESULTS_CE.md](docs/CREST_RESULTS_CE.md) | Full paper tables |

## Paper results (reference)

| Dataset | K | M | R@1 (CE, B=1) |
|---------|---|---|---------------|
| Flickr30K | 64 | 8 | 77.98% |
| MS-COCO | 128 | 8 | 53.65% |
| Visual News | 512 | 6 | 19.59% |

## Citation

```bibtex
@inproceedings{crest2026,
  title     = {CREST: Collapse-Free Routing with Equipartitioned Semantic Buckets for Cross-Modal Retrieval},
  author    = {...},
  booktitle = {CVPR},
  year      = {2025}
}
```

## License

MIT — see [LICENSE](LICENSE).
