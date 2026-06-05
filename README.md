# CREST

**CREST: Collapse-Free Routing with Equipartitioned Semantic Buckets for Cross-Modal Retrieval**

Official implementation of **CREST** — scalable cross-modal retrieval via query-aware semantic buckets, Sinkhorn equipartitioned assignment, learned routing, and cross-encoder reranking.

Paper results use **M-BEIR / GENIUS-aligned** CLIP-SF embeddings (768-d) on Flickr30K, MS-COCO, and Visual News.

## Repository layout

```
CREST-github/
├── src/                 # Core: query_aware, sinkhorn, router, cross_encoder
├── scripts/             # Training & evaluation pipeline
├── configs/             # Dataset YAML (run setup_mbeir_configs.py first)
├── tests/
├── benchmarks/          # QPS vs corpus size (VN), efficiency plots
├── docs/                # Paper tables & protocol notes
├── results/
│   ├── paper/           # Published eval JSON
│   └── efficiency/      # Latency / QPS artifacts
├── data/README.md       # How to obtain embeddings (not shipped)
├── REPRODUCE.md         # Step-by-step reproduction
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/CREST.git
cd CREST
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Point to M-BEIR data root, then generate configs
export CREST_DATA_ROOT=/path/to/mbeir_aligned
python scripts/setup_mbeir_configs.py

# Run Flickr pipeline
bash scripts/run_mbeir_ce_pipeline.sh flickr
```

See **[REPRODUCE.md](REPRODUCE.md)** for full commands and **[docs/CREST_RESULTS_CE.md](docs/CREST_RESULTS_CE.md)** for reported metrics.

## Citation

```bibtex
@inproceedings{crest2025,
  title     = {CREST: Collapse-Free Routing with Equipartitioned Semantic Buckets for Cross-Modal Retrieval},
  author    = {...},
  booktitle = {CVPR},
  year      = {2025}
}
```

## License

See [LICENSE](LICENSE).
