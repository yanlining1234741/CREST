# Reproduction guide — CREST

**CREST: Collapse-Free Routing with Equipartitioned Semantic Buckets for Cross-Modal Retrieval**

## 1. Environment

```bash
cd CREST-github
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib scipy   # for efficiency figures only
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

Requirements: Python 3.9+, CUDA GPU for training and timed benchmarks.

## 2. Data & configs

See [data/README.md](data/README.md). Then:

```bash
export CREST_DATA_ROOT=/path/to/mbeir_aligned
python scripts/setup_mbeir_configs.py
```

(`QSBA_DATA_ROOT` is still accepted as a legacy alias.)

## 3. Main retrieval pipeline (per dataset)

**Stages 1–3** (query-aware → Sinkhorn → router):

```bash
bash scripts/run_pipeline.sh configs/flickr_mbeir.yaml
```

**Full paper pipeline** (router + cross-encoder + capped eval):

```bash
bash scripts/run_mbeir_ce_pipeline.sh flickr
bash scripts/run_mbeir_ce_pipeline.sh mscoco
bash scripts/run_mbeir_ce_pipeline.sh visualnews_task3
```

## 4. Published numbers

- Main table: [docs/CREST_RESULTS_CE.md](docs/CREST_RESULTS_CE.md)
- JSON: [results/paper/](results/paper/)

## 5. Efficiency benchmarks (VN QPS vs size)

```bash
export CUDA_VISIBLE_DEVICES=0
export CREST_DATA_ROOT=/path/to/mbeir_aligned

python benchmarks/13_qps_vs_size.py
python benchmarks/19_qps_vs_size_all_methods.py
```

Outputs: `results/efficiency/qps_vs_size_three_methods.png`, `latency_table_vn_three.txt`.

## 6. Unit tests

```bash
pytest tests/ -v
```

## 7. Checkpoints

Model weights (`.pt`) are **not** in this repository. After training they appear under `$CREST_DATA_ROOT/outputs/` and `cross_encoder/`.
