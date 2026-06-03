# Reproduction guide

## 1. Environment

```bash
cd QSBA-github
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib scipy   # for efficiency figures only
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

Requirements: Python 3.9+, CUDA GPU for training and timed benchmarks.

## 2. Data & configs

See [data/README.md](data/README.md). Then:

```bash
export QSBA_DATA_ROOT=/path/to/mbeir_aligned
python scripts/setup_mbeir_configs.py
```

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

Key scripts:

| Step | Script |
|------|--------|
| Sinkhorn (capped buckets) | `scripts/03_run_sinkhorn_capped.py` |
| Router (disjoint train) | `scripts/04b_train_router_disjoint.py` |
| CE train | `scripts/07b_train_ce_disjoint.py` |
| CE eval (paper) | `scripts/06g_evaluate_ce_filter_capped.py` |

## 4. Published numbers

Pre-computed metrics (no GPU needed):

- Main table: [docs/QSBA_RESULTS_CE.md](docs/QSBA_RESULTS_CE.md)
- JSON: [results/paper/](results/paper/)

## 5. Efficiency benchmarks (VN QPS vs size)

Uses embeddings under `$QSBA_DATA_ROOT/data/visualnews_task3/`.

```bash
export CUDA_VISIBLE_DEVICES=0
export QSBA_DATA_ROOT=/path/to/mbeir_aligned

# QSBA router+CE sweep (3 trials per point, ~30 min)
python benchmarks/13_qps_vs_size.py

# Plot QSBA / GENIUS / GRACE figure
python benchmarks/19_qps_vs_size_all_methods.py
```

Outputs: `results/efficiency/qps_vs_size_three_methods.png`, `latency_table_vn_three.txt`.

GENIUS / GRACE latencies are reference baselines loaded from JSON (see `results/efficiency/genius_latency_raw.json`, `grace_latency_raw.json`).

## 6. Unit tests

```bash
pytest tests/ -v
```

## 7. Checkpoints

Model weights (`.pt`) are **not** in this repository. After training they appear under `$QSBA_DATA_ROOT/outputs/` and `cross_encoder/`. Use [Git LFS](https://git-lfs.github.com/) or release archives if you publish checkpoints.
