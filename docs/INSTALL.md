# Installation

## Requirements

- Python 3.9+
- CUDA GPU (recommended for training and timed benchmarks)
- ~20 GB disk for code + results; ~15 GB additional for embeddings (see [DATA.md](DATA.md))

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/CREST.git
cd CREST

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: efficiency figure scripts
pip install matplotlib scipy

export PYTHONPATH="$(pwd):$PYTHONPATH"
```

## Configure data path

```bash
export CREST_DATA_ROOT=/path/to/mbeir_aligned
python run.py setup --data-root $CREST_DATA_ROOT
```

This writes `configs/datasets/{flickr,mscoco,visualnews_task3}.yaml`.

## Verify installation

```bash
python run.py verify --dataset flickr
```

Expected: `01_prepare_data.log` under `$CREST_DATA_ROOT/outputs/flickr/` with embedding statistics.
