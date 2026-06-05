# Data

Embeddings are **not** included in this repository (~9–15 GB).

See **[docs/DATA.md](../docs/DATA.md)** for:

- Downloading `CREST-data-critical-*.tar.gz`
- Directory layout under `$CREST_DATA_ROOT`
- M-BEIR evaluation protocol slices

After data is ready:

```bash
export CREST_DATA_ROOT=/path/to/mbeir_aligned
python run.py setup
python run.py verify --dataset flickr
```
