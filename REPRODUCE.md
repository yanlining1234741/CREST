# Reproduction

See the structured guides in **[docs/](docs/)**:

1. [INSTALL.md](docs/INSTALL.md) — environment
2. [DATA.md](docs/DATA.md) — obtain embeddings
3. [PIPELINE.md](docs/PIPELINE.md) — run stages 0–5
4. [RESULTS.md](docs/RESULTS.md) — verify metrics

**One-liner after data is ready:**

```bash
export CREST_DATA_ROOT=/path/to/mbeir_aligned
python run.py setup
python run.py train --dataset flickr
python run.py train --dataset mscoco
python run.py train --dataset visualnews_task3
```
