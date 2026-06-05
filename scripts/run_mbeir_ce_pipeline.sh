#!/bin/bash
# Full CREST + cross-encoder pipeline on M-BEIR aligned data (paper main setting).
#
# Usage:
#   export CREST_DATA_ROOT=/path/to/mbeir_aligned
#   bash scripts/run_mbeir_ce_pipeline.sh flickr|mscoco|visualnews_task3
#
set -euo pipefail

TASK="${1:?Usage: $0 flickr|mscoco|visualnews_task3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$TASK" in
  flickr) CONFIG="configs/flickr_mbeir.yaml" ;;
  mscoco) CONFIG="configs/mscoco_mbeir.yaml" ;;
  visualnews_task3) CONFIG="configs/vn_task3.yaml" ;;
  *) echo "Unknown task: $TASK"; exit 1 ;;
esac

if [[ ! -f "$CONFIG" ]]; then
  echo "Config missing. Run: python scripts/setup_mbeir_configs.py"
  exit 1
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "=== CREST M-BEIR pipeline: $TASK ==="
bash scripts/run_pipeline.sh "$CONFIG"

OUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
ASSIGN=$(ls -t "$OUT_DIR"/assignment_K*.pt 2>/dev/null | head -1)
ROUTER="$OUT_DIR/router_K"*.pt
ROUTER=$(ls -t $ROUTER 2>/dev/null | head -1)

echo "=== Train cross-encoder (disjoint) ==="
python scripts/04b_train_router_disjoint.py --config "$CONFIG" 2>/dev/null || true
python scripts/07b_train_ce_disjoint.py --config "$CONFIG" \
  --assignment "$ASSIGN" --router "$ROUTER" || \
  python scripts/07_train_cross_encoder.py --config "$CONFIG" \
  --assignment "$ASSIGN" --router "$ROUTER"

echo "=== Evaluate with CE rerank (capped, paper protocol) ==="
python scripts/06g_evaluate_ce_filter_capped.py --config "$CONFIG" \
  --assignment "$ASSIGN" --router "$ROUTER" || \
  python scripts/06b_evaluate_disjoint_capped.py --config "$CONFIG" \
  --assignment "$ASSIGN" --router "$ROUTER"

echo "Done. Check $OUT_DIR for eval_*.json"
