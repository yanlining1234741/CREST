#!/bin/bash
# 一键跑完整 pipeline
#
# 用法:
#   bash scripts/run_pipeline.sh [config_path]
#
# 默认 config: configs/coco_siglip2.yaml

set -e

CONFIG="${1:-configs/coco_siglip2.yaml}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "========================================="
echo "CREST Pipeline"
echo "Config: $CONFIG"
echo "Working dir: $ROOT"
echo "========================================="

# 解析 output dir
OUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
mkdir -p "$OUT_DIR"

# Stage 0: 数据验证
echo
echo "[Stage 0] 验证数据..."
python scripts/01_prepare_data.py --config "$CONFIG"

# Stage 1: Query-aware
echo
echo "[Stage 1] 构造 query-aware 表征..."
ALPHA=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['query_aware']['alpha'])")
QA_PATH="$OUT_DIR/query_aware_emb_alpha${ALPHA}.pt"
python scripts/02_build_query_aware.py --config "$CONFIG" --alpha "$ALPHA" --output "$QA_PATH"

# Stage 2: Sinkhorn
echo
echo "[Stage 2] Sinkhorn 桶分配..."
K=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sinkhorn']['K'])")
M=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['sinkhorn']['M'])")
ASSIGN_PATH="$OUT_DIR/assignment_K${K}_M${M}.pt"
python scripts/03_run_sinkhorn.py --config "$CONFIG" --input "$QA_PATH" \
    --K "$K" --M "$M" --output "$ASSIGN_PATH"

# Stage 3: Router
echo
echo "[Stage 3] 训练 router..."
ROUTER_PATH="$OUT_DIR/router_K${K}.pt"
python scripts/04_train_router.py --config "$CONFIG" \
    --assignment "$ASSIGN_PATH" --output "$ROUTER_PATH"

# Stage 4: Eval
echo
echo "[Stage 4] End-to-end 评估..."
EVAL_PATH="$OUT_DIR/eval_K${K}.json"
python scripts/05_evaluate.py --config "$CONFIG" \
    --assignment "$ASSIGN_PATH" --router "$ROUTER_PATH" --output "$EVAL_PATH"

echo
echo "========================================="
echo "✓ Pipeline complete!"
echo "Eval results: $EVAL_PATH"
echo "========================================="
cat "$EVAL_PATH"
