#!/bin/bash
# Alpha ablation for Table 3: MS-COCO 5K pool, K=32, M=6
set -e
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
BASE_CFG="outputs_extra2_K32_M18/config.yaml"

for ALPHA in 1.0 0.75 0.5 0.25 0.0; do
  TAG=$(echo "$ALPHA" | tr '.' '_')
  OUT="outputs_table3_alpha${TAG}_K32_M6"
  mkdir -p "$OUT"
  echo "======== alpha=$ALPHA -> $OUT ========"

  # config
  "$PY" -c "
import yaml
cfg = yaml.safe_load(open('$BASE_CFG'))
cfg['output_dir'] = '$OUT'
cfg['data']['device'] = 'cpu'
cfg['query_aware']['alpha'] = $ALPHA
cfg['query_aware'].setdefault('l2_normalize', True)
cfg['sinkhorn']['K'] = 32
cfg['sinkhorn']['M'] = 6
cfg['sinkhorn'].setdefault('epsilon', 0.01)
cfg['sinkhorn'].setdefault('n_sinkhorn_iters', 20)
cfg['sinkhorn'].setdefault('n_em_iters', 30)
cfg['sinkhorn'].setdefault('init', 'kmeans++')
cfg['router']['epochs'] = 40
cfg['eval'] = {'top_B': [1, 3, 5], 'top_K': [1, 5, 10], 'rerank_with_dense': True}
yaml.dump(cfg, open('$OUT/config.yaml', 'w'), default_flow_style=False)
"

  "$PY" scripts/02_build_query_aware.py --config "$OUT/config.yaml" \
    --output "$OUT/query_aware_emb.pt"

  "$PY" scripts/03_run_sinkhorn.py --config "$OUT/config.yaml" \
    --input "$OUT/query_aware_emb.pt" --K 32 --M 6 \
    --output "$OUT/assignment_K32_M6.pt"

  "$PY" scripts/04_train_router.py --config "$OUT/config.yaml" \
    --assignment "$OUT/assignment_K32_M6.pt" \
    --output "$OUT/router_K32.pt" --epochs 40

  "$PY" scripts/06_evaluate_correct.py \
    --config "$OUT/config.yaml" \
    --assignment "$OUT/assignment_K32_M6.pt" \
    --router "$OUT/router_K32.pt" \
    --pool-start 0 --pool-end 5000 \
    --query-start 100000 \
    --output "$OUT/eval_correct.json"

  echo "Done alpha=$ALPHA"
done

echo "===== Table 3 summary ====="
"$PY" -c "
import json
from pathlib import Path
for a in ['1_0','0_75','0_5','0_25','0_0']:
    p = Path(f'outputs_table3_alpha{a}_K32_M6/eval_correct.json')
    if not p.exists():
        print(a, 'MISSING'); continue
    e = json.load(open(p))
    print(f\"alpha={a.replace('_','.')}: rR={e['router_recall@1']:.3f} R1={e['recall@1|B=1']:.3f} cand={e['candidates@B=1']:.0f}\")
"
