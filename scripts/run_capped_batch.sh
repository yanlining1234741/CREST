#!/bin/bash
# 容量约束批量实验 [L=300, U=1500]
# 对每个数据集: 备份 -> 03_capped -> build_text(按类型) -> router -> cosine eval -> 恢复
# 全程独立命名, 不污染原始结果。test 层面跟 GENIUS 一致。
set -u  # 用未定义变量报错 (但不用 -e, 避免一个数据集挂了全停)

ROOT=/sda/yxyang/qsba_workspace/mbeir_aligned
cd /home/yxyang/claudenew/qsba
L=300; U=1500
LOG=$ROOT/capped_batch_$(date +%m%d_%H%M).log
echo "==== 容量约束批量 [L=$L,U=$U] 开始 $(date) ====" | tee $LOG

# 数据集: "名字:config:类型"  类型1=train⊂pool(用capped build), 类型2=train⊄pool(用topk)
DATASETS=(
  "nights_task4:nights_task4.yaml:1"
  "fashion200k_task0:fashion200k_task0.yaml:1"
  "mscoco_task3:mscoco_task3.yaml:1"
  "mscoco:mscoco_mbeir.yaml:2"
)

run_one() {
  local ds=$1 cfg=$2 typ=$3
  local DATA=$ROOT/data/$ds
  local OUT=$ROOT/outputs/$ds
  echo "" | tee -a $LOG
  echo "========== $ds (config=$cfg, 类型$typ) ==========" | tee -a $LOG

  # 0. 备份 text_with_buckets
  if [ ! -f $DATA/text_with_buckets_ORIG.pt ]; then
    cp $DATA/text_with_buckets.pt $DATA/text_with_buckets_ORIG.pt
    echo "[0] 备份 text_with_buckets_ORIG" | tee -a $LOG
  else
    echo "[0] 备份已存在, 跳过" | tee -a $LOG
  fi

  # 1. 容量约束分桶 (若已存在则跳过, 避免重复)
  local ASG=$OUT/assignment_capped_L${L}_U${U}.pt
  if [ -f $ASG ]; then
    echo "[1] 分桶已存在: $ASG" | tee -a $LOG
  else
    echo "[1] 03_capped 分桶..." | tee -a $LOG
    python scripts/03_run_sinkhorn_capped.py --config configs/$cfg \
      --input $OUT/query_aware_emb.pt --L $L --U $U --M 8 \
      --output $ASG >> $LOG 2>&1
    if [ $? -ne 0 ]; then echo "[1] !! 分桶失败 $ds, 跳过" | tee -a $LOG; return 1; fi
  fi

  # 2. build_text (按类型选)
  echo "[2] build_text (类型$typ)..." | tee -a $LOG
  if [ "$typ" == "1" ]; then
    python scripts/flickr_build_text_capped.py \
      --data $DATA --assignment $ASG >> $LOG 2>&1
  else
    python scripts/flickr_build_text.py \
      --data $DATA --assignment $ASG --centroids-from $ASG >> $LOG 2>&1
  fi
  if [ $? -ne 0 ]; then echo "[2] !! build_text 失败 $ds" | tee -a $LOG; \
    cp $DATA/text_with_buckets_ORIG.pt $DATA/text_with_buckets.pt; return 1; fi
  local KCAP=$(python -c "import torch; print(int(torch.load('$DATA/text_with_buckets.pt',weights_only=False)['K']))")
  echo "    build K=$KCAP" | tee -a $LOG

  # 3. router
  echo "[3] 训 router (K=$KCAP)..." | tee -a $LOG
  python scripts/04b_train_router_disjoint.py --config configs/$cfg \
    --text-buckets $DATA/text_with_buckets.pt \
    --output $OUT/router_capped_L${L}_U${U}.pt >> $LOG 2>&1
  if [ $? -ne 0 ]; then echo "[3] !! router 失败 $ds" | tee -a $LOG; \
    cp $DATA/text_with_buckets_ORIG.pt $DATA/text_with_buckets.pt; return 1; fi

  # 4. cosine eval
  echo "[4] cosine eval..." | tee -a $LOG
  python scripts/06b_evaluate_disjoint_capped.py --config configs/$cfg \
    --assignment $ASG --router $OUT/router_capped_L${L}_U${U}.pt \
    --text-buckets $DATA/text_with_buckets.pt \
    --image-emb $DATA/image_embeddings.pt --rerank-mode flat \
    --output $OUT/eval_capped_L${L}_U${U}_cosine.json >> $LOG 2>&1
  if [ $? -ne 0 ]; then echo "[4] !! eval 失败 $ds" | tee -a $LOG; \
    cp $DATA/text_with_buckets_ORIG.pt $DATA/text_with_buckets.pt; return 1; fi

  # 5. 恢复 text_with_buckets
  cp $DATA/text_with_buckets_ORIG.pt $DATA/text_with_buckets.pt
  local KREST=$(python -c "import torch; print(int(torch.load('$DATA/text_with_buckets.pt',weights_only=False)['K']))")
  echo "[5] 恢复 text_with_buckets (K=$KREST)" | tee -a $LOG
  echo "[OK] $ds 完成" | tee -a $LOG
}

for entry in "${DATASETS[@]}"; do
  IFS=':' read -r ds cfg typ <<< "$entry"
  run_one "$ds" "$cfg" "$typ"
done

# ===== 汇总对比表 =====
echo "" | tee -a $LOG
echo "================ 汇总: 原始 vs 容量约束 ================" | tee -a $LOG
python << PYEOF 2>&1 | tee -a $LOG
import json, os
ROOT="$ROOT"
L,U=$L,$U
rows=[("nights_task4","eval_K128_M8_cosine.json"),
      ("fashion200k_task0",None),
      ("mscoco_task3",None),
      ("mscoco",None)]
print(f"{'数据集':<18}{'原始R@5':>9}{'容量R@5':>9}{'ΔR@5':>7}{'原始cand':>10}{'容量cand':>10}{'降幅':>7}")
for ds,orig_name in rows:
    out=f"{ROOT}/outputs/{ds}"
    # 容量约束结果
    cap_f=f"{out}/eval_capped_L{L}_U{U}_cosine.json"
    if not os.path.exists(cap_f):
        print(f"{ds:<18}  (容量约束结果缺失, 可能这步失败)")
        continue
    cap=json.load(open(cap_f))
    cap_r5=cap['recall@5|B=5']*100
    cap_cand=cap['candidates@B=5']
    # 原始结果 (找 cosine json)
    orig_r5=orig_cand=None
    for fn in os.listdir(out):
        if 'cosine' in fn and 'capped' not in fn and fn.endswith('.json'):
            try:
                o=json.load(open(f"{out}/{fn}"))
                if 'recall@5|B=5' in o:
                    orig_r5=o['recall@5|B=5']*100; orig_cand=o['candidates@B=5']; break
            except: pass
    if orig_r5 is None:
        print(f"{ds:<18}{'?':>9}{cap_r5:>9.2f}{'?':>7}{'?':>10}{cap_cand:>10.0f}")
    else:
        d=cap_r5-orig_r5
        drop=100*(1-cap_cand/orig_cand)
        print(f"{ds:<18}{orig_r5:>9.2f}{cap_r5:>9.2f}{d:>+7.2f}{orig_cand:>10.0f}{cap_cand:>10.0f}{drop:>6.0f}%")
PYEOF
echo "==== 批量完成 $(date) ====" | tee -a $LOG
echo "完整 log: $LOG"
