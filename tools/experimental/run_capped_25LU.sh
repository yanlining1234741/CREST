#!/bin/bash
# 容量约束 25 阈值大规模实验 (5数据集 × 25阈值 = 125 + flickr特殊[50,200])
# 全自动, 已存在跳过, 不污染原始, test层面跟GENIUS一致
set -u
ROOT=/sda/yxyang/qsba_workspace/mbeir_aligned
cd /home/yxyang/claudenew/qsba
LOG=$ROOT/capped_25LU_$(date +%m%d_%H%M).log
echo "==== 25阈值容量约束大规模实验 开始 $(date) ====" | tee $LOG

# 25 个阈值
LUS="50-200 100-300 100-500 150-400 200-600 200-800 250-750 300-900 300-1200 300-1500 400-1000 400-1600 500-1500 500-2000 600-1800 700-2000 800-2400 1000-2500 1000-3000 1200-3600 1500-4000 1500-5000 2000-5000 2000-6000 2500-7000"

# 数据集: "名:config:类型"  (类型1=train⊂pool, 类型2=train⊄pool用topk)
DSLIST=(
  "nights_task4:nights_task4.yaml:1"
  "fashion200k_task0:fashion200k_task0.yaml:1"
  "mscoco_task3:mscoco_task3.yaml:1"
  "mscoco:mscoco_mbeir.yaml:2"
  "flickr:flickr_mbeir.yaml:2"
)

run_one() {
  local ds=$1 cfg=$2 typ=$3 L=$4 U=$5
  local DATA=$ROOT/data/$ds OUT=$ROOT/outputs/$ds
  local ASG=$OUT/assignment_capped_L${L}_U${U}.pt
  local EVAL=$OUT/eval_capped_L${L}_U${U}_cosine.json
  if [ -f $EVAL ]; then echo "  [$ds L=$L U=$U] 已存在,跳过" | tee -a $LOG; return 0; fi
  echo "  [$ds L=$L U=$U 类型$typ] 开始 $(date +%H:%M:%S)" | tee -a $LOG
  [ -f $DATA/text_with_buckets_ORIG.pt ] || cp $DATA/text_with_buckets.pt $DATA/text_with_buckets_ORIG.pt

  if [ ! -f $ASG ]; then
    python scripts/03_run_sinkhorn_capped.py --config configs/$cfg \
      --input $OUT/query_aware_emb.pt --L $L --U $U --M 8 --output $ASG >> $LOG 2>&1
    if [ $? -ne 0 ]; then echo "    !! 分桶失败" | tee -a $LOG; \
      cp $DATA/text_with_buckets_ORIG.pt $DATA/text_with_buckets.pt; return 1; fi
  fi
  if [ "$typ" == "1" ]; then
    python scripts/flickr_build_text_capped.py --data $DATA --assignment $ASG >> $LOG 2>&1
  else
    python scripts/flickr_build_text.py --data $DATA --assignment $ASG --centroids-from $ASG >> $LOG 2>&1
  fi
  if [ $? -ne 0 ]; then echo "    !! build_text 失败" | tee -a $LOG; \
    cp $DATA/text_with_buckets_ORIG.pt $DATA/text_with_buckets.pt; return 1; fi
  python scripts/04b_train_router_disjoint.py --config configs/$cfg \
    --text-buckets $DATA/text_with_buckets.pt --output $OUT/router_capped_L${L}_U${U}.pt >> $LOG 2>&1
  if [ $? -ne 0 ]; then echo "    !! router 失败" | tee -a $LOG; \
    cp $DATA/text_with_buckets_ORIG.pt $DATA/text_with_buckets.pt; return 1; fi
  python scripts/06b_evaluate_disjoint_capped.py --config configs/$cfg \
    --assignment $ASG --router $OUT/router_capped_L${L}_U${U}.pt \
    --text-buckets $DATA/text_with_buckets.pt --image-emb $DATA/image_embeddings.pt \
    --rerank-mode flat --output $EVAL >> $LOG 2>&1
  if [ $? -ne 0 ]; then echo "    !! eval 失败" | tee -a $LOG; \
    cp $DATA/text_with_buckets_ORIG.pt $DATA/text_with_buckets.pt; return 1; fi
  cp $DATA/text_with_buckets_ORIG.pt $DATA/text_with_buckets.pt
  local r5=$(python -c "import json;print(round(json.load(open('$EVAL'))['recall@5|B=5']*100,2))")
  local cand=$(python -c "import json;print(round(json.load(open('$EVAL'))['candidates@B=5'],0))")
  local rr=$(python -c "import json;print(round(json.load(open('$EVAL'))['router_recall@5']*100,2))")
  echo "    [OK] R@5=$r5 router_R@5=$rr cand=$cand" | tee -a $LOG
}

# 主循环: 数据集 × 阈值
for entry in "${DSLIST[@]}"; do
  IFS=':' read -r ds cfg typ <<< "$entry"
  echo "" | tee -a $LOG
  echo "########## $ds ##########" | tee -a $LOG
  for lu in $LUS; do
    IFS='-' read -r L U <<< "$lu"
    run_one "$ds" "$cfg" "$typ" "$L" "$U"
  done
done

# flickr 特殊 [50,200] 已在上面25个里 (50-200), 单独标记
echo "" | tee -a $LOG
echo "=== flickr [50,200] 特殊实验已包含在主循环 ===" | tee -a $LOG

# ===== 汇总表 =====
echo "" | tee -a $LOG
echo "================ 25阈值汇总 (B=5 cosine) ================" | tee -a $LOG
python3 << PYEOF 2>&1 | tee -a $LOG
import json, os
ROOT="$ROOT"
LUS=[(50,200),(100,300),(100,500),(150,400),(200,600),(200,800),(250,750),(300,900),(300,1200),(300,1500),(400,1000),(400,1600),(500,1500),(500,2000),(600,1800),(700,2000),(800,2400),(1000,2500),(1000,3000),(1200,3600),(1500,4000),(1500,5000),(2000,5000),(2000,6000),(2500,7000)]
DS=["nights_task4","fashion200k_task0","mscoco_task3","mscoco","flickr"]
def orig(ds):
    out=f"{ROOT}/outputs/{ds}"
    if not os.path.isdir(out): return None,None
    for fn in os.listdir(out):
        if 'cosine' in fn and 'capped' not in fn and fn.endswith('.json'):
            try:
                o=json.load(open(f"{out}/{fn}"))
                if 'recall@5|B=5' in o: return o['recall@5|B=5']*100,o['candidates@B=5']
            except: pass
    return None,None
for ds in DS:
    o5,ocand=orig(ds)
    print(f"\n===== {ds} (原始 R@5={o5:.2f if o5 else 0}, cand={ocand:.0f if ocand else 0}) =====")
    print(f"{'[L,U]':>12}{'R@5':>8}{'Δ':>8}{'cand':>9}{'router_R':>9}")
    for L,U in LUS:
        f=f"{ROOT}/outputs/{ds}/eval_capped_L{L}_U{U}_cosine.json"
        if not os.path.exists(f):
            print(f"{f'[{L},{U}]':>12}{'--':>8}"); continue
        d=json.load(open(f))
        r5=d['recall@5|B=5']*100; cand=d['candidates@B=5']; rr=d['router_recall@5']*100
        delta=f"{r5-o5:+.2f}" if o5 else "?"
        print(f"{f'[{L},{U}]':>12}{r5:>8.2f}{delta:>8}{cand:>9.0f}{rr:>9.2f}")
PYEOF
echo "" | tee -a $LOG
echo "==== 全部完成 $(date) ====" | tee -a $LOG
