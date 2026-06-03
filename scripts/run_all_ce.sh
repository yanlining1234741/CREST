#!/bin/bash
# 一键跑所有 7 个数据集的 CE
set -e
cd /home/yxyang/claudenew/qsba

ROOT=/sda/yxyang/qsba_workspace/mbeir_aligned
CE_ROOT=$ROOT/cross_encoder

run_disjoint_ce() {
    local ds=$1; local K=$2; local M=$3; local KM="K${K}_M${M}"
    local DATA=$ROOT/data/$ds
    local OUT=$ROOT/outputs/$ds
    local CONFIG=configs/${ds}.yaml
    [ ! -f $CONFIG ] && CONFIG=configs/${ds}_mbeir.yaml
    [ ! -f $CONFIG ] && CONFIG=configs/${ds}_task0.yaml
    [ ! -f $CONFIG ] && CONFIG=configs/${ds}_task3.yaml
    [ ! -f $CONFIG ] && CONFIG=configs/${ds}_task4.yaml
    local CE_OUT=$CE_ROOT/${ds}_${KM}
    mkdir -p $CE_OUT

    echo "==========================================="
    echo "CE on $ds (K=$K, M=$M, config=$CONFIG)"
    echo "==========================================="

    # 1. Train CE
    python scripts/07b_train_ce_disjoint.py \
        --config $CONFIG \
        --assignment $OUT/assignment_${KM}.pt \
        --text-buckets $DATA/text_with_buckets.pt \
        --image-emb $DATA/image_embeddings.pt \
        --train-image-emb $DATA/train_image_embeddings.pt \
        --output $CE_OUT/cross_encoder.pt \
        --n-train-queries 80000 \
        --epochs 12 \
        2>&1 | tee $CE_OUT/train.log

    # 2. Eval CE
    python scripts/06b_evaluate_disjoint.py \
        --config $CONFIG \
        --assignment $OUT/assignment_${KM}.pt \
        --router $OUT/router_K${K}.pt \
        --text-buckets $DATA/text_with_buckets.pt \
        --image-emb $DATA/image_embeddings.pt \
        --output $CE_OUT/eval_cross_encoder.json \
        --rerank-mode cross_encoder \
        --cross-encoder $CE_OUT/cross_encoder.pt

    echo "Done: $ds CE"
}

# 跑顺序: 池子小的先 (出错快定位)
run_disjoint_ce "flickr" 64 8
run_disjoint_ce "mscoco" 128 8
run_disjoint_ce "mscoco_task3" 128 8
run_disjoint_ce "nights_task4" 128 8
run_disjoint_ce "fashion200k_task0" 256 8

echo ""
echo "All CE training done!"
