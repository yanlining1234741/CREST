"""
VisualNews task3 (qi -> ct, image query -> text caption).
单池, 跟 VN task0 镜像。
Pool: 537K text candidates; Test query: 20K image queries.
GENIUS^R = 28.4 (Table 1)
"""
import json, sys
import numpy as np
import torch
from pathlib import Path

GEN = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/gen_code/GENIUS_t5small/Large/Instruct/InBatch"
EXT = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/extracted_embed/CLIP_SF"
MBEIR = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/M-BEIR"
OUT = Path("/sda/yxyang/qsba_workspace/mbeir_aligned/data/visualnews_task3")
OUT.mkdir(parents=True, exist_ok=True)

def pos_to_int(p):
    """'0:N' (VN text did) -> int. VN dataset_id=0, text did 数值就是 N (无加偏移)."""
    return int(p.split(':')[1])


def main():
    print("="*60)
    print("VisualNews task3 conversion (qi -> ct)")
    print("="*60)

    # ---------- 1. Text pool (537K) ----------
    print("\n[1/4] Text pool (cand) 537K...")
    pool_emb = np.load(f"{GEN}/cand_pool/mbeir_visualnews_task3_cand_pool_embeddings.npy")
    pool_ids = np.load(f"{GEN}/cand_pool/mbeir_visualnews_task3_cand_pool_ids.npy")
    pool_did_to_row = {int(d): i for i, d in enumerate(pool_ids.tolist())}
    print(f"  pool: {pool_emb.shape}, dids 范围: {pool_ids.min()} ~ {pool_ids.max()}")
    
    # 这里"image_embeddings.pt"语义上是"pool of candidates" (虽然 task3 pool 是 text)
    # 为了复用 single-pool pipeline (脚本里都叫 image_emb), 保持文件名
    pool_feat = torch.from_numpy(pool_emb).float()
    torch.save({
        "features": pool_feat,
        "image_ids": torch.arange(len(pool_feat), dtype=torch.long),
    }, OUT / "image_embeddings.pt")
    print(f"  saved image_embeddings.pt (实为 text pool)")

    # ---------- 2. Test query (20K, image) ----------
    print("\n[2/4] Test query (image) 20K...")
    test_emb = np.load(f"{GEN}/test/mbeir_visualnews_task3_test_embeddings.npy")
    test_tgt = []
    with open(f"{MBEIR}/query/test/mbeir_visualnews_task3_test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            # 取第一个 pos
            tgt_did = pos_to_int(d['pos_cand_list'][0])
            if tgt_did not in pool_did_to_row:
                raise RuntimeError(f"qid {d['qid']} target did {tgt_did} not in pool!")
            test_tgt.append(pool_did_to_row[tgt_did])
    assert len(test_tgt) == len(test_emb)
    test_feat = torch.from_numpy(test_emb).float()
    print(f"  {test_feat.shape}, targets unique: {len(set(test_tgt))}")

    # ---------- 3. Train query (image) + train targets ----------
    # 跟 VN task0 不同: 这里 query 是 image, target 是 text
    # train dict 单 row 共用 img/text, 用 qid_int 索引
    print("\n[3/4] Train query (image) from VN train dict...")
    td = torch.load(f"{EXT}/train_visualnews/query_SFpretrained_instruction_IT_dict.pt",
                    map_location='cpu', weights_only=False)
    idx = td['id_to_index']

    train_feat_list = []
    train_tgt_list = []
    miss = 0
    with open(f"{MBEIR}/query/train/mbeir_visualnews_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get('task_id') != 3: continue
            qid_int = int(d['qid'].split(':')[1])
            if qid_int not in idx:
                miss += 1; continue
            row = idx[qid_int]
            # query 是 image
            img = td['img'][row].float()
            img = img / img.norm().clamp_min(1e-12)
            # target text did
            tgt_did = pos_to_int(d['pos_cand_list'][0])
            if tgt_did not in pool_did_to_row:
                miss += 1; continue
            train_feat_list.append(img)
            train_tgt_list.append(pool_did_to_row[tgt_did])
    train_feat = torch.stack(train_feat_list)
    train_tgt = torch.tensor(train_tgt_list, dtype=torch.long)
    print(f"  train_feat: {train_feat.shape}, miss={miss}")

    # ---------- 4. Save text_embeddings.pt (合并 train+test, 走单池 pipeline) ----------
    print("\n[4/4] Saving text_embeddings.pt (train+test merged)...")
    # 单池 pipeline 期待: text_features = [train; test], image_ids = target row
    # 复用 VN task0 的 text_embeddings.pt 结构
    all_feat = torch.cat([train_feat, test_feat], dim=0)
    all_tgt = torch.cat([train_tgt, torch.tensor(test_tgt, dtype=torch.long)], dim=0)
    torch.save({
        "features": all_feat,
        "image_ids": all_tgt,        # 每个 query 的 target candidate row
        "train_n": len(train_feat),  # 用于切分 train/test
    }, OUT / "text_embeddings.pt")
    print(f"  {all_feat.shape}, train_n={len(train_feat)}")

    # meta
    meta = {
        "dataset": "visualnews_task3",
        "task": "qi -> ct (image query -> text caption)",
        "protocol": "GENIUS M-BEIR task3, strict",
        "pool_size": len(pool_feat),
        "test_query": len(test_feat),
        "train_query": len(train_feat),
        "genius_R_R5": 28.4,
        "clip_sf_R5": 42.8,
    }
    json.dump(meta, open(OUT/"meta.json", "w"), indent=2)
    print(f"\n{json.dumps(meta, indent=2)}")
    print("DONE")


if __name__ == "__main__":
    main()
