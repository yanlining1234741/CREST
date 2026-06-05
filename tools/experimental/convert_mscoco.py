"""
MSCOCO task0 (qt -> ci) DISJOINT + MULTI-POS.
99.5% query 单 pos, 0.5% 多 pos, intersection eval 跟 GENIUS 协议一致.
"""
import json
import numpy as np
import torch
from pathlib import Path

GEN = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/gen_code/GENIUS_t5small/Large/Instruct/InBatch"
EXT = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/extracted_embed/CLIP_SF"
MBEIR = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/M-BEIR"
OUT = Path("/sda/yxyang/qsba_workspace/mbeir_aligned/data/mscoco")
OUT.mkdir(parents=True, exist_ok=True)

MSCOCO_HASH = 90000000


def main():
    print("="*60)
    print("MSCOCO task0 (qt -> ci) v3 multi-pos")
    print("="*60)

    # 1. Image pool 5K
    print("\n[1/5] Image pool 5K...")
    pool_emb = np.load(f"{GEN}/cand_pool/mbeir_mscoco_task0_test_cand_pool_embeddings.npy")
    pool_ids = np.load(f"{GEN}/cand_pool/mbeir_mscoco_task0_test_cand_pool_ids.npy")
    pool_did_to_row = {int(d): i for i, d in enumerate(pool_ids.tolist())}
    pool_feat = torch.from_numpy(pool_emb).float()
    torch.save({"features": pool_feat,
                "image_ids": torch.arange(len(pool_feat), dtype=torch.long)},
               OUT / "image_embeddings.pt")
    print(f"  {pool_feat.shape}")

    # 2. Test query + multi-pos
    print("\n[2/5] Test query + multi-pos targets...")
    test_emb = np.load(f"{GEN}/test/mbeir_mscoco_task0_test_embeddings.npy")
    test_targets_multi = []
    with open(f"{MBEIR}/query/test/mbeir_mscoco_task0_test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tgt_rows = []
            for p in d['pos_cand_list']:
                did = MSCOCO_HASH + int(p.split(':')[1])
                if did in pool_did_to_row:
                    tgt_rows.append(pool_did_to_row[did])
            assert len(tgt_rows) > 0
            test_targets_multi.append(tgt_rows)
    test_feat = torch.from_numpy(test_emb).float()
    avg_pos = sum(len(t) for t in test_targets_multi) / len(test_targets_multi)
    print(f"  query: {test_feat.shape}, avg pos: {avg_pos:.3f}")

    # 3. Train (跟之前一样, 复用 train_mscoco dict task3 行 img)
    print("\n[3/5] Train queries + targets...")
    td_q = torch.load(f"{EXT}/train_mscoco/query_SFpretrained_instruction_IT_dict.pt",
                      map_location='cpu', weights_only=False)
    idx_q = td_q['id_to_index']

    # 走之前的逻辑: task3 行的 img 是 train target image
    qid_to_imgpath_task3 = {}
    with open(f"{MBEIR}/query/train/mbeir_mscoco_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get('task_id') == 3:
                qid_to_imgpath_task3[d['qid']] = d['query_img_path']

    # qid_int → row, 但只对 task3 行有效
    task3_rows = ((td_q['img_mask']==1) & (td_q['text_mask']==1)).nonzero(as_tuple=True)[0]
    index_to_id = {v: k for k, v in idx_q.items()}
    imgpath_to_dictrow = {}
    for row in task3_rows.tolist():
        qid_int = index_to_id[row]
        N = qid_int - 4500000
        qid_str = f"9:{N}"
        path = qid_to_imgpath_task3.get(qid_str)
        if path:
            imgpath_to_dictrow[path] = row

    did_to_path = {}
    with open(f"{MBEIR}/cand_pool/local/mbeir_mscoco_train_cand_pool.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get('img_path'):
                did_to_path[d['did']] = d['img_path']

    # task0 task: text query + image target
    train_qid_to_targetpath = {}
    with open(f"{MBEIR}/query/train/mbeir_mscoco_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get('task_id') == 0:
                pos = d['pos_cand_list'][0]
                if pos in did_to_path:
                    train_qid_to_targetpath[d['qid']] = did_to_path[pos]

    unique_paths = sorted(set(train_qid_to_targetpath.values()))
    path_to_trainrow = {p: i for i, p in enumerate(unique_paths)}
    
    # 提取 train target image embedding
    train_img_list = []
    miss = 0
    for p in unique_paths:
        if p in imgpath_to_dictrow:
            row = imgpath_to_dictrow[p]
            emb = td_q['img'][row].float()
            train_img_list.append(emb / emb.norm().clamp_min(1e-12))
        else:
            miss += 1; train_img_list.append(torch.zeros(768))
    train_img = torch.stack(train_img_list)
    print(f"  train target img: {train_img.shape}, miss={miss}")
    torch.save({"features": train_img,
                "image_ids": torch.arange(len(train_img), dtype=torch.long)},
               OUT / "train_image_embeddings.pt")

    # task0 行 (text_mask=1, img_mask=0) = train query text
    task0_rows = ((td_q['text_mask']==1) & (td_q['img_mask']==0)).nonzero(as_tuple=True)[0]
    qid_to_task0_row = {}
    for row in task0_rows.tolist():
        qid_int = index_to_id[row]
        N = qid_int - 4500000
        qid_to_task0_row[f"9:{N}"] = row

    train_q_list = []
    train_target_trainrow_list = []
    for qid_str, tpath in train_qid_to_targetpath.items():
        if qid_str not in qid_to_task0_row: continue
        if tpath not in path_to_trainrow: continue
        row = qid_to_task0_row[qid_str]
        txt = td_q['text'][row].float()
        train_q_list.append(txt / txt.norm().clamp_min(1e-12))
        train_target_trainrow_list.append(path_to_trainrow[tpath])
    train_feat = torch.stack(train_q_list)
    train_target_trainrow = torch.tensor(train_target_trainrow_list, dtype=torch.long)
    print(f"  train query (text): {train_feat.shape}")

    # 4. Save text_raw.pt with multi-pos test target
    print("\n[4/5] Saving text_raw.pt (multi-pos)...")
    max_pos = max(len(t) for t in test_targets_multi)
    test_target_padded = torch.full((len(test_targets_multi), max_pos), -1, dtype=torch.long)
    for i, tgts in enumerate(test_targets_multi):
        test_target_padded[i, :len(tgts)] = torch.tensor(tgts)
    torch.save({
        "train_features": train_feat,
        "train_target_trainrow": train_target_trainrow,
        "test_features": test_feat,
        "test_target_multi": test_target_padded,
    }, OUT / "text_raw.pt")

    # 5. meta
    meta = {
        "dataset": "mscoco_task0",
        "task": "qt -> ci, multi-pos",
        "test_pool": len(pool_feat),
        "test_query": len(test_feat),
        "test_avg_pos": avg_pos,
        "train_query": len(train_feat),
        "train_target_unique": len(unique_paths),
        "max_pos": max_pos,
        "genius_R_R5": 68.0,
        "clip_sf_R5": 77.9,
    }
    json.dump(meta, open(OUT/"meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
