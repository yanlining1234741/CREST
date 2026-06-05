"""
MSCOCO task3 (qi -> ct) DISJOINT + MULTI-POS.
每条 test query 5 个 pos caption, eval 用 intersection.
"""
import json
import numpy as np
import torch
from pathlib import Path

GEN = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/gen_code/GENIUS_t5small/Large/Instruct/InBatch"
EXT = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/extracted_embed/CLIP_SF"
MBEIR = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/M-BEIR"
OUT = Path("/sda/yxyang/qsba_workspace/mbeir_aligned/data/mscoco_task3")
OUT.mkdir(parents=True, exist_ok=True)

MSCOCO_HASH = 90000000


def main():
    print("="*60)
    print("MSCOCO task3 conversion (qi -> ct) v3, multi-pos")
    print("="*60)

    # 1. Test pool
    print("\n[1/6] Test text pool 25K...")
    pool_emb = np.load(f"{GEN}/cand_pool/mbeir_mscoco_task3_test_cand_pool_embeddings.npy")
    pool_ids = np.load(f"{GEN}/cand_pool/mbeir_mscoco_task3_test_cand_pool_ids.npy")
    pool_did_to_row = {int(d): i for i, d in enumerate(pool_ids.tolist())}
    pool_feat = torch.from_numpy(pool_emb).float()
    torch.save({"features": pool_feat,
                "image_ids": torch.arange(len(pool_feat), dtype=torch.long)},
               OUT / "image_embeddings.pt")
    print(f"  pool: {pool_feat.shape}")

    # 2. Test query + MULTI-POS targets
    print("\n[2/6] Test query (5000 image) + multi-pos targets...")
    test_emb = np.load(f"{GEN}/test/mbeir_mscoco_task3_test_embeddings.npy")
    test_targets_multi = []  # list of list-of-row-indices
    with open(f"{MBEIR}/query/test/mbeir_mscoco_task3_test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tgt_rows = []
            for p in d['pos_cand_list']:
                tgt_did = MSCOCO_HASH + int(p.split(':')[1])
                if tgt_did in pool_did_to_row:
                    tgt_rows.append(pool_did_to_row[tgt_did])
            assert len(tgt_rows) > 0, f"query {d['qid']} no valid pos!"
            test_targets_multi.append(tgt_rows)
    test_feat = torch.from_numpy(test_emb).float()
    avg_pos = sum(len(t) for t in test_targets_multi) / len(test_targets_multi)
    print(f"  test query: {test_feat.shape}, avg pos per query: {avg_pos:.2f}")

    # 3. Train query (image) — train dict
    print("\n[3/6] Loading train dicts...")
    td_q = torch.load(f"{EXT}/train_mscoco/query_SFpretrained_instruction_IT_dict.pt",
                      map_location='cpu', weights_only=False)
    idx_q = td_q['id_to_index']

    print("\n[4/6] Loading universal train pool...")
    td_p = torch.load(f"{EXT}/train/pool_SFpretrained_IT_dict.pt",
                      map_location='cpu', weights_only=False, mmap=True)
    idx_p = td_p['id_to_index']

    # 5. Iterate task3 train
    print("\n[5/6] Iterating task3 train...")
    train_records = []
    target_dids_set = set()
    miss_q = miss_t = 0
    with open(f"{MBEIR}/query/train/mbeir_mscoco_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get('task_id') != 3: continue
            qid_int = int(d['qid'].split(':')[1]) + 4500000
            if qid_int not in idx_q:
                miss_q += 1; continue
            tgt_did = MSCOCO_HASH + int(d['pos_cand_list'][0].split(':')[1])
            if tgt_did not in idx_p:
                miss_t += 1; continue
            train_records.append((qid_int, tgt_did))
            target_dids_set.add(tgt_did)
    print(f"  records: {len(train_records)}, unique targets: {len(target_dids_set)}, miss_q={miss_q}, miss_t={miss_t}")

    target_did_to_row = {did: i for i, did in enumerate(sorted(target_dids_set))}
    train_target_text_list = []
    for did in sorted(target_dids_set):
        emb = td_p['text'][idx_p[did]].float()
        emb = emb / emb.norm().clamp_min(1e-12)
        train_target_text_list.append(emb)
    train_target_text = torch.stack(train_target_text_list)
    print(f"  train target text emb: {train_target_text.shape}")
    torch.save({"features": train_target_text,
                "image_ids": torch.arange(len(train_target_text), dtype=torch.long)},
               OUT / "train_image_embeddings.pt")

    train_q_list = []
    train_tgt_row_list = []
    for qid_int, tgt_did in train_records:
        img = td_q['img'][idx_q[qid_int]].float()
        img = img / img.norm().clamp_min(1e-12)
        train_q_list.append(img)
        train_tgt_row_list.append(target_did_to_row[tgt_did])
    train_feat = torch.stack(train_q_list)
    train_target_trainrow = torch.tensor(train_tgt_row_list, dtype=torch.long)

    # 6. Save with multi-pos targets
    print("\n[6/6] Saving text_raw.pt (with multi-pos targets)...")
    # Pad multi-pos targets to fixed length for tensor storage
    max_pos = max(len(t) for t in test_targets_multi)
    test_target_padded = torch.full((len(test_targets_multi), max_pos), -1, dtype=torch.long)
    for i, tgts in enumerate(test_targets_multi):
        test_target_padded[i, :len(tgts)] = torch.tensor(tgts)
    
    torch.save({
        "train_features": train_feat,
        "train_target_trainrow": train_target_trainrow,
        "test_features": test_feat,
        "test_target_multi": test_target_padded,   # (5000, max_pos), -1=padding
    }, OUT / "text_raw.pt")

    meta = {
        "dataset": "mscoco_task3",
        "task": "qi -> ct, multi-pos",
        "test_pool": len(pool_feat),
        "test_query": len(test_feat),
        "test_avg_pos": avg_pos,
        "train_query": len(train_feat),
        "train_target_unique": len(target_dids_set),
        "max_pos": max_pos,
        "genius_R_R5": 91.1,
        "clip_sf_R5": 92.3,
    }
    json.dump(meta, open(OUT/"meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
