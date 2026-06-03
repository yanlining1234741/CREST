"""
NIGHTS task4 (qi -> ci, image-to-image). Multi-pos.
dataset_id=4, pool 40K, test 2K, train ~16K.
GENIUS^R R@5 = 30.2, CLIP-SF = 32.0
"""
import json, numpy as np, torch
from pathlib import Path

GEN = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/gen_code/GENIUS_t5small/Large/Instruct/InBatch"
EXT = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/extracted_embed/CLIP_SF"
MBEIR = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/M-BEIR"
OUT = Path("/sda/yxyang/qsba_workspace/mbeir_aligned/data/nights_task4")
OUT.mkdir(parents=True, exist_ok=True)

QID_OFFSET = 500000
DID_OFFSET = 10000000
DS_ID = 4


def main():
    print("=" * 60)
    print("NIGHTS task4 (qi -> ci)")
    print("=" * 60)

    # 1. Image pool
    print("\n[1/5] Image pool...")
    pool_emb = np.load(f"{GEN}/cand_pool/mbeir_nights_task4_cand_pool_embeddings.npy")
    pool_ids = np.load(f"{GEN}/cand_pool/mbeir_nights_task4_cand_pool_ids.npy")
    pool_did_to_row = {int(d): i for i, d in enumerate(pool_ids.tolist())}
    pool_feat = torch.from_numpy(pool_emb).float()
    torch.save({"features": pool_feat,
                "image_ids": torch.arange(len(pool_feat), dtype=torch.long)},
               OUT / "image_embeddings.pt")
    print(f"  {pool_feat.shape}")

    # 2. Test query + multi-pos
    print("\n[2/5] Test query...")
    test_emb = np.load(f"{GEN}/test/mbeir_nights_task4_test_embeddings.npy")
    test_targets_multi = []
    with open(f"{MBEIR}/query/test/mbeir_nights_task4_test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            tgt_rows = []
            for p in d['pos_cand_list']:
                did = DS_ID * DID_OFFSET + int(p.split(':')[1])
                if did in pool_did_to_row:
                    tgt_rows.append(pool_did_to_row[did])
            assert len(tgt_rows) > 0
            test_targets_multi.append(tgt_rows)
    test_feat = torch.from_numpy(test_emb).float()
    avg_pos = sum(len(t) for t in test_targets_multi) / len(test_targets_multi)
    print(f"  test: {test_feat.shape}, avg pos: {avg_pos:.2f}")

    # 3. Train
    print("\n[3/5] Loading universal dicts...")
    td_q = torch.load(f"{EXT}/train/query_SFpretrained_instruction_IT_dict.pt",
                      map_location='cpu', weights_only=False, mmap=True)
    idx_q = td_q['id_to_index']
    td_p = torch.load(f"{EXT}/train/pool_SFpretrained_IT_dict.pt",
                      map_location='cpu', weights_only=False, mmap=True)
    idx_p = td_p['id_to_index']

    print("\n[4/5] Iterating NIGHTS task4 train...")
    train_records = []
    target_dids = set()
    miss_q = miss_t = 0
    with open(f"{MBEIR}/query/train/mbeir_nights_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get('task_id') != 4: continue
            qid_int = DS_ID * QID_OFFSET + int(d['qid'].split(':')[1])
            if qid_int not in idx_q:
                miss_q += 1; continue
            for p in d['pos_cand_list']:
                tgt_did = DS_ID * DID_OFFSET + int(p.split(':')[1])
                if tgt_did not in idx_p:
                    miss_t += 1
                    continue
                train_records.append((qid_int, tgt_did))
                target_dids.add(tgt_did)
                break
    print(f"  records: {len(train_records)}, miss_q={miss_q}, miss_t={miss_t}")

    target_did_to_row = {did: i for i, did in enumerate(sorted(target_dids))}
    # NIGHTS query 是 image
    train_img_list = []
    for did in sorted(target_dids):
        emb = td_p['img'][idx_p[did]].float()
        train_img_list.append(emb / emb.norm().clamp_min(1e-12))
    train_img = torch.stack(train_img_list)
    torch.save({"features": train_img,
                "image_ids": torch.arange(len(train_img), dtype=torch.long)},
               OUT / "train_image_embeddings.pt")
    print(f"  train target img: {train_img.shape}")

    train_q_list = []
    train_target_trainrow_list = []
    for qid_int, tgt_did in train_records:
        # query 是 image, 用 td_q['img']
        emb = td_q['img'][idx_q[qid_int]].float()
        train_q_list.append(emb / emb.norm().clamp_min(1e-12))
        train_target_trainrow_list.append(target_did_to_row[tgt_did])
    train_feat = torch.stack(train_q_list)
    train_target_trainrow = torch.tensor(train_target_trainrow_list, dtype=torch.long)
    print(f"  train query (image): {train_feat.shape}")

    # 5. Save
    print("\n[5/5] Save...")
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

    meta = {
        "dataset": "nights_task4",
        "task": "qi -> ci, multi-pos",
        "test_pool": len(pool_feat),
        "test_query": len(test_feat),
        "test_avg_pos": avg_pos,
        "train_query": len(train_feat),
        "train_target_unique": len(target_dids),
        "genius_R_R5": 30.2,
        "clip_sf_R5": 32.0,
    }
    json.dump(meta, open(OUT/"meta.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
