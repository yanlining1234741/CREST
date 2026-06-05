"""
Convert GENIUS VisualNews embeddings -> QSBA format (strict GENIUS M-BEIR task0 protocol).
text_embeddings.pt = [train 99903 ; test 19995], image_ids = target row index.
"""
import json
import numpy as np
import torch
from pathlib import Path

GENIUS = "/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25"
GEN_CODE = f"{GENIUS}/gen_code/GENIUS_t5small/Large/Instruct/InBatch"
EXTRACTED = f"{GENIUS}/extracted_embed/CLIP_SF"
MBEIR = f"{GENIUS}/M-BEIR"
OUT = Path("/sda/yxyang/qsba_workspace/mbeir_aligned/data/visualnews")
OUT.mkdir(parents=True, exist_ok=True)

def pos_to_n(p): return int(p.split(":")[1])

def main():
    print("VN conversion (GENIUS M-BEIR task0)")

    print("[1/5] Image pool 542K...")
    img_emb = np.load(f"{GEN_CODE}/cand_pool/mbeir_visualnews_task0_cand_pool_embeddings.npy")
    img_ids = np.load(f"{GEN_CODE}/cand_pool/mbeir_visualnews_task0_cand_pool_ids.npy")
    did_to_idx = {int(d): i for i, d in enumerate(img_ids.tolist())}
    img_feat = torch.from_numpy(img_emb).float()
    print(f"  {img_feat.shape}, norm[:3]={img_feat[:3].norm(dim=-1).tolist()}")
    torch.save({"features": img_feat,
                "image_ids": torch.arange(len(img_feat), dtype=torch.long)},
               OUT / "image_embeddings.pt")

    print("[2/5] Train query task0...")
    td = torch.load(f"{EXTRACTED}/train_visualnews/query_SFpretrained_instruction_IT_dict.pt",
                    map_location="cpu", weights_only=False)
    task0_rows = ((td["text_mask"]==1)&(td["img_mask"]==0)).nonzero(as_tuple=True)[0]
    index_to_id = {v: k for k, v in td["id_to_index"].items()}
    qid_to_target = {}
    with open(f"{MBEIR}/query/train/mbeir_visualnews_train.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("task_id")==0:
                qid_to_target[d["qid"]] = pos_to_n(d["pos_cand_list"][0])
    train_feats, train_tgt, miss = [], [], 0
    for row in task0_rows.tolist():
        qid_int = index_to_id[row]
        qid_str = f"0:{qid_int}"
        if qid_str not in qid_to_target: miss+=1; continue
        tn = qid_to_target[qid_str]
        if tn not in did_to_idx: miss+=1; continue
        train_feats.append(td["text"][row]); train_tgt.append(did_to_idx[tn])
    train_feat = torch.stack(train_feats).float()
    print(f"  train: {train_feat.shape}, missing={miss}")

    print("[3/5] Test query task0...")
    test_emb = np.load(f"{GEN_CODE}/test/mbeir_visualnews_task0_test_embeddings.npy")
    test_tgt = []
    with open(f"{MBEIR}/query/test/mbeir_visualnews_task0_test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            test_tgt.append(did_to_idx[pos_to_n(d["pos_cand_list"][0])])
    assert len(test_tgt)==len(test_emb), f"{len(test_tgt)} vs {len(test_emb)}"
    test_feat = torch.from_numpy(test_emb).float()
    print(f"  test: {test_feat.shape}")

    print("[4/5] Merge...")
    n_train, n_test = len(train_feat), len(test_feat)
    all_text = torch.cat([train_feat, test_feat], dim=0)
    all_tgt = torch.tensor(train_tgt + test_tgt, dtype=torch.long)
    torch.save({"features": all_text, "image_ids": all_tgt,
                "caption_ids": torch.arange(len(all_text), dtype=torch.long)},
               OUT / "text_embeddings.pt")
    print(f"  text: {all_text.shape} train[0:{n_train}] test[{n_train}:{n_train+n_test}]")

    print("[5/5] Meta...")
    meta = {"dataset":"visualnews","protocol":"GENIUS M-BEIR task0",
            "n_images":len(img_feat),"n_train_query":n_train,"n_test_query":n_test,
            "query_start_for_eval":n_train,"n_train_queries_for_router":n_train,
            "pool_start":0,"pool_end":len(img_feat)}
    with open(OUT/"meta.json","w") as f: json.dump(meta,f,indent=2)
    print(json.dumps(meta,indent=2)); print("DONE")

if __name__=="__main__": main()
