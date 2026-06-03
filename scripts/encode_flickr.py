import sys, os, json
import numpy as np, torch
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, '/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/src')
from models.uniir_clip.clip_scorefusion.clip_sf import CLIPScoreFusion

KARPATHY='/sda/yxyang/data/dataset_flickr30k.json'
IMG_DIR='/sda/yxyang/data/Flickr30K/flickr30k_images/flickr30k_images'
CKPT='/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/checkpoint/CLIP_SF/clip_sf_large.pth'
CLIP_DL='/sdd/web3bench-500g/lnyan/projects/GENIUS-CVPR25/checkpoint/CLIP'
OUT='/sda/yxyang/qsba_workspace/mbeir_aligned/data/flickr'
os.makedirs(OUT, exist_ok=True)
DEVICE='cuda'; BATCH=256

def load_model():
    print("Loading CLIP-SF ViT-L/14...")
    m=CLIPScoreFusion(model_name="ViT-L/14", device=DEVICE, download_root=CLIP_DL)
    m.float()
    ck=torch.load(CKPT, map_location='cpu', weights_only=False)
    m.load_state_dict(ck['model']); m.eval().to(DEVICE)
    return m, m.get_img_preprocess_fn(), m.get_tokenizer()

@torch.no_grad()
def enc_img(m, pp, paths):
    fs=[]
    for i in tqdm(range(0,len(paths),BATCH),desc="img"):
        b=paths[i:i+BATCH]
        im=torch.stack([pp(Image.open(p).convert("RGB")) for p in b]).to(DEVICE)
        e=torch.nn.functional.normalize(m.encode_image(im).float(),dim=-1)
        fs.append(e.cpu())
    return torch.cat(fs,0)

@torch.no_grad()
def enc_txt(m, tok, texts):
    fs=[]
    for i in tqdm(range(0,len(texts),BATCH),desc="txt"):
        b=texts[i:i+BATCH]
        t=tok(b).to(DEVICE)
        e=torch.nn.functional.normalize(m.encode_text(t).float(),dim=-1)
        fs.append(e.cpu())
    return torch.cat(fs,0)

def main():
    d=json.load(open(KARPATHY))['images']
    test=[i for i in d if i['split']=='test']
    train=[i for i in d if i['split'] in ('train','restval')]
    print(f"test {len(test)} train {len(train)}")
    m,pp,tok=load_model()

    # test pool image
    tip=[os.path.join(IMG_DIR,i['filename']) for i in test]
    tif=enc_img(m,pp,tip)
    # test caption
    tc,tct=[],[]
    for r,i in enumerate(test):
        for s in i['sentences'][:5]: tc.append(s['raw']); tct.append(r)
    tcf=enc_txt(m,tok,tc)
    # train image
    trp=[os.path.join(IMG_DIR,i['filename']) for i in train]
    trif=enc_img(m,pp,trp)
    # train caption
    trc,trct=[],[]
    for r,i in enumerate(train):
        for s in i['sentences'][:5]: trc.append(s['raw']); trct.append(r)
    trcf=enc_txt(m,tok,trc)

    torch.save({"features":tif,"image_ids":torch.arange(len(tif))}, f"{OUT}/image_embeddings.pt")
    torch.save({"features":trif,"image_ids":torch.arange(len(trif))}, f"{OUT}/train_image_embeddings.pt")
    torch.save({"test_features":tcf,"test_target":torch.tensor(tct),
                "train_features":trcf,"train_target_trainrow":torch.tensor(trct)},
               f"{OUT}/text_raw.pt")
    json.dump({"n_test_img":len(tif),"n_test_cap":len(tcf),
               "n_train_img":len(trif),"n_train_cap":len(trcf),
               "protocol":"GENIUS Table3 Karpathy, no instruction"},
              open(f"{OUT}/meta.json","w"),indent=2)
    print("DONE",OUT)
    print(f"test_img {tif.shape} test_cap {tcf.shape} train_img {trif.shape} train_cap {trcf.shape}")

if __name__=="__main__": main()
