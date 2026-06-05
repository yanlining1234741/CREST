"""Stage 1 alpha=1.0 FAST: image-only (copy + L2 normalize)."""
import argparse, torch
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-emb", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    blob = torch.load(args.image_emb, map_location="cpu", weights_only=False)
    feat = blob["features"].float()
    feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    out = {"features": feat, "image_ids": blob["image_ids"], "alpha": 1.0,
           "n_captions_per_image": [0]*len(feat),
           "diagnostic": {"alpha": 1.0, "note": "image-only fast"}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"Stage1 a=1.0: {feat.shape} -> {args.output}")

if __name__=="__main__": main()
