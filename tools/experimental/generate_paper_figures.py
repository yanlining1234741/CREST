#!/usr/bin/env python3
"""Generate figures for qsba_aaai.tex"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT.parent / "figures"
FIG.mkdir(exist_ok=True)


def fig2_recall_vs_candidates():
    """Recall@1 vs mean candidates for Flickr and COCO."""
    datasets = [
        ("Flickr30K", "outputs_flickr2_K16_M12/eval_correct.json", 1000),
        ("MS-COCO 5K", "outputs_extra2_K32_M18/eval_correct.json", 5000),
    ]
    baselines = {
        "CLIP dense": None,
        "AVG": (None, 79.2),
        "GENIUS": (None, 84.1),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, (title, ev_path, pool) in zip(axes, datasets):
        ev = json.load(open(ROOT / ev_path))
        bs = sorted({int(k.split("|B=")[1]) for k in ev if k.startswith("recall@1|B=")})
        cands, r1s = [], []
        for b in bs:
            cands.append(ev.get(f"candidates@B={b}", pool))
            r1s.append(ev[f"recall@1|B={b}"] * 100)
        ax.plot(cands, r1s, "o-", color="#2563eb", linewidth=2, markersize=6, label="CREST")
        ax.axhline(83.4 if "Flickr" in title else 58.4, color="gray", ls="--", lw=1, label="CLIP R@1")
        if "Flickr" in title:
            ax.scatter([1000], [84.1], marker="*", color="#dc2626", s=80, zorder=5, label="GENIUS")
            ax.scatter([1000], [79.2], marker="s", color="#f59e0b", s=50, zorder=5, label="AVG")
        else:
            ax.scatter([5000], [58.1], marker="*", color="#dc2626", s=80, zorder=5, label="GENIUS")
        ax.set_xlabel("Mean candidates per query")
        ax.set_ylabel("Recall@1 (%)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG / "recall_vs_candidates.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def fig3_bucket_balance():
    """Sinkhorn vs high-epsilon bucket size distributions."""
    paths = {
        "Sinkhorn ($\\varepsilon{=}0.01$)": ROOT / "outputs_v3_phH_K32_M6/assignment_K32_M6.pt",
        "High $\\varepsilon$ (0.03)": ROOT / "outputs_v3_phL_K64_M6_eps0.03_em30sk20/assignment_K64_M6.pt",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, (title, path) in zip(axes, paths.items()):
        if not path.exists():
            ax.text(0.5, 0.5, "missing", ha="center")
            continue
        a = torch.load(path, map_location="cpu", weights_only=False)
        hard = a["hard_assignment"]
        K = int(a["K"])
        counts = torch.zeros(K)
        for row in hard:
            for b in row.tolist():
                counts[b] += 1
        counts = counts.numpy()
        ax.bar(range(len(counts)), np.sort(counts), color="#2563eb", width=1.0)
        ratio = counts.std() / (counts.mean() + 1e-8)
        ax.set_title(f"{title}\nstd/mean={ratio:.2f}")
        ax.set_xlabel("Bucket rank (sorted by size)")
        ax.set_ylabel("Items per bucket")
    plt.tight_layout()
    out = FIG / "bucket_balance.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def write_fig1_tikz():
    """Regenerate figures/fig_pipeline.tex (TikZ only; caption in qsba_aaai.tex)."""
    src = (ROOT.parent / "figures" / "fig_pipeline.tex").read_text()
    out = ROOT.parent / "figures" / "fig_pipeline.tex"
    print(f"fig_pipeline.tex present ({len(src)} bytes)")


if __name__ == "__main__":
    fig2_recall_vs_candidates()
    fig3_bucket_balance()
    write_fig1_tikz()
