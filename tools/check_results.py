#!/usr/bin/env python3
"""Compare eval JSON against published reference values."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from crest.datasets import get_dataset
from crest.paths import cross_encoder_dir


def _extract_r1(data: dict, top_b: int = 1):
    for block in data.get("results", data.get("eval", [data])):
        if isinstance(block, dict):
            if block.get("top_B") == top_b or block.get("B") == top_b:
                for key in ("recall_at_1", "R@1", "r1", "recall@1"):
                    if key in block:
                        v = block[key]
                        return float(v) * 100 if v <= 1 else float(v)
            if "recall" in block and isinstance(block["recall"], dict):
                return float(block["recall"].get("1", block["recall"].get(1, 0))) * 100
    for key in ("recall_at_1", "R@1"):
        if key in data:
            v = data[key]
            return float(v) * 100 if v <= 1 else float(v)
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--top-b", type=int, default=1)
    args = p.parse_args()

    spec = get_dataset(args.dataset)
    eval_path = cross_encoder_dir(args.dataset, spec.K, spec.M) / "eval_cross_encoder.json"
    ref_names = {
        "flickr": "flickr_K64_M8",
        "mscoco": "mscoco_K128_M8",
        "visualnews_task3": "vn_task3_K512_M6",
    }
    ref_path = REPO / "results" / "paper" / ref_names[args.dataset] / "eval_cross_encoder.json"

    if not eval_path.exists():
        print(f"Missing: {eval_path}\nRun: python run.py train --dataset {args.dataset}")
        sys.exit(1)

    data = json.loads(eval_path.read_text())
    r1 = _extract_r1(data, args.top_b)
    print(f"Dataset: {args.dataset}  K={spec.K} M={spec.M}  B={args.top_b}")
    print(f"Your R@1: {r1:.2f}%" if r1 is not None else "Could not parse R@1 from JSON")

    if spec.paper_r1 is not None:
        print(f"Paper R@1: {spec.paper_r1:.2f}%")
        if r1 is not None:
            diff = abs(r1 - spec.paper_r1)
            status = "OK" if diff < 0.5 else "CHECK"
            print(f"Delta: {diff:.2f} pp  [{status}]")

    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        ref_r1 = _extract_r1(ref, args.top_b)
        if ref_r1 is not None:
            print(f"Bundled ref R@1: {ref_r1:.2f}%")


if __name__ == "__main__":
    main()
