#!/usr/bin/env python3
"""Print Table 3 rows from outputs_table3_alpha*_K32_M6 eval JSONs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    ("1.0 (image only)", "1_0"),
    ("0.75", "0_75"),
    ("0.50", "0_50"),
    ("0.25", "0_25"),
    ("0.0 (caption only)", "0_0"),
]

for label, tag in ORDER:
    p = ROOT / f"outputs_table3_alpha{tag}_K32_M6/eval_correct.json"
    if not p.exists():
        print(f"{label:20s} & -- & -- & -- \\\\")
        continue
    e = json.load(open(p))
    rr = e["router_recall@1"]
    cand = int(round(e["candidates@B=1"]))
    r1 = e["recall@1|B=1"]
    print(f"{label:20s} & {rr:.3f} & {cand:,} & {r1:.3f} \\\\".replace(",", "{,}"))
