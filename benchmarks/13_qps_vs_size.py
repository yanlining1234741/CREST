#!/usr/bin/env python3
"""Measure QSBA throughput (QPS) vs simulated corpus size |C| = B*N*M/K."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from paths import CE, DATA, MBEIR, QSBA, RESULTS

sys.path.insert(0, str(QSBA))

WARMUP = 50
N_MEASURE = 300
N_TRIALS = 3
CE_CHUNK = 4096
K, M, B = 512, 6, 1

CORPUS_SIZES_K = [1, 5, 10, 25, 50, 100, 150, 200, 250, 300, 537]
CORPUS_SIZES = [s * 1000 for s in CORPUS_SIZES_K]


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def load_embeddings():
    img_path = DATA["visualnews_task3"] / "image_embeddings.pt"
    txt_path = DATA["visualnews_task3"] / "text_embeddings.pt"
    for p in (img_path, txt_path):
        if not p.exists():
            raise FileNotFoundError(p)

    img = torch.load(img_path, map_location="cpu", weights_only=False)
    txt = torch.load(txt_path, map_location="cpu", weights_only=False)
    query_emb = l2norm(img["features"].float()).cpu()
    pool_emb = l2norm(txt["features"].float()).cpu()
    return query_emb, pool_emb


def load_router_model(device: torch.device):
    from src.router import RouterMLP, load_router

    ckpt = MBEIR / "outputs/visualnews_task3/router_K512.pt"
    if ckpt.exists():
        print(f"Router checkpoint: {ckpt}", flush=True)
        return load_router(str(ckpt), hidden_dim=512, n_layers=2, dropout=0.1).to(device).eval()
    print(f"Router checkpoint missing ({ckpt}); using dummy RouterMLP", flush=True)
    return RouterMLP(embed_dim=768, K=512, hidden_dim=512, n_layers=2, dropout=0.1).to(device).eval()


def load_ce_model(device: torch.device):
    from src.cross_encoder import CrossEncoderRerank, load_cross_encoder

    ckpt = CE["visualnews_task3"] / "cross_encoder.pt"
    if ckpt.exists():
        print(f"CE checkpoint: {ckpt}", flush=True)
        return load_cross_encoder(str(ckpt), hidden_dim=512, n_layers=3).to(device).eval()
    print(f"CE checkpoint missing ({ckpt}); using dummy CrossEncoderRerank", flush=True)
    return CrossEncoderRerank(embed_dim=768, hidden_dim=512, n_layers=3).to(device).eval()


def measure_qps_at_pool_size(
    query_emb: torch.Tensor,
    pool_emb: torch.Tensor,
    router: nn.Module,
    ce: nn.Module,
    pool_size: int,
    mode: str,
    device: torch.device,
    trial_seed: Optional[int] = None,
) -> dict:
    latencies = []
    n_q = query_emb.shape[0]
    pool_cap = pool_emb.shape[0]
    pool_size = max(1, min(pool_size, pool_cap))

    # Sample candidate pool once (outside timed loop); router_only skips this.
    candidates = None
    if mode in ("router_ce", "flat_cosine"):
        g = torch.Generator(device="cpu")
        if trial_seed is not None:
            g.manual_seed(trial_seed)
        idx = torch.randperm(pool_cap, generator=g)[:pool_size]
        candidates = pool_emb.index_select(0, idx).to(device)

    for i in range(WARMUP + N_MEASURE):
        q = query_emb[i % n_q].unsqueeze(0).to(device)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        with torch.no_grad():
            if mode == "router_only":
                _ = router(q)
            elif mode == "router_ce":
                _ = router(q)
                q_exp = q.expand(pool_size, -1)
                for s in range(0, pool_size, CE_CHUNK):
                    e = min(s + CE_CHUNK, pool_size)
                    _ = ce(q_exp[s:e], candidates[s:e])
            elif mode == "flat_cosine":
                _ = candidates @ q.T
            else:
                raise ValueError(mode)

        ender.record()
        torch.cuda.synchronize()
        if i >= WARMUP:
            latencies.append(starter.elapsed_time(ender))

    arr = np.asarray(latencies, dtype=np.float64)
    mean_ms = float(arr.mean())
    return {
        "pool_size": pool_size,
        "mean_ms": mean_ms,
        "std_ms": float(arr.std()),
        "p95_ms": float(np.percentile(arr, 95)),
        "qps": float(1000.0 / mean_ms),
    }


def measure_qps_trials(
    query_emb: torch.Tensor,
    pool_emb: torch.Tensor,
    router: nn.Module,
    ce: nn.Module,
    pool_size: int,
    mode: str,
    device: torch.device,
    n_trials: int = N_TRIALS,
) -> dict:
    trials = []
    for t in range(n_trials):
        trials.append(
            measure_qps_at_pool_size(
                query_emb,
                pool_emb,
                router,
                ce,
                pool_size,
                mode,
                device,
                trial_seed=1000 * pool_size + t,
            )
        )

    mean_ms_arr = np.array([x["mean_ms"] for x in trials], dtype=np.float64)
    mean_ms = float(mean_ms_arr.mean())
    trial_std_ms = float(mean_ms_arr.std(ddof=0))
    within_std = float(np.mean([x["std_ms"] for x in trials]))
    p95_ms = float(np.mean([x["p95_ms"] for x in trials]))

    return {
        "pool_size": trials[0]["pool_size"],
        "mean_ms": mean_ms,
        "std_ms": within_std,
        "trial_std_ms": trial_std_ms,
        "p95_ms": p95_ms,
        "qps": float(1000.0 / mean_ms),
        "n_trials": n_trials,
        "trials": trials,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for QPS benchmark (cuda.Event timing).")

    device = torch.device("cuda")
    print(f"QSBA_ROOT={QSBA}", flush=True)
    print(f"Device: {device}", flush=True)

    query_emb, pool_emb = load_embeddings()
    router = load_router_model(device)
    ce = load_ce_model(device)

    results = {"router_ce": [], "router_only": [], "flat_cosine": []}
    meta = {
        "K": K,
        "M": M,
        "B": B,
        "warmup": WARMUP,
        "n_measure": N_MEASURE,
        "n_trials": N_TRIALS,
        "query_source": str(DATA["visualnews_task3"] / "image_embeddings.pt"),
        "pool_source": str(DATA["visualnews_task3"] / "text_embeddings.pt"),
        "formula": "|C| = B * N * M / K",
    }

    for N in CORPUS_SIZES:
        pool_size = max(1, int(B * N * M / K))
        pool_size = min(pool_size, pool_emb.shape[0])
        print(f"\nN={N // 1000}K → |C|={pool_size}", flush=True)

        for mode in ("router_ce", "router_only", "flat_cosine"):
            res = measure_qps_trials(
                query_emb, pool_emb, router, ce, pool_size, mode, device
            )
            res["N_k"] = N // 1000
            res["N"] = N
            results[mode].append(res)
            trial_ms = [t["mean_ms"] for t in res["trials"]]
            print(
                f"  {mode}: {res['mean_ms']:.2f} ms ({res['qps']:.0f} QPS) "
                f"[trials: {', '.join(f'{x:.2f}' for x in trial_ms)}]",
                flush=True,
            )

    out_path = RESULTS / "qps_vs_size_raw.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "results": results}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
