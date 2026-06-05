#!/usr/bin/env python3
"""QPS vs dataset size: CLIP sweep + QSBA/GENIUS/GRACE from existing results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import make_interp_spline

from paths import DATA, MBEIR, RESULTS

WARMUP = 50
N_MEASURE = 300
TOPK = 10
N_K_POINTS = [1, 5, 10, 25, 50, 100, 150, 200, 250, 300, 400, 537]

# Plot style (broken y-axis)
COLOR_OURS = "#C62828"
COLOR_CLIP_OURS = "#555555"
COLOR_GENIUS = "#1565C0"
COLOR_GRACE = "#5C6BC0"
CLIP_PAPER_X = np.array([0, 50, 100, 150, 200, 250, 300], dtype=float)
CLIP_PAPER_Y = np.array([24, 16, 10, 7, 5.5, 4, 3], dtype=float)
GENIUS_PAPER_QPS = 19.0
GRACE_PAPER_QPS = 5.0
X_MAX = 580
Y_TOP_LO, Y_TOP_HI = 150, 2100
Y_BOT_LO, Y_BOT_HI = 0, 28


def _check_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Set CUDA_VISIBLE_DEVICES.")
    return torch.device("cuda")


def measure_latency(
    fn: Callable[[torch.Tensor], None],
    queries: torch.Tensor,
    warmup: int = WARMUP,
    n: int = N_MEASURE,
) -> Tuple[float, float, float]:
    device = queries.device
    lats: List[float] = []
    n_q = queries.shape[0]
    for i in range(warmup + n):
        q = queries[i % n_q]
        if q.dim() == 1:
            q = q.unsqueeze(0)
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        with torch.no_grad():
            fn(q)
        ender.record()
        torch.cuda.synchronize(device)
        if i >= warmup:
            lats.append(starter.elapsed_time(ender))
    arr = np.asarray(lats, dtype=np.float64)
    return (
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.percentile(arr, 95)),
    )


def load_vn_embeddings(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """VN task3: image queries, text pool (same as 13_qps_vs_size.py)."""
    img_path = DATA["visualnews_task3"] / "image_embeddings.pt"
    txt_path = DATA["visualnews_task3"] / "text_embeddings.pt"
    for p in (img_path, txt_path):
        if not p.exists():
            raise FileNotFoundError(p)

    img = torch.load(img_path, map_location="cpu", weights_only=False)
    txt = torch.load(txt_path, map_location="cpu", weights_only=False)
    query_emb = F.normalize(img["features"].float(), dim=-1)
    pool_emb = F.normalize(txt["features"].float(), dim=-1)
    print(
        f"Pool (text): {pool_emb.shape[0]:,}, queries (image): {query_emb.shape[0]:,}",
        flush=True,
    )
    return query_emb.to(device), pool_emb.to(device)


def bench_clip_sweep(
    test_queries: torch.Tensor, pool_emb: torch.Tensor
) -> List[dict]:
    results = []
    print("\n[CLIP brute-force] sweep vs N", flush=True)
    for n_k in N_K_POINTS:
        n = min(n_k * 1000, pool_emb.shape[0])
        pool_subset = pool_emb[:n]

        def clip_fn(q: torch.Tensor, pool=pool_subset):
            scores = pool @ q.T
            scores.topk(min(TOPK, n), dim=0)

        mean_ms, std_ms, p95_ms = measure_latency(clip_fn, test_queries)
        qps = 1000.0 / mean_ms
        row = {
            "N_k": n_k,
            "N": n,
            "mean_ms": round(mean_ms, 4),
            "std_ms": round(std_ms, 4),
            "p95_ms": round(p95_ms, 4),
            "qps": round(qps, 4),
        }
        results.append(row)
        print(f"  N={n_k}K (n={n:,}): {mean_ms:.3f} ms ({qps:.0f} QPS)", flush=True)
    return results


def clip_sf_from_clip(clip_results: List[dict]) -> List[dict]:
    """VN precomputed: CLIP-SF scoring = CLIP brute-force dot product."""
    print("\n[CLIP-SF] VN — same latencies as CLIP (precomputed embeddings)", flush=True)
    return [
        {**r, "method": "CLIP-SF", "dataset": "VN task3"} for r in clip_results
    ]


def load_genius_vn() -> dict:
    path = RESULTS / "genius_latency_raw.json"
    if path.exists():
        d = json.loads(path.read_text())
        ms = float(d["mean_ms"])
        return {
            "mean_ms": ms,
            "std_ms": d.get("std_ms"),
            "p95_ms": d.get("p95_ms"),
            "qps": float(d.get("queries_per_sec") or 1000.0 / ms),
            "dataset": "VN task3",
            "source": str(path),
            "note": d.get("note", ""),
        }
    return {
        "mean_ms": 162.0,
        "qps": 6.17,
        "dataset": "VN task3",
        "source": "fallback",
    }


def load_grace_coco() -> dict:
    """GRACE: use COCO measured latency (horizontal line on VN size plot)."""
    path = RESULTS / "grace_latency_raw.json"
    if path.exists():
        d = json.loads(path.read_text())
        ms = d.get("mean_ms")
        if ms and ms > 0:
            return {
                "mean_ms": float(ms),
                "std_ms": d.get("std_ms"),
                "p95_ms": d.get("p95_ms"),
                "qps": float(d.get("queries_per_sec") or 1000.0 / ms),
                "dataset": d.get("dataset", "coco"),
                "source": str(path),
                "note": f"COCO {d.get('bundle', '')} n={d.get('n_measured', '?')}",
            }
    return {"mean_ms": 5060.0, "qps": 0.2, "dataset": "coco", "source": "fallback"}


def load_qsba_ce_results() -> List[dict]:
    path = RESULTS / "qps_vs_size_raw.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run 13_qps_vs_size.py first.")
    raw = json.loads(path.read_text())
    rows = raw.get("results", raw).get("router_ce", raw.get("router_ce", []))
    out = []
    for r in rows:
        n_k = r.get("N_k")
        if n_k is None:
            n = int(r.get("N", 0))
            n_k = n // 1000 if n >= 1000 else n
        out.append(
            {
                "N_k": int(n_k),
                "N": int(r.get("pool_size", r.get("N", n_k * 1000))),
                "mean_ms": r.get("mean_ms"),
                "qps": r.get("qps"),
            }
        )
    n_trials = raw.get("meta", {}).get("n_trials", 1)
    print(
        f"\n[CREST router+CE] loaded {len(out)} points from {path.name} "
        f"(n_trials={n_trials})",
        flush=True,
    )
    for r in out:
        print(f"  N={r['N_k']}K: {r['mean_ms']:.2f} ms ({r['qps']:.0f} QPS)", flush=True)
    return out


def horizontal_line(qps: float, note: str) -> List[dict]:
    return [{"N_k": n_k, "qps": qps, "note": note} for n_k in N_K_POINTS]


def _draw_break_marks(ax_top, ax_bot, size: float = 0.012) -> None:
    kwargs = dict(color="k", clip_on=False, linewidth=0.9)
    for ax, y in ((ax_top, -size), (ax_bot, 1 + size)):
        ax.plot((-size, +size), (y, y - size), transform=ax.transAxes, **kwargs)
        ax.plot((1 - size, 1 + size), (y, y - size), transform=ax.transAxes, **kwargs)


def print_latency_table_three(
    qsba_results: List[dict],
    genius: dict,
    grace: dict,
) -> None:
    cols = [str(k) for k in N_K_POINTS]

    def row_cells(rows: List[dict]) -> List[str]:
        by_k = {int(r["N_k"]): r for r in rows}
        cells = []
        for k in N_K_POINTS:
            r = by_k.get(k)
            if r:
                ms = r.get("mean_ms")
                qps = r.get("qps")
                if ms is not None:
                    cells.append(f"{ms:.2f}ms ({qps:.0f})")
                else:
                    cells.append(f"— ({qps:.0f})")
            else:
                cells.append("—")
        return cells

    lines = []
    lines.append("=" * 100)
    lines.append("Latency table — CREST / GENIUS / GRACE (batch=1, V100)")
    lines.append("CREST+GENIUS: VN task3 sweep | GRACE: COCO (horizontal, N-independent)")
    lines.append("=" * 100)
    header = f"{'Method':<22}" + "".join(f"{c:>14}" for c in cols)
    lines.append(header)
    lines.append("-" * len(header))
    lines.append(
        f"{'CREST+CE (VN)':<22}" + "".join(f"{c:>14}" for c in row_cells(qsba_results))
    )
    lines.append(
        f"{'GENIUS (VN)':<22}"
        + f"{genius['mean_ms']:.1f}ms ({genius['qps']:.2f})".center(14 * len(cols))
    )
    lines.append(
        f"{'GRACE (COCO)':<22}"
        + f"{grace['mean_ms']:.0f}ms ({grace['qps']:.3f})".center(14 * len(cols))
    )
    lines.append("=" * 100)

    text = "\n".join(lines)
    print(text, flush=True)
    out = RESULTS / "latency_table_vn_three.txt"
    out.write_text(text + "\n")
    print(f"Saved: {out}", flush=True)


def print_latency_table(
    clip_results: List[dict],
    clip_sf_results: List[dict],
    qsba_results: List[dict],
    genius: dict,
    grace: dict,
) -> None:
    cols = [str(k) for k in N_K_POINTS]

    def row_cells(rows: List[dict]) -> List[str]:
        by_k = {int(r["N_k"]): r for r in rows}
        cells = []
        for k in N_K_POINTS:
            r = by_k.get(k)
            if r:
                cells.append(f"{r['mean_ms']:.2f}ms ({r['qps']:.0f})")
            else:
                cells.append("—")
        return cells

    lines = []
    lines.append("=" * 100)
    lines.append("Latency table — VN task3 sweep (batch=1, V100, cuda.Event)")
    lines.append("GRACE row: COCO measurement (N-independent, shown as horizontal line)")
    lines.append("=" * 100)
    header = f"{'Method':<22}" + "".join(f"{c:>14}" for c in cols)
    lines.append(header)
    lines.append("-" * len(header))

    for name, rows in (
        ("CLIP (VN)", clip_results),
        ("CLIP-SF (VN)", clip_sf_results),
        ("CREST+CE (VN)", qsba_results),
    ):
        lines.append(f"{name:<22}" + "".join(f"{c:>14}" for c in row_cells(rows)))

    lines.append(
        f"{'GENIUS (VN)':<22}"
        + f"{genius['mean_ms']:.1f}ms({genius['qps']:.1f})".center(14 * len(cols))
    )
    lines.append(
        f"{'GRACE (COCO)':<22}"
        + f"{grace['mean_ms']:.0f}ms({grace['qps']:.2f})".center(14 * len(cols))
    )
    lines.append("=" * 100)

    text = "\n".join(lines)
    print(text, flush=True)
    (RESULTS / "latency_table_vn.txt").write_text(text + "\n")
    print(f"Saved: {RESULTS / 'latency_table_vn.txt'}", flush=True)


def _plot_three_on_axes(
    ax,
    qsba_results: List[dict],
    genius_qps: float,
    grace_qps: float,
) -> list:
    handles = []

    handles.append(
        ax.axhline(
            y=grace_qps,
            color=COLOR_GRACE,
            linewidth=1.8,
            linestyle="-.",
            label=f"GRACE (COCO, {grace_qps:.2f} QPS)",
            zorder=2,
        )
    )
    handles.append(
        ax.axhline(
            y=genius_qps,
            color=COLOR_GENIUS,
            linewidth=2.0,
            linestyle="-.",
            label=f"GENIUS (VN, {genius_qps:.1f} QPS)",
            zorder=2,
        )
    )

    if qsba_results:
        qn = np.array([r["N_k"] for r in qsba_results], dtype=float)
        qq = np.array([r["qps"] for r in qsba_results], dtype=float)
        if len(qn) >= 4:
            n_smooth = np.linspace(qn.min(), qn.max(), 300)
            qps_smooth = np.clip(make_interp_spline(qn, qq, k=3)(n_smooth), 0, None)
            h, = ax.plot(
                n_smooth,
                qps_smooth,
                color=COLOR_OURS,
                linewidth=2.8,
                label="CREST router+CE (VN, V100)",
                zorder=4,
            )
        else:
            h, = ax.plot(
                qn, qq, color=COLOR_OURS, linewidth=2.8, label="QSBA router+CE (VN, V100)", zorder=4
            )
        handles.append(h)
        ax.scatter(qn, qq, color=COLOR_OURS, s=40, edgecolors="white", linewidths=0.5, zorder=5)

    ax.axvline(x=537, color="#BDBDBD", linestyle=":", linewidth=1.0, zorder=1)
    return handles


def plot_figure_three(
    qsba_results: List[dict],
    genius_qps: float,
    grace_qps: float,
    out_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.5, 5.0),
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )

    handles = _plot_three_on_axes(ax_top, qsba_results, genius_qps, grace_qps)
    _plot_three_on_axes(ax_bot, qsba_results, genius_qps, grace_qps)

    ax_top.set_ylim(80, 3200)
    ax_bot.set_ylim(0, 12)
    ax_top.set_xlim(0, X_MAX)
    ax_top.set_xticks([0, 100, 200, 300, 400, 500])
    ax_top.set_yticks([500, 1000, 1500, 2000, 2500, 3000])
    ax_bot.set_yticks([0, 2, 4, 6, 8, 10])

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False)
    ax_bot.xaxis.tick_bottom()
    _draw_break_marks(ax_top, ax_bot)

    ax_bot.set_xlabel("Dataset Size (K)", fontsize=12)
    fig.text(0.04, 0.5, "Queries per Second", va="center", rotation="vertical", fontsize=12)

    for ax in (ax_top, ax_bot):
        ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.7)

    ax_top.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.92, edgecolor="#dddddd")
    ax_bot.text(525, 0.8, "VN\n(537K)", fontsize=8, color="gray", ha="center")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out_path}", flush=True)


def _plot_on_axes(
    ax,
    clip_results: List[dict],
    clip_sf_results: List[dict],
    qsba_results: List[dict],
    genius_qps: float,
    grace_qps: float,
) -> list:
    handles = []

    if clip_results:
        cn = np.array([r["N_k"] for r in clip_results], dtype=float)
        cq = np.array([r["qps"] for r in clip_results], dtype=float)
        h, = ax.plot(
            cn, cq, color=COLOR_CLIP_OURS, linewidth=2.2, label="CLIP (VN, V100)", zorder=3
        )
        handles.append(h)

    if clip_sf_results:
        csf_n = np.array([r["N_k"] for r in clip_sf_results], dtype=float)
        csf_q = np.array([r["qps"] for r in clip_sf_results], dtype=float)
        h, = ax.plot(
            csf_n,
            csf_q,
            color="#888888",
            linewidth=1.5,
            linestyle=":",
            label="CLIP-SF (VN, V100)",
            zorder=3,
        )
        handles.append(h)

    h, = ax.plot(
        CLIP_PAPER_X,
        CLIP_PAPER_Y,
        color=COLOR_CLIP_OURS,
        linewidth=1.5,
        linestyle="--",
        label="CLIP (Kim et al., RTX3090)",
        zorder=2,
    )
    handles.append(h)
    clip_ext_x = np.linspace(300, 537, 40)
    ax.plot(
        clip_ext_x,
        3.0 * (300.0 / clip_ext_x),
        color=COLOR_CLIP_OURS,
        linewidth=1.5,
        linestyle="--",
        zorder=2,
    )

    handles.append(
        ax.axhline(
            y=grace_qps,
            color=COLOR_GRACE,
            linewidth=1.8,
            linestyle="-.",
            label=f"GRACE (COCO, {grace_qps:.2f} QPS)",
            zorder=2,
        )
    )
    handles.append(
        ax.axhline(
            y=GRACE_PAPER_QPS,
            color="#9C27B0",
            linewidth=1.5,
            linestyle=":",
            label="GRACE (Kim et al., 5 QPS)",
            zorder=2,
        )
    )

    handles.append(
        ax.axhline(
            y=genius_qps,
            color=COLOR_GENIUS,
            linewidth=2.0,
            linestyle="-.",
            label=f"GENIUS (VN, {genius_qps:.1f} QPS)",
            zorder=2,
        )
    )
    handles.append(
        ax.axhline(
            y=GENIUS_PAPER_QPS,
            color="#F44336",
            linewidth=1.5,
            linestyle=":",
            label="GENIUS (Kim et al., 19 QPS)",
            zorder=2,
        )
    )

    if qsba_results:
        qn = np.array([r["N_k"] for r in qsba_results], dtype=float)
        qq = np.array([r["qps"] for r in qsba_results], dtype=float)
        if len(qn) >= 4:
            n_smooth = np.linspace(qn.min(), qn.max(), 300)
            qps_smooth = np.clip(make_interp_spline(qn, qq, k=3)(n_smooth), 0, None)
            h, = ax.plot(
                n_smooth,
                qps_smooth,
                color=COLOR_OURS,
                linewidth=2.8,
                label="CREST router+CE (VN, V100)",
                zorder=4,
            )
        else:
            h, = ax.plot(
                qn, qq, color=COLOR_OURS, linewidth=2.8, label="QSBA router+CE (VN, V100)", zorder=4
            )
        handles.append(h)
        ax.scatter(qn, qq, color=COLOR_OURS, s=40, edgecolors="white", linewidths=0.5, zorder=5)

    ax.axvline(x=537, color="#BDBDBD", linestyle=":", linewidth=1.0, zorder=1)
    return handles


def plot_figure(
    clip_results: List[dict],
    clip_sf_results: List[dict],
    qsba_results: List[dict],
    genius_qps: float,
    grace_qps: float,
    out_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.5, 5.4),
        gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.06},
    )

    handles = _plot_on_axes(
        ax_top, clip_results, clip_sf_results, qsba_results, genius_qps, grace_qps
    )
    _plot_on_axes(ax_bot, clip_results, clip_sf_results, qsba_results, genius_qps, grace_qps)

    ax_top.set_ylim(Y_TOP_LO, Y_TOP_HI)
    ax_bot.set_ylim(Y_BOT_LO, Y_BOT_HI)
    ax_top.set_xlim(0, X_MAX)
    ax_top.set_xticks([0, 100, 200, 300, 400, 500])
    ax_top.set_yticks([500, 1000, 1500, 2000])
    ax_bot.set_yticks([0, 5, 10, 15, 20, 25])

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False)
    ax_bot.xaxis.tick_bottom()
    _draw_break_marks(ax_top, ax_bot)

    ax_bot.set_xlabel("Dataset Size (K)", fontsize=12)
    fig.text(0.04, 0.5, "Queries per Second", va="center", rotation="vertical", fontsize=12)

    for ax in (ax_top, ax_bot):
        ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.7)

    ax_top.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92, edgecolor="#dddddd", ncol=2)
    ax_bot.text(525, 2.0, "VN\n(537K)", fontsize=8, color="gray", ha="center")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: {out_path}", flush=True)


def _load_previous_clip() -> List[dict]:
    path = RESULTS / "qps_vs_size_all_methods.json"
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("clip"):
            print(f"Reusing CLIP sweep from {path.name}", flush=True)
            return data["clip"]
    return []


def run_three_methods(args: argparse.Namespace) -> None:
    qsba_results = load_qsba_ce_results()
    genius_info = load_genius_vn()
    grace_info = load_grace_coco()
    genius_qps = genius_info["qps"]
    grace_qps = grace_info["qps"]

    print(
        f"\n[GENIUS VN] {genius_info['mean_ms']:.1f} ms → {genius_qps:.2f} QPS "
        f"({genius_info.get('source', '')})",
        flush=True,
    )
    print(
        f"[GRACE COCO] {grace_info['mean_ms']:.1f} ms → {grace_qps:.4f} QPS "
        f"({grace_info.get('note', '')})",
        flush=True,
    )

    payload = {
        "dataset": "visualnews_task3",
        "methods": ["CREST+CE", "GENIUS", "GRACE"],
        "qsba_ce": qsba_results,
        "genius_info": genius_info,
        "grace_info": grace_info,
        "genius": horizontal_line(genius_qps, "VN measured"),
        "grace": horizontal_line(grace_qps, "COCO measured"),
        "metadata": {
            "hardware": "V100",
            "batch": 1,
            "qsba_source": str(RESULTS / "qps_vs_size_raw.json"),
            "genius_source": genius_info.get("source"),
            "grace_source": grace_info.get("source"),
            "grace_dataset": "coco",
        },
    }

    out_json = RESULTS / "qps_vs_size_three_methods.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_json}", flush=True)

    print_latency_table_three(qsba_results, genius_info, grace_info)

    if not args.skip_plot:
        plot_figure_three(
            qsba_results,
            genius_qps,
            grace_qps,
            RESULTS / "qps_vs_size_three_methods.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-clip",
        action="store_true",
        help="Include CLIP/CLIP-SF sweep and full comparison plot (default: QSBA/GENIUS/GRACE only)",
    )
    parser.add_argument(
        "--refresh-qsba-only",
        action="store_true",
        help="Skip CLIP sweep; reload QSBA from qps_vs_size_raw.json and replot",
    )
    parser.add_argument("--skip-plot", action="store_true", help="Only write JSON, no figure")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    if not args.with_clip:
        print("=== VN task3 — CREST / GENIUS / GRACE ===", flush=True)
        run_three_methods(args)
        return

    if args.refresh_qsba_only:
        prev = json.loads((RESULTS / "qps_vs_size_all_methods.json").read_text())
        clip_results = prev.get("clip", _load_previous_clip())
        clip_sf_results = prev.get("clip_sf", clip_results)
        qsba_results = load_qsba_ce_results()
        genius_info = load_genius_vn()
        grace_info = load_grace_coco()
        genius_qps = genius_info["qps"]
        grace_qps = grace_info["qps"]
        genius_results = horizontal_line(genius_qps, "VN " + genius_info.get("source", ""))
        grace_results = horizontal_line(grace_qps, "COCO " + grace_info.get("source", ""))
        payload = {
            "clip": clip_results,
            "clip_sf": clip_sf_results,
            "genius": genius_results,
            "genius_info": genius_info,
            "qsba_ce": qsba_results,
            "grace": grace_results,
            "grace_info": grace_info,
            "genius_orig": {"qps": GENIUS_PAPER_QPS, "note": "Kim et al. RTX3090"},
            "grace_orig": {"qps": GRACE_PAPER_QPS, "note": "Kim et al. RTX3090"},
            "metadata": {
                "hardware": "V100",
                "batch": 1,
                "qsba_source": str(RESULTS / "qps_vs_size_raw.json"),
                "qsba_refreshed": True,
            },
        }
        out_json = RESULTS / "qps_vs_size_all_methods.json"
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved: {out_json}", flush=True)
        print_latency_table(clip_results, clip_sf_results, qsba_results, genius_info, grace_info)
        if not args.skip_plot:
            plot_figure(
                clip_results,
                clip_sf_results,
                qsba_results,
                genius_qps,
                grace_qps,
                RESULTS / "qps_vs_size_all_methods.png",
            )
        return

    device = _check_cuda()
    print("=== VN task3 QPS vs dataset size ===", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"N_K_POINTS: {N_K_POINTS}", flush=True)

    query_emb, pool_emb = load_vn_embeddings(device)
    n_need = WARMUP + N_MEASURE
    test_queries = query_emb[-max(20000, n_need) :]

    clip_results = bench_clip_sweep(test_queries, pool_emb)
    clip_sf_results = clip_sf_from_clip(clip_results)

    genius_info = load_genius_vn()
    genius_qps = genius_info["qps"]
    print(
        f"\n[GENIUS VN] {genius_info['mean_ms']:.1f} ms → {genius_qps:.2f} QPS "
        f"({genius_info.get('source', '')})",
        flush=True,
    )
    genius_results = horizontal_line(genius_qps, "VN measured")

    qsba_results = load_qsba_ce_results()

    grace_info = load_grace_coco()
    grace_qps = grace_info["qps"]
    print(
        f"[GRACE COCO] {grace_info['mean_ms']:.1f} ms → {grace_qps:.4f} QPS "
        f"({grace_info.get('note', '')})",
        flush=True,
    )
    grace_results = horizontal_line(grace_qps, "COCO measured (horizontal on VN plot)")

    payload = {
        "dataset": "visualnews_task3",
        "clip": clip_results,
        "clip_sf": clip_sf_results,
        "genius": genius_results,
        "genius_info": genius_info,
        "qsba_ce": qsba_results,
        "grace": grace_results,
        "grace_info": grace_info,
        "genius_orig": {"qps": GENIUS_PAPER_QPS, "note": "Kim et al. RTX3090"},
        "grace_orig": {"qps": GRACE_PAPER_QPS, "note": "Kim et al. RTX3090"},
        "metadata": {
            "hardware": "V100",
            "batch": 1,
            "warmup": WARMUP,
            "n_measure": N_MEASURE,
            "pool": "VN task3 text pool (image queries)",
            "pool_path": str(DATA["visualnews_task3"] / "text_embeddings.pt"),
            "query_path": str(DATA["visualnews_task3"] / "image_embeddings.pt"),
            "qsba_source": str(RESULTS / "qps_vs_size_raw.json"),
            "genius_source": genius_info.get("source"),
            "grace_source": grace_info.get("source"),
            "grace_dataset": "coco",
            "emb_precomputed": True,
            "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
        },
    }

    out_json = RESULTS / "qps_vs_size_all_methods.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_json}", flush=True)

    print_latency_table(clip_results, clip_sf_results, qsba_results, genius_info, grace_info)

    plot_figure(
        clip_results,
        clip_sf_results,
        qsba_results,
        genius_qps,
        grace_qps,
        RESULTS / "qps_vs_size_all_methods.png",
    )


if __name__ == "__main__":
    main()
