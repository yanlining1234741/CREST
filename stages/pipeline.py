#!/usr/bin/env python3
"""
CREST end-to-end pipeline — stage-by-stage reproduction.

Stages:
  0  verify_data      Check embeddings load correctly
  1  query_aware      Build query-aware representations (α=1.0 for M-BEIR)
  2  assign_buckets   Sinkhorn equipartitioned assignment + text_with_buckets
  3  train_router     Train query→bucket router (disjoint protocol)
  4  train_cross_encoder   Train CE reranker on router candidates
  5  evaluate         End-to-end retrieval with CE rerank

Usage:
  python stages/pipeline.py --dataset flickr --stage all
  python stages/pipeline.py --dataset mscoco --stage 3
  python stages/pipeline.py --dataset visualnews_task3 --stage 5 --skip-if-exists
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from crest.datasets import get_dataset
from crest.paths import config_path, cross_encoder_dir, data_root, dataset_dir, output_dir

STAGES = ("0", "1", "2", "3", "4", "5")
STAGE_NAMES = {
    "0": "verify_data",
    "1": "query_aware",
    "2": "assign_buckets",
    "3": "train_router",
    "4": "train_cross_encoder",
    "5": "evaluate",
}


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO), check=True)


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def stage0_verify(ds_name: str, cfg: Path, dry_run: bool) -> None:
    _run([sys.executable, "scripts/01_prepare_data.py", "--config", str(cfg)], dry_run)


def stage1_query_aware(ds_name: str, cfg: Path, out: Path, dry_run: bool, skip: bool) -> Path:
    spec = get_dataset(ds_name)
    alpha = 1.0
    qa_path = out / f"query_aware_emb_alpha{alpha}.pt"
    if skip and _exists(qa_path):
        print(f"[skip] {qa_path}")
        return qa_path
    _run(
        [
            sys.executable,
            "scripts/02_build_query_aware.py",
            "--config",
            str(cfg),
            "--alpha",
            str(alpha),
            "--output",
            str(qa_path),
        ],
        dry_run,
    )
    return qa_path


def stage2_assign(
    ds_name: str, cfg: Path, out: Path, data: Path, qa_path: Path, dry_run: bool, skip: bool
) -> tuple[Path, Path]:
    spec = get_dataset(ds_name)
    km = f"K{spec.K}_M{spec.M}"
    assign_path = out / f"assignment_{km}.pt"
    text_buckets = data / "text_with_buckets.pt"

    if not (skip and _exists(assign_path)):
        _run(
            [
                sys.executable,
                "scripts/03_run_sinkhorn.py",
                "--config",
                str(cfg),
                "--input",
                str(qa_path),
                "--K",
                str(spec.K),
                "--M",
                str(spec.M),
                "--output",
                str(assign_path),
            ],
            dry_run,
        )
    else:
        print(f"[skip] {assign_path}")

    if not (skip and _exists(text_buckets)):
        _run(
            [
                sys.executable,
                "scripts/flickr_build_text.py",
                "--data",
                str(data),
                "--assignment",
                str(assign_path),
                "--centroids-from",
                str(assign_path),
            ],
            dry_run,
        )
    else:
        print(f"[skip] {text_buckets}")

    return assign_path, text_buckets


def stage3_router(
    ds_name: str, cfg: Path, out: Path, text_buckets: Path, dry_run: bool, skip: bool
) -> Path:
    spec = get_dataset(ds_name)
    router_path = out / f"router_K{spec.K}.pt"
    if skip and _exists(router_path):
        print(f"[skip] {router_path}")
        return router_path
    cmd = [
        sys.executable,
        "scripts/04b_train_router_disjoint.py",
        "--config",
        str(cfg),
        "--text-buckets",
        str(text_buckets),
        "--output",
        str(router_path),
    ]
    if spec.router_epochs:
        cmd += ["--epochs", str(spec.router_epochs)]
    _run(cmd, dry_run)
    return router_path


def stage4_ce(
    ds_name: str,
    cfg: Path,
    data: Path,
    out: Path,
    assign_path: Path,
    text_buckets: Path,
    dry_run: bool,
    skip: bool,
) -> Path:
    spec = get_dataset(ds_name)
    ce_out = cross_encoder_dir(ds_name, spec.K, spec.M)
    ce_out.mkdir(parents=True, exist_ok=True)
    ce_pt = ce_out / "cross_encoder.pt"
    if skip and _exists(ce_pt):
        print(f"[skip] {ce_pt}")
        return ce_pt
    train_img = data / "train_image_embeddings.pt"
    if not train_img.exists():
        train_img = data / "image_embeddings.pt"
        print(f"[note] train_image_embeddings.pt missing, using image_embeddings.pt")

    _run(
        [
            sys.executable,
            "scripts/07b_train_ce_disjoint.py",
            "--config",
            str(cfg),
            "--assignment",
            str(assign_path),
            "--text-buckets",
            str(text_buckets),
            "--image-emb",
            str(data / "image_embeddings.pt"),
            "--train-image-emb",
            str(train_img),
            "--output",
            str(ce_pt),
            "--epochs",
            str(spec.ce_epochs),
        ],
        dry_run,
    )
    return ce_pt


def stage5_eval(
    ds_name: str,
    cfg: Path,
    data: Path,
    out: Path,
    assign_path: Path,
    router_path: Path,
    text_buckets: Path,
    ce_pt: Path,
    dry_run: bool,
    skip: bool,
) -> Path:
    spec = get_dataset(ds_name)
    ce_out = cross_encoder_dir(ds_name, spec.K, spec.M)
    eval_json = ce_out / "eval_cross_encoder.json"
    if skip and _exists(eval_json):
        print(f"[skip] {eval_json}")
        return eval_json
    _run(
        [
            sys.executable,
            "scripts/06b_evaluate_disjoint.py",
            "--config",
            str(cfg),
            "--assignment",
            str(assign_path),
            "--router",
            str(router_path),
            "--text-buckets",
            str(text_buckets),
            "--image-emb",
            str(data / "image_embeddings.pt"),
            "--output",
            str(eval_json),
            "--rerank-mode",
            "cross_encoder",
            "--cross-encoder",
            str(ce_pt),
        ],
        dry_run,
    )
    return eval_json


def run_pipeline(
    dataset: str,
    stages: list[str],
    skip_if_exists: bool = False,
    dry_run: bool = False,
) -> None:
    spec = get_dataset(dataset)
    cfg = config_path(dataset)
    data = dataset_dir(dataset)
    out = output_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CREST pipeline — {dataset}")
    print(f"  K={spec.K}, M={spec.M}")
    print(f"  data:   {data}")
    print(f"  output: {out}")
    print(f"  config: {cfg}")
    print("=" * 60)

    qa_path = out / "query_aware_emb_alpha1.0.pt"
    assign_path = out / f"assignment_K{spec.K}_M{spec.M}.pt"
    text_buckets = data / "text_with_buckets.pt"
    router_path = out / f"router_K{spec.K}.pt"
    ce_pt = cross_encoder_dir(dataset, spec.K, spec.M) / "cross_encoder.pt"

    for s in stages:
        name = STAGE_NAMES[s]
        print(f"\n{'='*60}\nStage {s}: {name}\n{'='*60}")
        if s == "0":
            stage0_verify(dataset, cfg, dry_run)
        elif s == "1":
            qa_path = stage1_query_aware(dataset, cfg, out, dry_run, skip_if_exists)
        elif s == "2":
            assign_path, text_buckets = stage2_assign(
                dataset, cfg, out, data, qa_path, dry_run, skip_if_exists
            )
        elif s == "3":
            router_path = stage3_router(dataset, cfg, out, text_buckets, dry_run, skip_if_exists)
        elif s == "4":
            ce_pt = stage4_ce(
                dataset, cfg, data, out, assign_path, text_buckets, dry_run, skip_if_exists
            )
        elif s == "5":
            eval_json = stage5_eval(
                dataset,
                cfg,
                data,
                out,
                assign_path,
                router_path,
                text_buckets,
                ce_pt,
                dry_run,
                skip_if_exists,
            )
            print(f"\n✓ Evaluation saved: {eval_json}")

    print("\n✓ Pipeline finished.")


def main() -> None:
    p = argparse.ArgumentParser(description="CREST reproduction pipeline")
    from crest.datasets import DATASETS

    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))

    p.add_argument(
        "--stage",
        default="all",
        help="Stage number 0-5, or 'all' (default)",
    )
    p.add_argument("--skip-if-exists", action="store_true", help="Skip stage if outputs exist")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = p.parse_args()

    if args.stage == "all":
        stages = list(STAGES)
    else:
        stages = [args.stage]

    run_pipeline(args.dataset, stages, args.skip_if_exists, args.dry_run)


if __name__ == "__main__":
    main()
