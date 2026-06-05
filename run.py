#!/usr/bin/env python3
"""CREST top-level entry point.

Examples:
  python run.py setup --data-root /path/to/mbeir_aligned
  python run.py train --dataset flickr
  python run.py train --dataset mscoco --stage 3
  python run.py benchmark --task qps
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def cmd_setup(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "tools/setup_configs.py"]
    if args.data_root:
        cmd += ["--data-root", args.data_root]
    subprocess.run(cmd, cwd=REPO, check=True)


def cmd_train(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "stages/pipeline.py", "--dataset", args.dataset]
    if args.stage:
        cmd += ["--stage", args.stage]
    if args.skip_if_exists:
        cmd += ["--skip-if-exists"]
    subprocess.run(cmd, cwd=REPO, check=True)


def cmd_benchmark(args: argparse.Namespace) -> None:
    if args.task == "qps":
        subprocess.run([sys.executable, "benchmarks/13_qps_vs_size.py"], cwd=REPO, check=True)
        subprocess.run([sys.executable, "benchmarks/19_qps_vs_size_all_methods.py"], cwd=REPO, check=True)
    else:
        raise ValueError(f"Unknown benchmark: {args.task}")


def cmd_verify(args: argparse.Namespace) -> None:
    subprocess.run(
        [sys.executable, "stages/pipeline.py", "--dataset", args.dataset, "--stage", "0"],
        cwd=REPO,
        check=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        prog="crest",
        description="CREST: Collapse-Free Routing with Equipartitioned Semantic Buckets",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="Generate dataset configs from CREST_DATA_ROOT")
    s.add_argument("--data-root", default=None)
    s.set_defaults(func=cmd_setup)

    t = sub.add_parser("train", help="Run training/eval pipeline")
    t.add_argument("--dataset", required=True, choices=["flickr", "mscoco", "visualnews_task3"])
    t.add_argument("--stage", default="all", help="0-5 or 'all'")
    t.add_argument("--skip-if-exists", action="store_true")
    t.set_defaults(func=cmd_train)

    v = sub.add_parser("verify", help="Verify data loading (stage 0)")
    v.add_argument("--dataset", required=True, choices=["flickr", "mscoco", "visualnews_task3"])
    v.set_defaults(func=cmd_verify)

    b = sub.add_parser("benchmark", help="Run efficiency benchmarks")
    b.add_argument("--task", default="qps", choices=["qps"])
    b.set_defaults(func=cmd_benchmark)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
