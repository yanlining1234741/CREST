"""ablate: 消融实验,定位是哪个模块影响 router recall。

用法:
    python scripts/ablate.py --config configs/coco_siglip2.yaml \\
        --sweep alpha --values 1.0,0.7,0.5,0.3,0.0

支持的 sweep:
    alpha    : Stage 1 query-aware 混合系数
    K        : 桶数量
    M        : multi-view 每 item 桶数
    sinkhorn : on/off (off 退化为纯 k-means)

每个值跑完整 pipeline 并记录 router_recall@B,输出 CSV。
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, load_config, setup_logger

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_cmd(cmd, logger):
    logger.info(f"$ {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"stderr:\n{res.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return res.stdout


def run_one(cfg_path: str, out_dir: Path, alpha: float, K: int, M: int,
            init: str, epochs: int, logger) -> dict:
    """跑完整 pipeline 一次,返回 metrics。"""
    tag = f"a{alpha}_K{K}_M{M}_{init}"
    qa_path = out_dir / f"qa_{tag}.pt"
    assign_path = out_dir / f"assign_{tag}.pt"
    router_path = out_dir / f"router_{tag}.pt"
    eval_path = out_dir / f"eval_{tag}.json"

    # Stage 1
    run_cmd([PY, str(ROOT / "scripts/02_build_query_aware.py"),
             "--config", cfg_path, "--alpha", str(alpha),
             "--output", str(qa_path)], logger)

    # Stage 2
    run_cmd([PY, str(ROOT / "scripts/03_run_sinkhorn.py"),
             "--config", cfg_path, "--input", str(qa_path),
             "--K", str(K), "--M", str(M), "--init", init,
             "--output", str(assign_path)], logger)

    # Stage 3
    run_cmd([PY, str(ROOT / "scripts/04_train_router.py"),
             "--config", cfg_path, "--assignment", str(assign_path),
             "--epochs", str(epochs),
             "--output", str(router_path)], logger)

    # Eval
    run_cmd([PY, str(ROOT / "scripts/05_evaluate.py"),
             "--config", cfg_path, "--assignment", str(assign_path),
             "--router", str(router_path),
             "--output", str(eval_path)], logger)

    with open(eval_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sweep", required=True,
                        choices=["alpha", "K", "M", "sinkhorn"])
    parser.add_argument("--values", required=True,
                        help="逗号分隔,如 1.0,0.5,0.3")
    parser.add_argument("--epochs", type=int, default=15,
                        help="消融用短训练即可")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(Path(cfg["output_dir"]) / f"ablate_{args.sweep}")
    logger = setup_logger(log_file=str(out_dir / "ablate.log"))

    base_alpha = cfg["query_aware"]["alpha"]
    base_K = cfg["sinkhorn"]["K"]
    base_M = cfg["sinkhorn"]["M"]
    base_init = cfg["sinkhorn"]["init"]

    values = args.values.split(",")

    rows = []
    for v in values:
        if args.sweep == "alpha":
            alpha, K, M, init = float(v), base_K, base_M, base_init
        elif args.sweep == "K":
            alpha, K, M, init = base_alpha, int(v), base_M, base_init
        elif args.sweep == "M":
            alpha, K, M, init = base_alpha, base_K, int(v), base_init
        elif args.sweep == "sinkhorn":
            # off → random init,可视为不平衡
            alpha = base_alpha
            K, M = base_K, base_M
            init = "random" if v.lower() == "off" else "kmeans++"
        else:
            raise ValueError(f"Unknown sweep: {args.sweep}")

        logger.info(f"=== Sweep {args.sweep}={v} ===")
        metrics = run_one(args.config, out_dir, alpha, K, M, init,
                          args.epochs, logger)
        row = {args.sweep: v}
        # 关键指标列
        for b in cfg["eval"]["top_B"]:
            row[f"router_R@{b}"] = metrics.get(f"router_recall@{b}", 0)
            row[f"cands@B={b}"] = metrics.get(f"candidates@B={b}", 0)
            for k in cfg["eval"]["top_K"]:
                row[f"R@{k}|B={b}"] = metrics.get(f"recall@{k}|B={b}", 0)
        rows.append(row)

    # 输出 CSV
    csv_path = out_dir / f"ablate_{args.sweep}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"✓ Saved sweep results to {csv_path}")

    # 打印简表
    logger.info(f"\n{args.sweep} sweep summary:")
    for r in rows:
        logger.info(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
