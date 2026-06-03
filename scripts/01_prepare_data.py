"""01: 验证 embedding 数据加载是否正确。

用法:
    python scripts/01_prepare_data.py --config configs/coco_siglip2.yaml

输出:
    outputs/data_stats.txt: 各项诊断指标
"""
import argparse
import json
import sys
from pathlib import Path

# 让 scripts 能 import src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import diagnostic_stats, load_embeddings
from src.utils import ensure_dir, load_config, setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "01_prepare_data.log"))

    logger.info(f"Loading embeddings from:\n"
                f"  image: {cfg['data']['image_emb_path']}\n"
                f"  text:  {cfg['data']['text_emb_path']}")

    emb = load_embeddings(
        cfg["data"]["image_emb_path"],
        cfg["data"]["text_emb_path"],
        expected_dim=cfg["data"]["embed_dim"],
        normalize=True,
    )

    stats = diagnostic_stats(emb)
    logger.info(f"Diagnostic stats:\n{json.dumps(stats, indent=2)}")

    # 健康检查
    issues = []
    if stats["paired_text_image_sim"] < 0.2:
        issues.append("Paired text-image cosine sim < 0.2 — encoder mismatch?")
    if stats["paired_text_image_sim"] - stats["random_text_image_sim"] < 0.1:
        issues.append("Paired vs random gap too small — captions not aligned?")
    if abs(stats["img_norm_mean"] - 1.0) > 0.01:
        issues.append("Image embeddings not L2-normalized (expected norm ≈ 1)")

    if issues:
        logger.warning("Issues detected:")
        for i in issues:
            logger.warning(f"  - {i}")
    else:
        logger.info("✓ All checks passed")

    out_path = out_dir / "data_stats.json"
    with open(out_path, "w") as f:
        json.dump({"stats": stats, "issues": issues}, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
