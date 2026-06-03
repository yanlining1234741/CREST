"""02: 构造 Query-Aware Item Representation (Stage 1)。

用法:
    python scripts/02_build_query_aware.py --config configs/coco_siglip2.yaml --alpha 0.5

输出:
    outputs/query_aware_emb_alpha<a>.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_embeddings
from src.query_aware import build_query_aware_embeddings, save_query_aware
from src.utils import ensure_dir, load_config, set_seed, setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--alpha", type=float, default=None,
                        help="覆盖 config 中的 alpha")
    parser.add_argument("--output", type=str, default=None,
                        help="覆盖输出路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "02_build_query_aware.log"))

    alpha = args.alpha if args.alpha is not None else cfg["query_aware"]["alpha"]
    output = args.output or str(out_dir / f"query_aware_emb_alpha{alpha}.pt")

    logger.info(f"Loading embeddings...")
    emb = load_embeddings(
        cfg["data"]["image_emb_path"],
        cfg["data"]["text_emb_path"],
        expected_dim=cfg["data"]["embed_dim"],
    )
    logger.info(f"Loaded: {emb.n_images} images, {emb.n_captions} captions")

    logger.info(f"Building query-aware features with alpha={alpha}")
    result = build_query_aware_embeddings(
        emb,
        alpha=alpha,
        l2_norm=cfg["query_aware"]["l2_normalize"],
    )

    logger.info(f"Diagnostic:\n{json.dumps(result.diagnostic, indent=2)}")

    save_query_aware(result, output)
    logger.info(f"✓ Saved to {output}")

    # 关键诊断: query-aware 后的 sim 应该高于原 image
    delta = result.diagnostic["delta"]
    if alpha < 1.0:
        if delta > 0.01:
            logger.info(f"✓ Query-aware effective: paired sim +{delta:.4f}")
        else:
            logger.warning(f"⚠ Query-aware delta {delta:.4f} too small, "
                           f"alpha may need tuning")


if __name__ == "__main__":
    main()
