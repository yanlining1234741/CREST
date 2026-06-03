"""评估: 含 cross-encoder rerank 选项"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_embeddings
from src.evaluate import evaluate_end_to_end
from src.router import load_router
from src.cross_encoder import load_cross_encoder
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--pool-start", type=int, default=82783)
    parser.add_argument("--pool-end", type=int, default=None)
    parser.add_argument("--query-start", type=int, default=100000)
    parser.add_argument("--query-end", type=int, default=None)
    parser.add_argument("--router-prior-lambda", type=float, default=0.0)
    parser.add_argument("--rerank-mode", type=str, default="flat",
                        choices=["flat", "groupwise", "cross_encoder"])
    parser.add_argument("--cross-encoder", type=str, default=None,
                        help="Path to trained cross_encoder.pt")
    parser.add_argument("--ce-hidden-dim", type=int, default=512)
    parser.add_argument("--ce-n-layers", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["output_dir"])
    logger = setup_logger(log_file=str(out_dir / "06_evaluate_correct.log"))

    logger.info("Loading embeddings...")
    emb = load_embeddings(
        cfg["data"]["image_emb_path"],
        cfg["data"]["text_emb_path"],
        expected_dim=cfg["data"]["embed_dim"],
    )
    assignment_blob = torch.load(args.assignment, map_location="cpu", weights_only=False)

    logger.info(f"Loading router from {args.router}")
    router = load_router(
        args.router,
        hidden_dim=cfg["router"]["hidden_dim"],
        n_layers=cfg["router"]["n_layers"],
        dropout=cfg["router"]["dropout"],
    )

    cross_encoder = None
    if args.rerank_mode == "cross_encoder":
        assert args.cross_encoder, "--cross-encoder required for cross_encoder mode"
        logger.info(f"Loading cross-encoder from {args.cross_encoder}")
        cross_encoder = load_cross_encoder(
            args.cross_encoder,
            hidden_dim=args.ce_hidden_dim,
            n_layers=args.ce_n_layers,
        )

    K = assignment_blob["K"]
    output = args.output or str(out_dir / f"eval_K{K}_correct.json")
    device = str(get_device(cfg["data"]["device"]))

    pool_end = args.pool_end or emb.image_features.shape[0]
    pool_image_features = emb.image_features[args.pool_start:pool_end]
    pool_image_ids = emb.image_ids[args.pool_start:pool_end]
    pool_assignment = assignment_blob["hard_assignment"][args.pool_start:pool_end]

    logger.info(f"Candidate pool: idx [{args.pool_start}, {pool_end}), size = {len(pool_image_ids)}")

    query_end = args.query_end or emb.text_features.shape[0]
    eval_text_features = emb.text_features[args.query_start:query_end]
    eval_text_image_ids = emb.text_image_ids[args.query_start:query_end]

    logger.info(f"Eval queries: idx [{args.query_start}, {query_end}), size = {len(eval_text_image_ids)}")
    logger.info(f"Mode: {args.rerank_mode}, lambda: {args.router_prior_lambda}")

    metrics = evaluate_end_to_end(
        router=router,
        text_features=eval_text_features,
        text_image_ids=eval_text_image_ids,
        image_features=pool_image_features,
        image_ids=pool_image_ids,
        hard_assignment=pool_assignment,
        top_B_list=tuple(cfg["eval"]["top_B"]),
        top_K_list=tuple(cfg["eval"]["top_K"]),
        device=device,
        candidate_pool_size=len(pool_image_ids),
        router_prior_lambda=args.router_prior_lambda,
        rerank_mode=args.rerank_mode,
        cross_encoder=cross_encoder,
    )

    logger.info(f"Saving to {output}")
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
