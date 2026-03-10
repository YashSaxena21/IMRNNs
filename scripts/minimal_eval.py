from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imrnns.beir_data import load_beir_source
from imrnns.checkpoints import load_model
from imrnns.data import load_cached_split
from imrnns.encoders import get_encoder_spec
from imrnns.evaluation import evaluate_model
from imrnns.model import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal IMRNN checkpoint evaluator.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--datasets-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--feedback-k", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    encoder_spec = get_encoder_spec(args.encoder)
    source = load_beir_source(args.dataset, datasets_dir=args.datasets_dir)
    cached_test = load_cached_split(args.cache_dir, "test", source, encoder_spec, args.device)
    model, metadata, missing, unexpected = load_model(
        checkpoint_path=args.checkpoint,
        model_config=ModelConfig(input_dim=encoder_spec.embedding_dim),
        device=args.device,
    )
    metrics = evaluate_model(
        model=model,
        cached_split=cached_test,
        device=args.device,
        feedback_k=args.feedback_k,
        ranking_k=args.k,
        k_values=[args.k],
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "metrics": metrics,
                "metadata": metadata,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
