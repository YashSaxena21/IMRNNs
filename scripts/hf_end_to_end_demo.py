from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

# Allow the demo to import the local IMRNN package directly from this model repo
# without requiring a separate editable installation step.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imrnns.caching import build_cache
from imrnns.beir_data import load_beir_source
from imrnns.checkpoints import load_model
from imrnns.data import load_cached_split
from imrnns.encoders import get_encoder_spec, normalize_encoder_name
from imrnns.evaluation import evaluate_model
from imrnns.model import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end IMRNN demo: download checkpoint from Hugging Face, build cache if needed, and evaluate."
    )
    parser.add_argument("--repo-id", default="yashsaxena21/IMRNNs")
    parser.add_argument("--encoder", required=True, help="minilm or e5")
    parser.add_argument("--dataset", required=True, help="BEIR dataset name")
    parser.add_argument("--checkpoint-path", help="Optional path inside the HF repo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--feedback-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-negatives", type=int, default=20)
    parser.add_argument("--negative-pool", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--datasets-dir", type=Path, default=Path("datasets"))
    return parser.parse_args()


def default_hf_checkpoint_path(encoder: str, dataset: str) -> str:
    normalized = normalize_encoder_name(encoder)
    display = "minilm" if normalized == "mini" else normalized
    return f"checkpoints/pretrained/{display}/imrnns-{display}-{dataset}.pt"


def main() -> int:
    args = parse_args()

    # Resolve which base retriever family should be used.
    # `minilm` maps to `all-MiniLM-L6-v2` and `e5` maps to `intfloat/e5-large-v2`.
    encoder_spec = get_encoder_spec(args.encoder)
    normalized_encoder = "minilm" if encoder_spec.key == "mini" else encoder_spec.key

    # Step 1:
    # Download the requested IMRNN checkpoint from the public Hugging Face model repo.
    # By default, the checkpoint path is inferred from the selected encoder and dataset.
    checkpoint_repo_path = args.checkpoint_path or default_hf_checkpoint_path(args.encoder, args.dataset)
    checkpoint_local_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=checkpoint_repo_path,
        repo_type="model",
    )

    # Step 2:
    # Choose where the local BEIR cache should live.
    # The cache contains:
    # - document embeddings
    # - query embeddings
    # - mined negatives
    # - a cache manifest
    cache_dir = args.cache_dir or Path("demo_cache") / f"cache_{normalized_encoder}_{args.dataset}"
    datasets_dir = args.datasets_dir

    # Step 3:
    # If the cache for this encoder/dataset pair does not exist yet, build it from scratch.
    # This uses the matching base retriever to embed the BEIR dataset locally.
    if not (cache_dir / "test" / "embeddings.pt").exists():
        build_cache(
            dataset_name=args.dataset,
            encoder_spec=encoder_spec,
            cache_dir=cache_dir,
            datasets_dir=datasets_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_negatives=args.num_negatives,
            negative_pool=args.negative_pool,
            max_queries=args.max_queries,
        )

    # Step 4:
    # Load the BEIR dataset and align it with the cached split artifacts so evaluation uses
    # the same query/document ids as the cached embeddings.
    source = load_beir_source(args.dataset, datasets_dir=datasets_dir, max_queries=args.max_queries)
    cached_test = load_cached_split(cache_dir, "test", source, encoder_spec, args.device)

    # Step 5:
    # Load the IMRNN checkpoint on top of the matching base retriever family.
    # The checkpoint contains the learned adapter weights used to modulate query and document
    # embeddings before ranking.
    model, metadata, missing, unexpected = load_model(
        checkpoint_path=Path(checkpoint_local_path),
        model_config=ModelConfig(input_dim=encoder_spec.embedding_dim),
        device=args.device,
    )

    # Step 6:
    # Run retrieval evaluation and report the final top-k metrics.
    # This prints the end-to-end result that users typically want to inspect first.
    metrics = evaluate_model(
        model=model,
        cached_split=cached_test,
        device=args.device,
        feedback_k=args.feedback_k,
        ranking_k=args.k,
        k_values=[args.k],
    )

    # Final output:
    # Return the checkpoint path, cache location, and evaluation metrics as JSON so the
    # script is easy to use in terminals, notebooks, or shell pipelines.
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "checkpoint": checkpoint_repo_path,
                "local_checkpoint": checkpoint_local_path,
                "encoder": args.encoder,
                "dataset": args.dataset,
                "cache_dir": str(cache_dir),
                "metrics": metrics,
                "metadata": metadata,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
            },
            indent=2,
        )
    )

    # Note for public users:
    # This demo focuses on loading a released checkpoint and evaluating it end to end.
    # If you want to train IMRNNs for additional retrievers or datasets, use the full
    # GitHub implementation, which includes the complete caching, training, and evaluation pipeline.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
