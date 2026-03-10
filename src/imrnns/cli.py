from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import (
    default_assets_root,
    discover_cached_embeddings,
    discover_checkpoints,
    discover_repo_checkpoints,
    package_root,
    resolve_cache_dir,
    resolve_checkpoint_path,
)
from .beir_data import load_beir_source
from .caching import build_cache
from .checkpoints import default_checkpoint_name, load_model, save_checkpoint
from .data import ContrastiveCachedDataset, load_cached_split
from .encoders import get_encoder_spec
from .evaluation import evaluate_model
from .model import BiHyperNetIR, ModelConfig
from .training import TrainingConfig, train_model


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--assets-root", type=Path, default=default_assets_root())
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")


def _command_list_assets(args: argparse.Namespace) -> int:
    payload = {
        "assets_root": str(args.assets_root),
        "repo_root": str(package_root()),
        "cached_embeddings": [
            {"encoder": item.encoder, "dataset": item.dataset, "path": str(item.path)}
            for item in discover_cached_embeddings(args.assets_root)
        ],
        "workspace_checkpoints": [
            {"encoder": item.encoder, "dataset": item.dataset, "path": str(item.path)}
            for item in discover_checkpoints(args.assets_root)
        ],
        "repo_checkpoints": [
            {"encoder": item.encoder, "dataset": item.dataset, "path": str(item.path)}
            for item in discover_repo_checkpoints(package_root())
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _load_training_inputs(args: argparse.Namespace):
    encoder_spec = get_encoder_spec(args.encoder)
    cache_dir = args.cache_dir or resolve_cache_dir(args.assets_root, args.encoder, args.dataset)
    datasets_dir = args.assets_root / "datasets"
    beir_source = load_beir_source(args.dataset, datasets_dir=datasets_dir, max_queries=args.max_queries)
    train_split = load_cached_split(cache_dir, "train", beir_source, encoder_spec, args.device)
    val_split = load_cached_split(cache_dir, "val", beir_source, encoder_spec, args.device)
    test_split = load_cached_split(cache_dir, "test", beir_source, encoder_spec, args.device)
    return encoder_spec, cache_dir, train_split, val_split, test_split


def _k_values(args: argparse.Namespace) -> list[int]:
    return [args.k]


def _command_cache(args: argparse.Namespace) -> int:
    encoder_spec = get_encoder_spec(args.encoder)
    cache_dir = args.cache_dir or (args.assets_root / f"cache_{args.encoder}_{args.dataset}")
    built = build_cache(
        dataset_name=args.dataset,
        encoder_spec=encoder_spec,
        cache_dir=cache_dir,
        datasets_dir=args.assets_root / "datasets",
        device=args.device,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        negative_pool=args.negative_pool,
    )
    print(json.dumps({"cache_dir": str(built), "encoder": args.encoder, "dataset": args.dataset}, indent=2))
    return 0


def _command_train(args: argparse.Namespace) -> int:
    encoder_spec, cache_dir, train_split, val_split, test_split = _load_training_inputs(args)
    model = BiHyperNetIR(
        ModelConfig(
            input_dim=encoder_spec.embedding_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    )

    train_dataset = ContrastiveCachedDataset(train_split, args.num_negatives)
    val_dataset = ContrastiveCachedDataset(val_split, args.num_negatives)
    if len(train_dataset) == 0:
        raise ValueError(
            "No training examples were constructed from the cached split. "
            "Increase --max-queries or verify that cached negatives/embeddings match the dataset split."
        )
    if len(val_dataset) == 0:
        raise ValueError(
            "No validation examples were constructed from the cached split. "
            "Increase --max-queries or verify that cached negatives/embeddings match the dataset split."
        )
    metrics = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_negatives=args.num_negatives,
        ),
        device=args.device,
    )

    eval_metrics = evaluate_model(
        model=model,
        cached_split=test_split,
        device=args.device,
        feedback_k=args.feedback_k,
        ranking_k=args.ranking_k,
        k_values=_k_values(args),
    )

    output_dir = args.output_dir or args.assets_root
    checkpoint_path = output_dir / default_checkpoint_name(args.encoder, args.dataset)
    metadata = {
        "encoder": args.encoder,
        "dataset": args.dataset,
        "cache_dir": str(cache_dir),
        "model_config": {
            "input_dim": encoder_spec.embedding_dim,
            "output_dim": args.output_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
        "training": metrics,
        "evaluation": eval_metrics,
    }
    save_checkpoint(checkpoint_path, model, metadata)
    print(json.dumps({"checkpoint": str(checkpoint_path), "training": metrics, "evaluation": eval_metrics}, indent=2))
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    encoder_spec = get_encoder_spec(args.encoder)
    cache_dir = args.cache_dir or resolve_cache_dir(args.assets_root, args.encoder, args.dataset)
    checkpoint_path = args.checkpoint or resolve_checkpoint_path(args.assets_root, args.encoder, args.dataset)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found for encoder='{args.encoder}' dataset='{args.dataset}'. Provide --checkpoint."
        )

    datasets_dir = args.assets_root / "datasets"
    beir_source = load_beir_source(args.dataset, datasets_dir=datasets_dir, max_queries=args.max_queries)
    test_split = load_cached_split(cache_dir, "test", beir_source, encoder_spec, args.device)
    model, metadata, missing, unexpected = load_model(
        checkpoint_path=checkpoint_path,
        model_config=ModelConfig(
            input_dim=encoder_spec.embedding_dim,
            output_dim=args.output_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ),
        device=args.device,
    )
    metrics = evaluate_model(
        model=model,
        cached_split=test_split,
        device=args.device,
        feedback_k=args.feedback_k,
        ranking_k=args.ranking_k,
        k_values=_k_values(args),
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "metrics": metrics,
                "metadata": metadata,
                "missing_keys": missing,
                "unexpected_keys": unexpected,
            },
            indent=2,
        )
    )
    return 0


def _command_run(args: argparse.Namespace) -> int:
    cache_dir = args.cache_dir or (args.assets_root / f"cache_{args.encoder}_{args.dataset}")
    if not cache_dir.exists():
        cache_args = argparse.Namespace(**vars(args))
        cache_args.cache_dir = cache_dir
        _command_cache(cache_args)

    train_args = argparse.Namespace(**vars(args))
    train_args.cache_dir = cache_dir
    return _command_train(train_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate IMRNNs over cached BEIR embeddings.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_assets = subparsers.add_parser("list-assets", help="List cached embeddings and checkpoints.")
    list_assets.add_argument("--assets-root", type=Path, default=default_assets_root())
    list_assets.set_defaults(func=_command_list_assets)

    train = subparsers.add_parser("train", help="Train IMRNNs from cached embeddings.")
    _add_common_args(train)
    train.add_argument("--cache-dir", type=Path)
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--max-queries", type=int)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--num-negatives", type=int, default=20)
    train.add_argument("--output-dim", type=int, default=256)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--feedback-k", type=int, default=100)
    train.add_argument("--ranking-k", type=int, default=10)
    train.add_argument("--k", type=int, default=10)
    train.set_defaults(func=_command_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate an IMRNN checkpoint.")
    _add_common_args(evaluate)
    evaluate.add_argument("--cache-dir", type=Path)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--max-queries", type=int)
    evaluate.add_argument("--output-dim", type=int, default=256)
    evaluate.add_argument("--hidden-dim", type=int, default=128)
    evaluate.add_argument("--dropout", type=float, default=0.1)
    evaluate.add_argument("--feedback-k", type=int, default=100)
    evaluate.add_argument("--ranking-k", type=int, default=10)
    evaluate.add_argument("--k", type=int, default=10)
    evaluate.set_defaults(func=_command_evaluate)

    cache = subparsers.add_parser("cache", help="Download a BEIR dataset and cache embeddings plus negatives.")
    _add_common_args(cache)
    cache.add_argument("--cache-dir", type=Path)
    cache.add_argument("--batch-size", type=int, default=64)
    cache.add_argument("--num-negatives", type=int, default=20)
    cache.add_argument("--negative-pool", type=int, default=200)
    cache.set_defaults(func=_command_cache)

    run = subparsers.add_parser("run", help="Cache embeddings if needed, then train and evaluate IMRNNs end to end.")
    _add_common_args(run)
    run.add_argument("--cache-dir", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--max-queries", type=int)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--epochs", type=int, default=10)
    run.add_argument("--lr", type=float, default=1e-4)
    run.add_argument("--weight-decay", type=float, default=1e-5)
    run.add_argument("--num-negatives", type=int, default=20)
    run.add_argument("--negative-pool", type=int, default=200)
    run.add_argument("--output-dim", type=int, default=256)
    run.add_argument("--hidden-dim", type=int, default=128)
    run.add_argument("--dropout", type=float, default=0.1)
    run.add_argument("--feedback-k", type=int, default=100)
    run.add_argument("--ranking-k", type=int, default=10)
    run.add_argument("--k", type=int, default=10)
    run.set_defaults(func=_command_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)
