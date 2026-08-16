from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .api import cache_embeddings, evaluate, run, train
from .assets import default_assets_root, discover_cached_embeddings, discover_checkpoints, discover_repo_checkpoints
from .hub import DEFAULT_REPO_ID, download_checkpoint


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="BEIR dataset name, such as scifact.")
    group.add_argument("--dataset-path", type=Path, help="Local BEIR-format dataset directory.")
    parser.add_argument("--datasets-dir", type=Path, help="BEIR download root (default: <assets>/datasets).")


def _add_encoder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder", help="Built-in encoder alias: minilm, e5, or mpnet.")
    parser.add_argument("--encoder-model-name", help="Custom SentenceTransformers model name.")
    parser.add_argument("--embedding-dim", type=int, help="Required with --encoder-model-name.")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--passage-prefix", default="")
    parser.add_argument("--encoder-revision", help="Optional immutable revision for a custom base encoder.")


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--assets-root", type=Path, default=default_assets_root())
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--seed", type=int, default=42)


def _dataset_values(args: argparse.Namespace) -> tuple[str, Path]:
    if args.dataset_path:
        path = args.dataset_path.resolve()
        return path.name, path.parent
    return args.dataset, args.datasets_dir or (args.assets_root / "datasets")


def _encoder_label(args: argparse.Namespace) -> str:
    return args.encoder or args.encoder_model_name or "custom"


def _cache_dir(args: argparse.Namespace, dataset: str) -> Path:
    return args.cache_dir or args.assets_root / f"cache_{_encoder_label(args).replace('/', '-')}_{dataset}"


def _command_info(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "version": __version__,
                "default_hugging_face_repo": DEFAULT_REPO_ID,
                "available_checkpoint": "minilm/scifact",
                "training_recipe": {
                    "objective": "improvement-margin",
                    "improvement_margin": 0.05,
                    "hard_negatives": 63,
                    "hard_negative_method": "dense",
                    "hard_negative_pool": 100,
                    "projector_initialization": "identity",
                    "optimizer": "adam",
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-5,
                    "batch_size": 32,
                    "epochs": 30,
                    "patience": 7,
                    "seed": 42,
                },
                "supported_encoders": ["minilm", "e5", "mpnet"],
            },
            indent=2,
        )
    )
    return 0


def _command_list_assets(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "assets_root": str(args.assets_root),
                "cached_embeddings": [
                    item.__dict__ | {"path": str(item.path)} for item in discover_cached_embeddings(args.assets_root)
                ],
                "workspace_checkpoints": [
                    item.__dict__ | {"path": str(item.path)} for item in discover_checkpoints(args.assets_root)
                ],
                "repo_checkpoints": [
                    item.__dict__ | {"path": str(item.path)}
                    for item in discover_repo_checkpoints(Path(__file__).resolve().parents[2])
                ],
            },
            indent=2,
            default=str,
        )
    )
    return 0


def _command_download(args: argparse.Namespace) -> int:
    downloaded = download_checkpoint(
        encoder=args.encoder,
        dataset=args.dataset,
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.hub_cache_dir,
    )
    print(json.dumps({"checkpoint": str(downloaded.checkpoint_path), "repo_id": downloaded.repo_id}, indent=2))
    return 0


def _command_cache(args: argparse.Namespace) -> int:
    dataset, datasets_dir = _dataset_values(args)
    built = cache_embeddings(
        encoder=args.encoder,
        dataset=dataset,
        cache_dir=_cache_dir(args, dataset),
        datasets_dir=datasets_dir,
        device=args.device,
        encoder_model_name=args.encoder_model_name,
        embedding_dim=args.embedding_dim,
        query_prefix=args.query_prefix,
        passage_prefix=args.passage_prefix,
        encoder_revision=args.encoder_revision,
        batch_size=args.batch_size,
        num_negatives=args.num_negatives,
        negative_pool=args.negative_pool,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    print(json.dumps({"cache_dir": str(built), "dataset": dataset}, indent=2))
    return 0


def _training_kwargs(args: argparse.Namespace) -> dict:
    dataset, datasets_dir = _dataset_values(args)
    return {
        "encoder": args.encoder,
        "dataset": dataset,
        "cache_dir": _cache_dir(args, dataset),
        "datasets_dir": datasets_dir,
        "output_dir": args.output_dir or args.assets_root,
        "device": args.device,
        "encoder_model_name": args.encoder_model_name,
        "embedding_dim": args.embedding_dim,
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
        "encoder_revision": args.encoder_revision,
        "max_queries": args.max_queries,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_negatives": args.num_negatives,
        "feedback_k": args.candidate_k,
        "ranking_k": args.ranking_k,
        "k": args.k,
        "improvement_margin": args.improvement_margin,
        "patience": args.patience,
        "seed": args.seed,
    }


def _print_result(payload: dict) -> int:
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _command_train(args: argparse.Namespace) -> int:
    return _print_result(train(**_training_kwargs(args)))


def _command_run(args: argparse.Namespace) -> int:
    kwargs = _training_kwargs(args)
    kwargs["negative_pool"] = args.negative_pool
    return _print_result(run(**kwargs))


def _command_evaluate(args: argparse.Namespace) -> int:
    dataset, datasets_dir = _dataset_values(args)
    return _print_result(
        evaluate(
            encoder=args.encoder,
            dataset=dataset,
            cache_dir=_cache_dir(args, dataset),
            datasets_dir=datasets_dir,
            checkpoint_path=args.checkpoint,
            device=args.device,
            encoder_model_name=args.encoder_model_name,
            embedding_dim=args.embedding_dim,
            query_prefix=args.query_prefix,
            passage_prefix=args.passage_prefix,
            encoder_revision=args.encoder_revision,
            max_queries=args.max_queries,
            seed=args.seed,
            feedback_k=args.candidate_k,
            ranking_k=args.ranking_k,
            k=args.k,
        )
    )


def _configure_training_parser(parser: argparse.ArgumentParser) -> None:
    _add_dataset_args(parser)
    _add_encoder_args(parser)
    _add_runtime_args(parser)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-negatives", type=int, default=63)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--ranking-k", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--improvement-margin", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=7)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imrnns",
        description="Train, evaluate, and use IMRNN dense-retrieval adapters.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="Show package and recipe information.")
    info.set_defaults(func=_command_info)

    assets = subparsers.add_parser("list-assets", help="List local caches and checkpoints.")
    assets.add_argument("--assets-root", type=Path, default=default_assets_root())
    assets.set_defaults(func=_command_list_assets)

    download = subparsers.add_parser("download", help="Download the validated adapter checkpoint.")
    download.add_argument("--encoder", required=True, choices=["minilm"])
    download.add_argument("--dataset", required=True, choices=["scifact"])
    download.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    download.add_argument("--revision")
    download.add_argument("--hub-cache-dir", type=Path)
    download.set_defaults(func=_command_download)

    cache = subparsers.add_parser("cache", help="Encode a dataset and mine dense hard negatives.")
    _add_dataset_args(cache)
    _add_encoder_args(cache)
    _add_runtime_args(cache)
    cache.add_argument("--batch-size", type=int, default=64)
    cache.add_argument("--num-negatives", type=int, default=63)
    cache.add_argument("--negative-pool", type=int, default=100)
    cache.set_defaults(func=_command_cache)

    train_parser = subparsers.add_parser("train", help="Train an adapter from a prepared cache.")
    _configure_training_parser(train_parser)
    train_parser.set_defaults(func=_command_train)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint against its base retriever.")
    _add_dataset_args(evaluate_parser)
    _add_encoder_args(evaluate_parser)
    _add_runtime_args(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--candidate-k", type=int, default=100)
    evaluate_parser.add_argument("--ranking-k", type=int, default=10)
    evaluate_parser.add_argument("--k", type=int, default=10)
    evaluate_parser.set_defaults(func=_command_evaluate)

    run_parser = subparsers.add_parser("run", help="Cache, train, and evaluate end to end.")
    _configure_training_parser(run_parser)
    run_parser.add_argument("--negative-pool", type=int, default=100)
    run_parser.set_defaults(func=_command_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"imrnns: error: {exc}", file=sys.stderr)
        return 2
