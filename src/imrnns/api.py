from __future__ import annotations

import hashlib
import json
import subprocess
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Optional

import torch

from .beir_data import load_beir_splits
from .caching import build_cache
from .checkpoints import default_checkpoint_name, load_model, save_checkpoint
from .data import ContrastiveCachedDataset, load_cached_split
from .encoders import encoder_storage_key, resolve_encoder_spec
from .evaluation import evaluate_model, evaluate_model_with_baseline
from .model import IMRNN, ModelConfig
from .training import TrainingConfig, initialize_projector, set_seed, train_model


def _dependency_version(distribution: str) -> str | None:
    try:
        return package_metadata.version(distribution)
    except package_metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _load_cache_manifest(cache_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError(f"Cache manifest must contain a JSON object: {path}")
    return manifest, hashlib.sha256(raw).hexdigest()


def _validate_cache(
    cache_dir: Path,
    *,
    dataset: str,
    model_name: str,
    model_revision: str | None,
    max_queries: int | None,
    seed: int,
    num_negatives: int | None = None,
) -> dict[str, Any]:
    manifest, _ = _load_cache_manifest(cache_dir)
    if manifest is None:
        raise ValueError(f"Cache manifest is missing: {cache_dir / 'manifest.json'}")
    expected = {
        "schema_version": 2,
        "dataset": dataset,
        "model_name": model_name,
        "model_revision": model_revision,
        "negative_method": "dense",
        "max_queries": max_queries,
        "seed": seed,
    }
    mismatches = {
        key: {"expected": value, "found": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if num_negatives is not None:
        cached_negatives = int(manifest.get("num_negatives", 0))
        if cached_negatives < num_negatives:
            mismatches["num_negatives"] = {"expected_at_least": num_negatives, "found": cached_negatives}
    if mismatches:
        raise ValueError(f"Training cache is incompatible with this run: {mismatches}")
    return manifest


def cache_embeddings(
    *,
    encoder: Optional[str],
    dataset: str,
    cache_dir: Path,
    datasets_dir: Path,
    device: str = "cpu",
    encoder_model_name: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    query_prefix: str = "",
    passage_prefix: str = "",
    encoder_revision: str | None = None,
    batch_size: int = 64,
    num_negatives: int = 63,
    negative_pool: int = 100,
    max_queries: Optional[int] = None,
    seed: int = 42,
) -> Path:
    encoder_spec = resolve_encoder_spec(
        encoder=encoder,
        encoder_model_name=encoder_model_name,
        embedding_dim=embedding_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        encoder_revision=encoder_revision,
    )
    return build_cache(
        dataset_name=dataset,
        encoder_spec=encoder_spec,
        cache_dir=cache_dir,
        datasets_dir=datasets_dir,
        device=device,
        batch_size=batch_size,
        num_negatives=num_negatives,
        negative_pool=negative_pool,
        max_queries=max_queries,
        seed=seed,
    )


def train(
    *,
    encoder: Optional[str],
    dataset: str,
    cache_dir: Path,
    datasets_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    encoder_model_name: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    query_prefix: str = "",
    passage_prefix: str = "",
    encoder_revision: str | None = None,
    max_queries: Optional[int] = None,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    num_negatives: int = 63,
    feedback_k: int = 100,
    ranking_k: int = 10,
    k: int = 10,
    improvement_margin: float = 0.05,
    patience: int = 7,
    seed: int = 42,
) -> dict[str, Any]:
    encoder_spec = resolve_encoder_spec(
        encoder=encoder,
        encoder_model_name=encoder_model_name,
        embedding_dim=embedding_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        encoder_revision=encoder_revision,
    )
    _validate_cache(
        cache_dir,
        dataset=dataset,
        model_name=encoder_spec.model_name,
        model_revision=encoder_spec.revision,
        max_queries=max_queries,
        seed=seed,
        num_negatives=num_negatives,
    )
    sources = load_beir_splits(dataset, datasets_dir=datasets_dir, max_queries=max_queries, seed=seed)
    train_split = load_cached_split(cache_dir, "train", sources["train"], encoder_spec, device)
    val_split = load_cached_split(cache_dir, "validation", sources["validation"], encoder_spec, device)
    test_split = load_cached_split(cache_dir, "test", sources["test"], encoder_spec, device)

    set_seed(seed)
    model = IMRNN(ModelConfig(input_dim=encoder_spec.embedding_dim))
    initialize_projector(model)

    train_dataset = ContrastiveCachedDataset(train_split, num_negatives)
    val_dataset = ContrastiveCachedDataset(val_split, num_negatives)
    if len(train_dataset) == 0:
        raise ValueError("No training examples were constructed from the cached training split.")
    if len(val_dataset) == 0:
        raise ValueError("No validation examples were constructed from the cached validation split.")

    training_metrics = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=TrainingConfig(
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            num_negatives=num_negatives,
            improvement_margin=improvement_margin,
            patience=patience,
            seed=seed,
        ),
        device=device,
        validation_metric_fn=lambda current_model: (
            sum(
                evaluate_model(
                    model=current_model,
                    cached_split=val_split,
                    device=device,
                    feedback_k=feedback_k,
                    ranking_k=ranking_k,
                    k_values=[k],
                ).values()
            )
            / 3.0
        ),
        validation_metric_name=f"mean(NDCG@{k},Recall@{k},MRR@{k})",
    )
    evaluation_metrics, base_metrics = evaluate_model_with_baseline(
        model=model,
        cached_split=test_split,
        device=device,
        feedback_k=feedback_k,
        ranking_k=ranking_k,
        k_values=[k],
    )
    evaluation_delta = {name: evaluation_metrics[name] - base_metrics[name] for name in evaluation_metrics}
    beats_base = bool(evaluation_delta) and all(delta > 0 for delta in evaluation_delta.values())

    checkpoint_stem = encoder_storage_key(encoder or encoder_spec.key)
    checkpoint_path = output_dir / default_checkpoint_name(checkpoint_stem, dataset)
    cache_manifest, cache_manifest_sha256 = _load_cache_manifest(cache_dir)
    metadata = {
        "encoder": checkpoint_stem,
        "encoder_model_name": encoder_spec.model_name,
        "encoder_revision": encoder_spec.revision,
        "query_prefix": encoder_spec.query_prefix,
        "passage_prefix": encoder_spec.passage_prefix,
        "dataset": dataset,
        "cache_dir": str(cache_dir),
        "model_config": {
            "input_dim": encoder_spec.embedding_dim,
            "output_dim": model.config.output_dim,
            "hidden_dim": model.config.hidden_dim,
            "dropout": model.config.dropout,
        },
        "training": training_metrics,
        "evaluation": evaluation_metrics,
        "base_evaluation": base_metrics,
        "evaluation_delta": evaluation_delta,
        "beats_base_all_metrics": beats_base,
        "seed": seed,
        "objective": "improvement-margin",
        "optimizer": "adam",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "improvement_margin": improvement_margin,
        "projector_initialization": "identity",
        "number_of_negatives": num_negatives,
        "negative_retrieval_method": cache_manifest.get("negative_method") if cache_manifest else None,
        "negative_pool": cache_manifest.get("negative_pool") if cache_manifest else None,
        "candidate_k": feedback_k,
        "feedback_k": feedback_k,
        "ranking_k": ranking_k,
        "base_encoder": encoder_spec.model_name,
        "base_encoder_revision": encoder_spec.revision,
        "dataset_manifest": cache_manifest,
        "dataset_manifest_sha256": cache_manifest_sha256,
        "package_version": _dependency_version("imrnns"),
        "git_commit": _git_commit(),
        "environment": {
            "torch": str(torch.__version__),
            "sentence-transformers": _dependency_version("sentence-transformers"),
            "transformers": _dependency_version("transformers"),
        },
    }
    save_checkpoint(checkpoint_path, model, metadata)
    return {
        "checkpoint": checkpoint_path,
        "training": training_metrics,
        "evaluation": evaluation_metrics,
        "base_evaluation": base_metrics,
        "evaluation_delta": evaluation_delta,
        "beats_base_all_metrics": beats_base,
        "metadata": metadata,
    }


def evaluate(
    *,
    encoder: Optional[str],
    dataset: str,
    cache_dir: Path,
    datasets_dir: Path,
    checkpoint_path: Path,
    device: str = "cpu",
    encoder_model_name: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    query_prefix: str = "",
    passage_prefix: str = "",
    encoder_revision: str | None = None,
    max_queries: Optional[int] = None,
    seed: int = 42,
    feedback_k: int = 100,
    ranking_k: int = 10,
    k: int = 10,
) -> dict[str, Any]:
    encoder_spec = resolve_encoder_spec(
        encoder=encoder,
        encoder_model_name=encoder_model_name,
        embedding_dim=embedding_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        encoder_revision=encoder_revision,
    )
    _validate_cache(
        cache_dir,
        dataset=dataset,
        model_name=encoder_spec.model_name,
        model_revision=encoder_spec.revision,
        max_queries=max_queries,
        seed=seed,
    )
    sources = load_beir_splits(dataset, datasets_dir=datasets_dir, max_queries=max_queries, seed=seed)
    test_split = load_cached_split(cache_dir, "test", sources["test"], encoder_spec, device)
    model, metadata, missing, unexpected = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    if model.config.input_dim != encoder_spec.embedding_dim:
        raise ValueError(
            f"Checkpoint expects {model.config.input_dim}-dimensional embeddings; encoder produces "
            f"{encoder_spec.embedding_dim}."
        )
    metrics, base_metrics = evaluate_model_with_baseline(
        model=model,
        cached_split=test_split,
        device=device,
        feedback_k=feedback_k,
        ranking_k=ranking_k,
        k_values=[k],
    )
    metric_delta = {name: metrics[name] - base_metrics[name] for name in metrics}
    return {
        "checkpoint": checkpoint_path,
        "metrics": metrics,
        "base_metrics": base_metrics,
        "metric_delta": metric_delta,
        "beats_base_all_metrics": bool(metric_delta) and all(delta > 0 for delta in metric_delta.values()),
        "metadata": metadata,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def run(
    *,
    encoder: Optional[str],
    dataset: str,
    cache_dir: Path,
    datasets_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    encoder_model_name: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    query_prefix: str = "",
    passage_prefix: str = "",
    encoder_revision: str | None = None,
    max_queries: Optional[int] = None,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    num_negatives: int = 63,
    negative_pool: int = 100,
    feedback_k: int = 100,
    ranking_k: int = 10,
    k: int = 10,
    improvement_margin: float = 0.05,
    patience: int = 7,
    seed: int = 42,
) -> dict[str, Any]:
    cache_embeddings(
        encoder=encoder,
        dataset=dataset,
        cache_dir=cache_dir,
        datasets_dir=datasets_dir,
        device=device,
        encoder_model_name=encoder_model_name,
        embedding_dim=embedding_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        encoder_revision=encoder_revision,
        batch_size=batch_size,
        num_negatives=num_negatives,
        negative_pool=negative_pool,
        max_queries=max_queries,
        seed=seed,
    )
    return train(
        encoder=encoder,
        dataset=dataset,
        cache_dir=cache_dir,
        datasets_dir=datasets_dir,
        output_dir=output_dir,
        device=device,
        encoder_model_name=encoder_model_name,
        embedding_dim=embedding_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        encoder_revision=encoder_revision,
        max_queries=max_queries,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        num_negatives=num_negatives,
        feedback_k=feedback_k,
        ranking_k=ranking_k,
        k=k,
        improvement_margin=improvement_margin,
        patience=patience,
        seed=seed,
    )
