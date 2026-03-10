from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .beir_data import load_beir_source
from .caching import build_cache
from .checkpoints import default_checkpoint_name, load_model, save_checkpoint
from .data import ContrastiveCachedDataset, load_cached_split
from .encoders import resolve_encoder_spec
from .evaluation import evaluate_model
from .model import IMRNN, ModelConfig
from .training import TrainingConfig, train_model


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
    batch_size: int = 64,
    num_negatives: int = 20,
    negative_pool: int = 200,
    max_queries: Optional[int] = None,
) -> Path:
    encoder_spec = resolve_encoder_spec(
        encoder=encoder,
        encoder_model_name=encoder_model_name,
        embedding_dim=embedding_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
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
    max_queries: Optional[int] = None,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    num_negatives: int = 20,
    output_dim: int = 256,
    hidden_dim: int = 128,
    dropout: float = 0.1,
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
    )
    beir_source = load_beir_source(dataset, datasets_dir=datasets_dir, max_queries=max_queries)
    train_split = load_cached_split(cache_dir, "train", beir_source, encoder_spec, device)
    val_split = load_cached_split(cache_dir, "val", beir_source, encoder_spec, device)
    test_split = load_cached_split(cache_dir, "test", beir_source, encoder_spec, device)

    model = IMRNN(
        ModelConfig(
            input_dim=encoder_spec.embedding_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    )

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
        ),
        device=device,
    )
    evaluation_metrics = evaluate_model(
        model=model,
        cached_split=test_split,
        device=device,
        feedback_k=feedback_k,
        ranking_k=ranking_k,
        k_values=[k],
    )

    checkpoint_stem = encoder or encoder_spec.key
    checkpoint_path = output_dir / default_checkpoint_name(checkpoint_stem, dataset)
    metadata = {
        "encoder": checkpoint_stem,
        "encoder_model_name": encoder_spec.model_name,
        "dataset": dataset,
        "cache_dir": str(cache_dir),
        "model_config": {
            "input_dim": encoder_spec.embedding_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        },
        "training": training_metrics,
        "evaluation": evaluation_metrics,
    }
    save_checkpoint(checkpoint_path, model, metadata)
    return {
        "checkpoint": checkpoint_path,
        "training": training_metrics,
        "evaluation": evaluation_metrics,
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
    max_queries: Optional[int] = None,
    output_dim: int = 256,
    hidden_dim: int = 128,
    dropout: float = 0.1,
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
    )
    beir_source = load_beir_source(dataset, datasets_dir=datasets_dir, max_queries=max_queries)
    test_split = load_cached_split(cache_dir, "test", beir_source, encoder_spec, device)
    model, metadata, missing, unexpected = load_model(
        checkpoint_path=checkpoint_path,
        model_config=ModelConfig(
            input_dim=encoder_spec.embedding_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ),
        device=device,
    )
    metrics = evaluate_model(
        model=model,
        cached_split=test_split,
        device=device,
        feedback_k=feedback_k,
        ranking_k=ranking_k,
        k_values=[k],
    )
    return {
        "checkpoint": checkpoint_path,
        "metrics": metrics,
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
    max_queries: Optional[int] = None,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    num_negatives: int = 20,
    negative_pool: int = 200,
    output_dim: int = 256,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    feedback_k: int = 100,
    ranking_k: int = 10,
    k: int = 10,
) -> dict[str, Any]:
    if not cache_dir.exists():
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
            batch_size=batch_size,
            num_negatives=num_negatives,
            negative_pool=negative_pool,
            max_queries=max_queries,
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
        max_queries=max_queries,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        num_negatives=num_negatives,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        feedback_k=feedback_k,
        ranking_k=ranking_k,
        k=k,
    )
