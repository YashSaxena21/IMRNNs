from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import HfApi, hf_hub_download

from .checkpoints import default_checkpoint_name, load_model
from .encoders import EncoderSpec, normalize_encoder_name, resolve_encoder_spec
from .model import IMRNN

DEFAULT_REPO_ID = "yashsaxena21/IMRNNs"
CONFIG_FILENAME = "config.json"
AVAILABLE_CHECKPOINTS = {("mini", "scifact")}


@dataclass(frozen=True)
class PretrainedCheckpoint:
    repo_id: str
    encoder: str
    dataset: str
    checkpoint_path: Path
    config: dict[str, Any]


def checkpoint_repo_path(encoder: str, dataset: str) -> str:
    normalized = normalize_encoder_name(encoder)
    key = (normalized, dataset.lower())
    if key not in AVAILABLE_CHECKPOINTS:
        raise ValueError(
            "The validated checkpoint release currently provides only encoder='minilm', dataset='scifact'."
        )
    display = "minilm" if normalized == "mini" else normalized
    return f"checkpoints/validated/{display}/{default_checkpoint_name(normalized, dataset.lower())}"


def load_repo_config(
    repo_id: str = DEFAULT_REPO_ID,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=CONFIG_FILENAME,
        repo_type="model",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
    )
    with open(config_path) as handle:
        return json.load(handle)


def download_checkpoint(
    *,
    encoder: str,
    dataset: str,
    repo_id: str = DEFAULT_REPO_ID,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    local_files_only: bool = False,
) -> PretrainedCheckpoint:
    repo_path = checkpoint_repo_path(encoder, dataset)
    config = load_repo_config(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=repo_path,
        repo_type="model",
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=local_files_only,
    )
    return PretrainedCheckpoint(
        repo_id=repo_id,
        encoder=normalize_encoder_name(encoder),
        dataset=dataset,
        checkpoint_path=Path(checkpoint_path),
        config=config,
    )


def load_pretrained(
    *,
    encoder: Optional[str] = None,
    dataset: str,
    repo_id: str = DEFAULT_REPO_ID,
    device: str = "cpu",
    encoder_model_name: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    query_prefix: str = "",
    passage_prefix: str = "",
    encoder_revision: str | None = None,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    local_files_only: bool = False,
) -> tuple[IMRNN, dict[str, Any], EncoderSpec]:
    encoder_spec = resolve_encoder_spec(
        encoder=encoder,
        encoder_model_name=encoder_model_name,
        embedding_dim=embedding_dim,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        encoder_revision=encoder_revision,
    )
    pretrained = download_checkpoint(
        encoder=encoder or encoder_spec.key,
        dataset=dataset,
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model, metadata, missing, unexpected = load_model(
        checkpoint_path=pretrained.checkpoint_path,
        device=device,
    )
    if model.config.input_dim != encoder_spec.embedding_dim:
        raise ValueError(
            f"Checkpoint expects {model.config.input_dim}-dimensional embeddings, but "
            f"encoder '{encoder_spec.model_name}' produces {encoder_spec.embedding_dim}."
        )
    metadata = {
        **metadata,
        "repo_id": repo_id,
        "downloaded_checkpoint": str(pretrained.checkpoint_path),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "hub_config": pretrained.config,
    }
    return model, metadata, encoder_spec


def get_download_count(repo_id: str = DEFAULT_REPO_ID) -> Optional[int]:
    info = HfApi().model_info(repo_id)
    return info.downloads
