from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import HfApi, hf_hub_download

from .checkpoints import default_checkpoint_name, load_model
from .encoders import EncoderSpec, get_encoder_spec, normalize_encoder_name
from .model import IMRNN, ModelConfig

DEFAULT_REPO_ID = "yashsaxena21/IMRNNs"
CONFIG_FILENAME = "config.json"


@dataclass(frozen=True)
class PretrainedCheckpoint:
    repo_id: str
    encoder: str
    dataset: str
    checkpoint_path: Path
    config: dict[str, Any]


def checkpoint_repo_path(encoder: str, dataset: str) -> str:
    normalized = normalize_encoder_name(encoder)
    display = "minilm" if normalized == "mini" else normalized
    return f"checkpoints/pretrained/{display}/{default_checkpoint_name(normalized, dataset)}"


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
    checkpoint_filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    local_files_only: bool = False,
) -> PretrainedCheckpoint:
    config = load_repo_config(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_filename or checkpoint_repo_path(encoder, dataset),
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
    encoder: str,
    dataset: str,
    repo_id: str = DEFAULT_REPO_ID,
    device: str = "cpu",
    checkpoint_filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    local_files_only: bool = False,
) -> tuple[IMRNN, dict[str, Any], EncoderSpec]:
    encoder_spec = get_encoder_spec(encoder)
    pretrained = download_checkpoint(
        encoder=encoder,
        dataset=dataset,
        repo_id=repo_id,
        checkpoint_filename=checkpoint_filename,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model, metadata, missing, unexpected = load_model(
        checkpoint_path=pretrained.checkpoint_path,
        model_config=ModelConfig(input_dim=encoder_spec.embedding_dim),
        device=device,
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
