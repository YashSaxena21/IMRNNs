from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .encoders import encoder_storage_key
from .model import HIDDEN_DIM, IMRNN, ModelConfig

CHECKPOINT_FORMAT = "imrnns-final-v1"


class CheckpointCompatibilityError(ValueError):
    """Raised when checkpoint tensors do not match the final architecture."""


def default_checkpoint_name(encoder: str, dataset: str) -> str:
    return f"imrnns-{encoder_storage_key(encoder)}-{dataset}.pt"


def _require_tensor(state_dict: Mapping[str, Any], key: str) -> torch.Tensor:
    value = state_dict.get(key)
    if not isinstance(value, torch.Tensor):
        raise CheckpointCompatibilityError(f"Checkpoint is missing required tensor '{key}'.")
    return value


def infer_model_config(
    state_dict: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> ModelConfig:
    """Validate the final tensor topology and infer its embedding dimension."""

    metadata = metadata or {}
    projector = _require_tensor(state_dict, "projector.weight")
    first_query_layer = _require_tensor(state_dict, "query_hypernet.hypernet.0.weight")
    last_query_layer = _require_tensor(state_dict, "query_hypernet.hypernet.6.weight")
    first_document_layer = _require_tensor(state_dict, "doc_hypernet.hypernet.0.weight")
    if projector.dim() != 2 or projector.shape[0] != projector.shape[1]:
        raise CheckpointCompatibilityError("The final projector must be a square rank-2 tensor.")

    embedding_dim = int(projector.shape[0])
    expected_first = (HIDDEN_DIM, embedding_dim)
    expected_last = (embedding_dim * 2, HIDDEN_DIM // 2)
    if tuple(first_query_layer.shape) != expected_first or tuple(first_document_layer.shape) != expected_first:
        raise CheckpointCompatibilityError("Checkpoint hypernetwork input shapes do not match the final architecture.")
    if tuple(last_query_layer.shape) != expected_last:
        raise CheckpointCompatibilityError(
            "Checkpoint hypernetwork output shape does not match the final architecture."
        )

    saved_config = metadata.get("model_config") or {}
    expected = {
        "input_dim": embedding_dim,
        "output_dim": embedding_dim,
        "hidden_dim": HIDDEN_DIM,
        "dropout": 0.0,
    }
    for field, value in expected.items():
        if field in saved_config and saved_config[field] != value:
            raise CheckpointCompatibilityError(
                f"Checkpoint metadata declares {field}={saved_config[field]}, but the final architecture requires {value}."
            )
    return ModelConfig(input_dim=embedding_dim)


def save_checkpoint(path: Path, model: IMRNN, metadata: dict[str, Any] | None = None) -> None:
    safe_metadata = _safe_metadata(dict(metadata or {}))
    safe_metadata["checkpoint_format"] = CHECKPOINT_FORMAT
    safe_metadata["model_config"] = model.config.to_dict()
    payload = {"model_state": model.state_dict(), "metadata": safe_metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _safe_metadata(value: Any) -> Any:
    """Restrict metadata to values accepted by Torch's safe weights loader."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item) for item in value]
    return str(value)


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise CheckpointCompatibilityError(f"Unsupported checkpoint format in {checkpoint_path}.")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise CheckpointCompatibilityError(f"Checkpoint metadata in {checkpoint_path} is not a mapping.")
    if metadata.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise CheckpointCompatibilityError(f"Checkpoint '{checkpoint_path}' is not an {CHECKPOINT_FORMAT} checkpoint.")
    model_state = payload["model_state"]
    if not isinstance(model_state, dict):
        raise CheckpointCompatibilityError(f"'model_state' in {checkpoint_path} is not a state dictionary.")
    return dict(model_state), dict(metadata)


def load_model(
    checkpoint_path: str | Path,
    model_config: ModelConfig | None = None,
    device: str = "cpu",
) -> tuple[IMRNN, dict[str, Any], list[str], list[str]]:
    state_dict, metadata = load_checkpoint(checkpoint_path)
    inferred_config = infer_model_config(state_dict, metadata)
    if model_config is not None and model_config != inferred_config:
        raise CheckpointCompatibilityError(
            f"Checkpoint input_dim is {inferred_config.input_dim}, but the requested model uses {model_config.input_dim}."
        )
    config = model_config or inferred_config
    model = IMRNN(config)
    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint '{checkpoint_path}' is incompatible with the final architecture: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    metadata = {
        **metadata,
        "model_config": config.to_dict(),
        "checkpoint_path": str(checkpoint_path),
    }
    return model, metadata, list(incompatible.missing_keys), list(incompatible.unexpected_keys)
