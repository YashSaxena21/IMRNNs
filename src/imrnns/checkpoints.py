from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from .encoders import normalize_encoder_name
from .model import BiHyperNetIR, ModelConfig


def default_checkpoint_name(encoder: str, dataset: str) -> str:
    normalized = normalize_encoder_name(encoder)
    display = "minilm" if normalized == "mini" else normalized
    return f"imrnns-{display}-{dataset}.pt"


def sanitize_legacy_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("e5_model.") or key.startswith("sbert."):
            continue
        mapped_key = key
        mapped_key = re.sub(r"^(e5_projector|sbert_projector)\.", "projector.", mapped_key)
        cleaned[mapped_key] = value
    return cleaned


def save_checkpoint(
    path: Path,
    model: BiHyperNetIR,
    metadata: dict[str, Any],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model_state" in payload:
        return sanitize_legacy_state_dict(payload["model_state"]), payload.get("metadata", {})
    if isinstance(payload, dict):
        return sanitize_legacy_state_dict(payload), {}
    raise TypeError(f"Unsupported checkpoint format in {path}")


def load_model(
    checkpoint_path: Path,
    model_config: ModelConfig,
    device: str,
) -> tuple[BiHyperNetIR, dict[str, Any], list[str], list[str]]:
    state_dict, metadata = load_checkpoint(checkpoint_path)
    model = BiHyperNetIR(model_config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, metadata, missing, unexpected
