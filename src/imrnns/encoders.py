from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    model_name: str
    embedding_dim: int
    query_prefix: str = ""
    passage_prefix: str = ""


ENCODER_SPECS = {
    "mini": EncoderSpec(
        key="mini",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
    ),
    "e5": EncoderSpec(
        key="e5",
        model_name="intfloat/e5-large-v2",
        embedding_dim=1024,
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "mpnet": EncoderSpec(
        key="mpnet",
        model_name="sentence-transformers/all-mpnet-base-v2",
        embedding_dim=768,
    ),
}


def normalize_encoder_name(name: str) -> str:
    key = name.strip().lower()
    aliases = {
        "all-minilm-l6-v2": "mini",
        "minilm": "mini",
        "mini-lm": "mini",
        "e5-large-v2": "e5",
        "intfloat/e5-large-v2": "e5",
        "all-mpnet-base-v2": "mpnet",
    }
    return aliases.get(key, key)


def encoder_storage_key(name: str) -> str:
    normalized = normalize_encoder_name(name)
    if normalized == "mini":
        return "minilm"
    return re.sub(r"[^a-z0-9._-]+", "-", normalized.lower()).strip("-")


def get_encoder_spec(name: str) -> EncoderSpec:
    key = normalize_encoder_name(name)
    if key not in ENCODER_SPECS:
        supported = ", ".join(sorted(ENCODER_SPECS))
        raise ValueError(f"Unsupported encoder '{name}'. Supported encoders: {supported}")
    return ENCODER_SPECS[key]


def resolve_encoder_spec(
    *,
    encoder: Optional[str] = None,
    encoder_model_name: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    query_prefix: str = "",
    passage_prefix: str = "",
) -> EncoderSpec:
    if encoder_model_name is not None:
        if embedding_dim is None:
            raise ValueError("embedding_dim is required when encoder_model_name is provided.")
        key = normalize_encoder_name(encoder or encoder_model_name)
        return EncoderSpec(
            key=key,
            model_name=encoder_model_name,
            embedding_dim=embedding_dim,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )

    if encoder is None:
        raise ValueError("Provide either encoder or encoder_model_name.")
    return get_encoder_spec(encoder)
