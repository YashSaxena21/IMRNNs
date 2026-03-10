from __future__ import annotations

from dataclasses import dataclass


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


def get_encoder_spec(name: str) -> EncoderSpec:
    key = normalize_encoder_name(name)
    if key not in ENCODER_SPECS:
        supported = ", ".join(sorted(ENCODER_SPECS))
        raise ValueError(f"Unsupported encoder '{name}'. Supported encoders: {supported}")
    return ENCODER_SPECS[key]
