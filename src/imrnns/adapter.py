from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from sentence_transformers import SentenceTransformer

from .checkpoints import load_model
from .encoders import EncoderSpec
from .hub import DEFAULT_REPO_ID, load_pretrained
from .model import IMRNN, ModelConfig
from .encoders import resolve_encoder_spec


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    index: int
    text: str
    score: float


def _format_query(text: str, encoder_spec: EncoderSpec) -> str:
    return f"{encoder_spec.query_prefix}{text}" if encoder_spec.query_prefix else text


def _format_document(text: str, encoder_spec: EncoderSpec) -> str:
    return f"{encoder_spec.passage_prefix}{text}" if encoder_spec.passage_prefix else text


class IMRNNAdapter:
    """Inference wrapper for applying a pretrained IMRNN adapter to a base retriever."""

    def __init__(
        self,
        *,
        model: IMRNN,
        encoder: SentenceTransformer,
        encoder_spec: EncoderSpec,
        metadata: dict[str, Any],
        device: str,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.encoder_spec = encoder_spec
        self.metadata = metadata
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        *,
        encoder: str | None = None,
        dataset: str,
        repo_id: str = DEFAULT_REPO_ID,
        device: str = "cpu",
        encoder_model_name: str | None = None,
        embedding_dim: int | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> "IMRNNAdapter":
        model, metadata, encoder_spec = load_pretrained(
            encoder=encoder,
            dataset=dataset,
            repo_id=repo_id,
            device=device,
            encoder_model_name=encoder_model_name,
            embedding_dim=embedding_dim,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
        encoder_model = SentenceTransformer(encoder_spec.model_name, device=device)
        return cls(
            model=model,
            encoder=encoder_model,
            encoder_spec=encoder_spec,
            metadata=metadata,
            device=device,
        )

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str | Path,
        encoder: str | None = None,
        encoder_model_name: str | None = None,
        embedding_dim: int | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        device: str = "cpu",
    ) -> "IMRNNAdapter":
        encoder_spec = resolve_encoder_spec(
            encoder=encoder,
            encoder_model_name=encoder_model_name,
            embedding_dim=embedding_dim,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
        model, metadata, missing, unexpected = load_model(
            checkpoint_path=Path(checkpoint_path),
            model_config=ModelConfig(input_dim=encoder_spec.embedding_dim),
            device=device,
        )
        metadata = {
            **metadata,
            "checkpoint_path": str(checkpoint_path),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
        encoder_model = SentenceTransformer(encoder_spec.model_name, device=device)
        return cls(
            model=model,
            encoder=encoder_model,
            encoder_spec=encoder_spec,
            metadata=metadata,
            device=device,
        )

    def score(self, query: str, documents: Sequence[str], top_k: int | None = None) -> list[RetrievalResult]:
        if not documents:
            return []

        formatted_query = _format_query(query, self.encoder_spec)
        formatted_documents = [_format_document(document, self.encoder_spec) for document in documents]

        with torch.no_grad():
            query_embedding = self.encoder.encode(
                [formatted_query],
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
            )[0].to(self.device)
            document_embeddings = self.encoder.encode(
                formatted_documents,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
            ).to(self.device)
            _, _, scores = self.model.score_candidates(query_embedding, document_embeddings)

        ranked_indices = torch.argsort(scores, descending=True).tolist()
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        return [
            RetrievalResult(
                rank=rank,
                index=index,
                text=documents[index],
                score=float(scores[index].item()),
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]
