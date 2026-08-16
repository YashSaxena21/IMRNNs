from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoints import CheckpointCompatibilityError, load_model
from .encoders import EncoderSpec, resolve_encoder_spec
from .explain import RetrievalExplanation, TokenAttribution
from .hub import DEFAULT_REPO_ID, load_pretrained
from .model import IMRNN


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    index: int
    document_id: str | None
    text: str | None
    base_score: float
    adapted_score: float
    score_delta: float


def _format_query(text: str, encoder_spec: EncoderSpec) -> str:
    return f"{encoder_spec.query_prefix}{text}" if encoder_spec.query_prefix else text


def _format_document(text: str, encoder_spec: EncoderSpec) -> str:
    return f"{encoder_spec.passage_prefix}{text}" if encoder_spec.passage_prefix else text


def _as_float_tensor(value: np.ndarray | torch.Tensor | Sequence[float], *, name: str) -> torch.Tensor:
    try:
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric NumPy array, PyTorch tensor, or sequence.") from exc
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.float()


class IMRNNAdapter:
    """Public inference wrapper for text and embedding-native reranking."""

    def __init__(
        self,
        *,
        model: IMRNN,
        encoder_spec: EncoderSpec | None,
        metadata: dict[str, Any],
        device: str,
        encoder: Any | None = None,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.encoder_spec = encoder_spec
        self.metadata = metadata
        self.device = device
        self._projector_pinv: torch.Tensor | None = None

    @staticmethod
    def _load_text_encoder(encoder_spec: EncoderSpec, device: str) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(encoder_spec.model_name, device=device, revision=encoder_spec.revision)

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
        encoder_revision: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        load_encoder: bool = True,
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
            encoder_revision=encoder_revision,
            revision=revision,
            cache_dir=Path(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
        )
        encoder_model = cls._load_text_encoder(encoder_spec, device) if load_encoder else None
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
        checkpoint_path: str | Path,
        *,
        encoder: str | None = None,
        encoder_model_name: str | None = None,
        embedding_dim: int | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        encoder_revision: str | None = None,
        device: str = "cpu",
        load_encoder: bool = True,
    ) -> "IMRNNAdapter":
        model, metadata, missing, unexpected = load_model(checkpoint_path, device=device)
        inferred_dim = model.config.input_dim
        encoder_name = encoder or metadata.get("encoder") or metadata.get("normalized_encoder")
        stored_model_name = encoder_model_name or metadata.get("encoder_model_name")
        encoder_spec: EncoderSpec | None = None
        if stored_model_name is not None:
            encoder_spec = resolve_encoder_spec(
                encoder=encoder_name,
                encoder_model_name=stored_model_name,
                embedding_dim=embedding_dim or inferred_dim,
                query_prefix=query_prefix or metadata.get("query_prefix", ""),
                passage_prefix=passage_prefix or metadata.get("passage_prefix", ""),
                encoder_revision=encoder_revision or metadata.get("encoder_revision"),
            )
        elif encoder_name is not None:
            encoder_spec = resolve_encoder_spec(encoder=str(encoder_name))
        elif embedding_dim is not None and embedding_dim != inferred_dim:
            raise CheckpointCompatibilityError(
                f"Checkpoint input_dim is {inferred_dim}, but embedding_dim={embedding_dim} was supplied."
            )

        if encoder_spec is not None and encoder_spec.embedding_dim != inferred_dim:
            raise CheckpointCompatibilityError(
                f"Encoder '{encoder_spec.model_name}' produces {encoder_spec.embedding_dim}-dimensional embeddings, "
                f"but the checkpoint expects {inferred_dim}."
            )
        if load_encoder and encoder_spec is None:
            raise ValueError(
                "The checkpoint does not identify its base encoder. Supply encoder='minilm'/'e5', or provide "
                "encoder_model_name and embedding_dim. Use load_encoder=False for embedding-only inference."
            )

        metadata = {
            **metadata,
            "checkpoint_path": str(checkpoint_path),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
        encoder_model = cls._load_text_encoder(encoder_spec, device) if load_encoder and encoder_spec else None
        return cls(
            model=model,
            encoder=encoder_model,
            encoder_spec=encoder_spec,
            metadata=metadata,
            device=device,
        )

    def _require_encoder(self) -> tuple[Any, EncoderSpec]:
        if self.encoder is None or self.encoder_spec is None:
            raise RuntimeError(
                "This adapter was loaded without a text encoder. Use rerank_embeddings(), or reload with an "
                "encoder configuration and load_encoder=True."
            )
        return self.encoder, self.encoder_spec

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        document_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not documents:
            return []
        encoder, encoder_spec = self._require_encoder()
        formatted_query = _format_query(query, encoder_spec)
        formatted_documents = [_format_document(document, encoder_spec) for document in documents]
        with torch.no_grad():
            query_embedding = encoder.encode(
                [formatted_query],
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
            )[0]
            document_embeddings = encoder.encode(
                formatted_documents,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
            )
        return self._rerank_embeddings(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            document_ids=document_ids,
            texts=documents,
            top_k=top_k,
        )

    def rerank_embeddings(
        self,
        query_embedding: np.ndarray | torch.Tensor | Sequence[float],
        document_embeddings: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
        *,
        document_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        return self._rerank_embeddings(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            document_ids=document_ids,
            texts=None,
            top_k=top_k,
        )

    def _rerank_embeddings(
        self,
        *,
        query_embedding: np.ndarray | torch.Tensor | Sequence[float],
        document_embeddings: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
        document_ids: Sequence[str] | None,
        texts: Sequence[str] | None,
        top_k: int | None,
    ) -> list[RetrievalResult]:
        query_tensor = _as_float_tensor(query_embedding, name="query_embedding")
        document_tensor = _as_float_tensor(document_embeddings, name="document_embeddings")
        if query_tensor.dim() == 2 and query_tensor.shape[0] == 1:
            query_tensor = query_tensor.squeeze(0)
        if query_tensor.dim() != 1:
            raise ValueError("query_embedding must have shape [embedding_dim] or [1, embedding_dim].")
        if document_tensor.dim() != 2:
            raise ValueError("document_embeddings must have shape [documents, embedding_dim].")
        count = document_tensor.shape[0]
        if count == 0:
            return []
        expected_dim = self.model.config.input_dim
        if query_tensor.shape[0] != expected_dim or document_tensor.shape[1] != expected_dim:
            raise ValueError(
                f"Checkpoint expects {expected_dim}-dimensional embeddings; received query dimension "
                f"{query_tensor.shape[0]} and document dimension {document_tensor.shape[1]}."
            )
        if document_ids is not None and len(document_ids) != count:
            raise ValueError("document_ids must contain exactly one ID per document embedding.")
        if texts is not None and len(texts) != count:
            raise ValueError("texts must contain exactly one item per document embedding.")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative or None.")
        if top_k == 0:
            return []

        query_tensor = query_tensor.to(self.device)
        document_tensor = document_tensor.to(self.device)
        with torch.no_grad():
            base_scores = F.cosine_similarity(document_tensor, query_tensor.unsqueeze(0), dim=-1)
            _, _, adapted_scores = self.model.score_candidates(query_tensor, document_tensor)

        ranked_indices = torch.argsort(adapted_scores, descending=True).tolist()
        if top_k is not None:
            ranked_indices = ranked_indices[: min(top_k, count)]
        return [
            RetrievalResult(
                rank=rank,
                index=index,
                document_id=str(document_ids[index]) if document_ids is not None else None,
                text=texts[index] if texts is not None else None,
                base_score=float(base_scores[index].item()),
                adapted_score=float(adapted_scores[index].item()),
                score_delta=float((adapted_scores[index] - base_scores[index]).item()),
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]

    def _token_embedding_table(self) -> tuple[torch.Tensor, Any]:
        encoder, _ = self._require_encoder()
        tokenizer = getattr(encoder, "tokenizer", None)
        first_module = encoder[0] if hasattr(encoder, "__getitem__") else encoder
        auto_model = getattr(first_module, "auto_model", first_module)
        getter = getattr(auto_model, "get_input_embeddings", None)
        if tokenizer is None or getter is None:
            raise RuntimeError("The configured encoder does not expose a tokenizer and input embedding table.")
        table = getter().weight.detach().to(self.device).float()
        if table.shape[1] != self.model.config.input_dim:
            raise RuntimeError(
                f"Token embedding dimension {table.shape[1]} does not match checkpoint input_dim "
                f"{self.model.config.input_dim}; Moore-Penrose back-projection is unavailable."
            )
        return table, tokenizer

    def _backproject_tokens(self, delta: torch.Tensor, top_tokens: int) -> list[TokenAttribution]:
        table, tokenizer = self._token_embedding_table()
        if self._projector_pinv is None:
            self._projector_pinv = torch.linalg.pinv(self.model.projector.weight.detach().float()).to(self.device)
        original_delta = self._projector_pinv @ delta.float()
        similarities = F.cosine_similarity(table, original_delta.unsqueeze(0), dim=1)
        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        if special_ids:
            special_index = torch.tensor(sorted(special_ids), device=similarities.device)
            similarities[special_index] = 0.0
        count = min(top_tokens, similarities.numel() - len(special_ids))
        indices = torch.topk(similarities.abs(), k=count).indices.tolist()
        tokens = tokenizer.convert_ids_to_tokens(indices)
        return [
            TokenAttribution(token=str(token), score=float(similarities[index].item()))
            for token, index in zip(tokens, indices)
        ]

    def explain(self, query: str, document: str, *, top_tokens: int = 10) -> RetrievalExplanation:
        if top_tokens <= 0:
            raise ValueError("top_tokens must be positive.")
        encoder, encoder_spec = self._require_encoder()
        with torch.no_grad():
            query_embedding = encoder.encode(
                [_format_query(query, encoder_spec)],
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
            ).float()
            document_embedding = encoder.encode(
                [_format_document(document, encoder_spec)],
                convert_to_tensor=True,
                show_progress_bar=False,
                device=self.device,
            ).float()
            modulated_query, modulated_documents, scores, details = self.model.forward_with_details(
                query_embedding, document_embedding.unsqueeze(0)
            )
            projected_query = details["projected_queries"][0]
            projected_document = details["projected_documents"][0, 0]
            query_delta = modulated_query[0] - projected_query
            document_delta = modulated_documents[0, 0] - projected_document
            base_score = float(F.cosine_similarity(query_embedding[0], document_embedding[0], dim=0).item())
            adapted_score = float(scores[0, 0].item())
            query_tokens = self._backproject_tokens(query_delta, top_tokens)
            document_tokens = self._backproject_tokens(document_delta, top_tokens)
        return RetrievalExplanation(
            top_query_tokens=query_tokens,
            top_document_tokens=document_tokens,
            base_score=base_score,
            adapted_score=adapted_score,
            score_delta=adapted_score - base_score,
            query_modulation=query_delta.detach().cpu().tolist(),
            document_modulation=document_delta.detach().cpu().tolist(),
        )
