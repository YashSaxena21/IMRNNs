from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIM = 128
DROPOUT = 0.0


class HyperNet(nn.Module):
    """Predict a diagonal affine transform represented by scale/bias vectors."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hypernet = nn.Sequential(
            nn.Linear(embedding_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, embedding_dim * 2),
        )
        for layer in self.hypernet:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hyper_output = self.hypernet(embedding)
        scale = torch.sigmoid(hyper_output[..., : self.embedding_dim])
        bias = torch.tanh(hyper_output[..., self.embedding_dim :])
        return scale, bias


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int

    def __post_init__(self) -> None:
        if isinstance(self.input_dim, bool) or not isinstance(self.input_dim, int) or self.input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")

    @property
    def output_dim(self) -> int:
        return self.input_dim

    @property
    def hidden_dim(self) -> int:
        return HIDDEN_DIM

    @property
    def dropout(self) -> float:
        return DROPOUT

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }


class IMRNN(nn.Module):
    """Query/document embedding modulation for dense-retrieval reranking."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dimension = config.input_dim
        self.projector = nn.Linear(dimension, dimension)
        self.query_hypernet = HyperNet(dimension)
        self.doc_hypernet = HyperNet(dimension)
        self.query_norm = nn.LayerNorm(dimension)
        self.doc_norm = nn.LayerNorm(dimension)

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected embeddings with final dimension {self.config.input_dim}, received {embeddings.shape[-1]}."
            )
        return F.normalize(self.projector(embeddings), p=2, dim=-1)

    def _modulate(
        self,
        projected_queries: torch.Tensor,
        projected_documents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        query_scale, query_bias = self.query_hypernet(projected_queries)
        modulated_documents = self.doc_norm(projected_documents * query_scale.unsqueeze(1) + query_bias.unsqueeze(1))

        document_scale, document_bias = self.doc_hypernet(projected_documents)
        aggregate_scale = document_scale.mean(dim=1)
        aggregate_bias = document_bias.mean(dim=1)
        modulated_queries = self.query_norm(projected_queries * aggregate_scale + aggregate_bias)
        details = {
            "query_scale": query_scale,
            "query_bias": query_bias,
            "document_scale": document_scale,
            "document_bias": document_bias,
            "aggregate_document_scale": aggregate_scale,
            "aggregate_document_bias": aggregate_bias,
        }
        return modulated_queries, modulated_documents, details

    def forward_with_details(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if query_embeddings.dim() != 2:
            raise ValueError("query_embeddings must have shape [batch, input_dim].")
        if document_embeddings.dim() != 3:
            raise ValueError("document_embeddings must have shape [batch, documents, input_dim].")
        if query_embeddings.shape[0] != document_embeddings.shape[0]:
            raise ValueError("Query and document batch sizes must match.")
        if document_embeddings.shape[1] == 0:
            raise ValueError("At least one document embedding is required.")

        projected_queries = self.project(query_embeddings)
        projected_documents = self.project(document_embeddings)
        modulated_queries, modulated_documents, details = self._modulate(projected_queries, projected_documents)
        normalized_queries = F.normalize(modulated_queries, p=2, dim=-1)
        normalized_documents = F.normalize(modulated_documents, p=2, dim=-1)
        scores = torch.einsum("bd,bkd->bk", normalized_queries, normalized_documents)
        details = {
            **details,
            "projected_queries": projected_queries,
            "projected_documents": projected_documents,
            "modulated_queries": modulated_queries,
            "modulated_documents": modulated_documents,
        }
        return modulated_queries, modulated_documents, scores, details

    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modulated_queries, modulated_documents, scores, _ = self.forward_with_details(
            query_embeddings, document_embeddings
        )
        return modulated_queries, modulated_documents, scores

    def score_candidates(
        self,
        query_embedding: torch.Tensor,
        candidate_document_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if candidate_document_embeddings.dim() == 2:
            candidate_document_embeddings = candidate_document_embeddings.unsqueeze(0)
        modulated_query, modulated_docs, scores = self.forward(query_embedding, candidate_document_embeddings)
        return modulated_query.squeeze(0), modulated_docs.squeeze(0), scores.squeeze(0)
