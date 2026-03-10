from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hypernet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim * 2),
        )

        for layer in self.hypernet:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hyper_output = self.hypernet(embedding)
        scale = torch.sigmoid(hyper_output[:, : self.output_dim])
        bias = torch.tanh(hyper_output[:, self.output_dim :])
        return scale, bias


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    output_dim: int = 256
    hidden_dim: int = 128
    dropout: float = 0.1


class IMRNN(nn.Module):
    """
    Adapter-only IMRNN implementation over cached dense embeddings.

    The model keeps the legacy module names (`query_hypernet`, `doc_hypernet`,
    `query_norm`, `doc_norm`) so existing `bihypernet_*.pt` checkpoints can be
    loaded with key remapping and `strict=False`.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.projector = nn.Linear(config.input_dim, config.output_dim)
        self.query_hypernet = HyperNet(config.output_dim, config.hidden_dim, config.output_dim, config.dropout)
        self.doc_hypernet = HyperNet(config.output_dim, config.hidden_dim, config.output_dim, config.dropout)
        self.query_norm = nn.LayerNorm(config.output_dim)
        self.doc_norm = nn.LayerNorm(config.output_dim)

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(embeddings), p=2, dim=-1)

    def modulate_documents(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        q_scale, q_bias = self.query_hypernet(query_embeddings)
        return self.doc_norm(
            document_embeddings * q_scale.unsqueeze(1) + q_bias.unsqueeze(1)
        )

    def modulate_query(
        self,
        query_embeddings: torch.Tensor,
        modulated_documents: torch.Tensor,
    ) -> torch.Tensor:
        document_summary = modulated_documents.mean(dim=1)
        d_scale, d_bias = self.doc_hypernet(document_summary)
        return self.query_norm(query_embeddings * d_scale + d_bias)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: [batch, input_dim]
            document_embeddings: [batch, docs_per_query, input_dim]
        """
        projected_queries = self.project(query_embeddings)
        projected_documents = self.project(document_embeddings)
        modulated_documents = self.modulate_documents(projected_queries, projected_documents)
        modulated_queries = self.modulate_query(projected_queries, modulated_documents)
        scores = torch.einsum("bd,bkd->bk", F.normalize(modulated_queries, p=2, dim=-1), F.normalize(modulated_documents, p=2, dim=-1))
        return modulated_queries, modulated_documents, scores

    def encode_candidates(
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


class BiHyperNetIR(IMRNN):
    """Backward-compatible alias for legacy checkpoints and code paths."""
