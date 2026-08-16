from __future__ import annotations

import html
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenAttribution:
    token: str
    score: float


@dataclass(frozen=True)
class RetrievalExplanation:
    top_query_tokens: list[TokenAttribution]
    top_document_tokens: list[TokenAttribution]
    base_score: float
    adapted_score: float
    score_delta: float
    query_modulation: list[float]
    document_modulation: list[float]
    method: str = "moore-penrose-vocabulary-backprojection"

    @property
    def query_tokens(self) -> list[str]:
        return [item.token for item in self.top_query_tokens]

    @property
    def document_tokens(self) -> list[str]:
        return [item.token for item in self.top_document_tokens]

    @property
    def query_token_scores(self) -> list[float]:
        return [item.score for item in self.top_query_tokens]

    @property
    def document_token_scores(self) -> list[float]:
        return [item.score for item in self.top_document_tokens]

    def to_html(self) -> str:
        """Render a dependency-free HTML fragment."""

        def render(items: list[TokenAttribution]) -> str:
            rows = "".join(f"<li><code>{html.escape(item.token)}</code> {item.score:+.4f}</li>" for item in items)
            return f"<ol>{rows}</ol>"

        return (
            '<section class="imrnns-explanation">'
            f"<p>Base: {self.base_score:.4f} · Adapted: {self.adapted_score:.4f} · "
            f"Delta: {self.score_delta:+.4f}</p>"
            f"<h3>Query concepts</h3>{render(self.top_query_tokens)}"
            f"<h3>Document concepts</h3>{render(self.top_document_tokens)}"
            "</section>"
        )
