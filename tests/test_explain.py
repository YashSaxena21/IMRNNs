import math

from conftest import make_adapter


def test_explanation_schema_alignment_and_determinism():
    adapter, _ = make_adapter()
    first = adapter.explain("query", "document", top_tokens=4)
    second = adapter.explain("query", "document", top_tokens=4)
    assert len(first.top_query_tokens) == len(first.top_document_tokens) == 4
    assert first.query_tokens == second.query_tokens
    assert first.document_tokens == second.document_tokens
    assert all(math.isfinite(value) for value in first.query_token_scores + first.document_token_scores)
    assert math.isclose(first.score_delta, first.adapted_score - first.base_score)
    assert "Query concepts" in first.to_html()
