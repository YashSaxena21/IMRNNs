from pathlib import Path

from imrnns import IMRNNAdapter

adapter = IMRNNAdapter.from_pretrained(encoder="minilm", dataset="scifact")
explanation = adapter.explain(
    query="What currency is used in Mexico?",
    document="The Mexican peso is the currency of Mexico.",
)
print(explanation.top_query_tokens)
print(explanation.top_document_tokens)
print(explanation.base_score, explanation.adapted_score, explanation.score_delta)
Path("explanation.html").write_text(explanation.to_html(), encoding="utf-8")
