from pathlib import Path

from conftest import make_adapter

from imrnns import IMRNNAdapter
from imrnns.checkpoints import save_checkpoint
from imrnns.model import IMRNN, ModelConfig


def test_text_rerank_returns_structured_scores():
    adapter, encoder = make_adapter()
    results = adapter.rerank("query", ["first", "second", "third"], document_ids=["a", "b", "c"], top_k=2)
    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]
    assert {result.document_id for result in results}.issubset({"a", "b", "c"})
    assert encoder.calls == 2


def test_text_rerank_empty_does_not_encode():
    adapter, encoder = make_adapter()
    assert adapter.rerank("query", []) == []
    assert encoder.calls == 0


def test_checkpoint_preserves_custom_encoder_revision_for_embedding_only_use(tmp_path: Path):
    path = tmp_path / "custom.pt"
    save_checkpoint(
        path,
        IMRNN(ModelConfig(input_dim=8)),
        {
            "encoder_model_name": "example/retriever",
            "encoder_revision": "immutable-sha",
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
        },
    )
    adapter = IMRNNAdapter.from_checkpoint(path, load_encoder=False)
    assert adapter.encoder_spec is not None
    assert adapter.encoder_spec.revision == "immutable-sha"
    assert adapter.encoder_spec.query_prefix == "query: "
    assert adapter.encoder_spec.passage_prefix == "passage: "
