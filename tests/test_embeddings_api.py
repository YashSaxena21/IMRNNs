import numpy as np
import pytest
import torch
from conftest import make_adapter


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_rerank_embeddings_does_not_invoke_encoder(kind):
    adapter, encoder = make_adapter()
    query = np.arange(8, dtype=np.float32)
    documents = np.stack([query, -query, np.ones(8, dtype=np.float32)])
    if kind == "torch":
        query = torch.from_numpy(query)
        documents = torch.from_numpy(documents)
    results = adapter.rerank_embeddings(query, documents, document_ids=["x", "y", "z"], top_k=10)
    assert len(results) == 3
    assert encoder.calls == 0
    assert all(result.text is None for result in results)
    by_index = {result.index: result for result in results}
    assert by_index[0].base_score == pytest.approx(1.0)
    assert by_index[1].base_score == pytest.approx(-1.0)


def test_rerank_embeddings_validates_dimensions_and_ids():
    adapter, _ = make_adapter()
    with pytest.raises(ValueError, match="expects 8-dimensional"):
        adapter.rerank_embeddings(np.ones(7), np.ones((2, 7)))
    with pytest.raises(ValueError, match="exactly one ID"):
        adapter.rerank_embeddings(np.ones(8), np.ones((2, 8)), document_ids=["only-one"])
    assert adapter.rerank_embeddings(np.ones(8), np.empty((0, 8))) == []
