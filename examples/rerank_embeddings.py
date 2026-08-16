import numpy as np

from imrnns import IMRNNAdapter

adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="scifact",
    load_encoder=False,
)
query_embedding = np.random.default_rng(42).normal(size=384).astype("float32")
document_embeddings = np.random.default_rng(7).normal(size=(100, 384)).astype("float32")
results = adapter.rerank_embeddings(
    query_embedding,
    document_embeddings,
    document_ids=[f"doc-{index}" for index in range(100)],
    top_k=10,
)
for result in results:
    print(result.rank, result.document_id, result.base_score, result.adapted_score)
