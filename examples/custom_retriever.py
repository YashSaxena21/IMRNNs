"""Rerank candidates returned by any external vector store."""

from imrnns import IMRNNAdapter


def rerank_vector_store_candidates(query_vector, candidates):
    adapter = IMRNNAdapter.from_checkpoint(
        "my-adapter.pt",
        encoder_model_name="my-org/my-retriever",
        embedding_dim=len(query_vector),
        load_encoder=False,
    )
    return adapter.rerank_embeddings(
        query_vector,
        [candidate["vector"] for candidate in candidates],
        document_ids=[candidate["id"] for candidate in candidates],
    )
