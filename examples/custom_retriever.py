"""Apply IMRNN scoring to documents from an external vector store."""

from imrnns import IMRNNAdapter


def rank_vector_store_documents(query_vector, documents):
    adapter = IMRNNAdapter.from_checkpoint(
        "my-adapter.pt",
        encoder_model_name="my-org/my-retriever",
        embedding_dim=len(query_vector),
        load_encoder=False,
    )
    return adapter.rank_embeddings(
        query_vector,
        [document["vector"] for document in documents],
        document_ids=[document["id"] for document in documents],
    )
