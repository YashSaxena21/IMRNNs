from imrnns import IMRNNAdapter

adapter = IMRNNAdapter.from_pretrained(encoder="minilm", dataset="scifact")
results = adapter.rerank(
    query="What is the incubation period of COVID-19?",
    documents=[
        "Symptoms can appear 2 to 14 days after exposure.",
        "Markets closed higher today.",
        "Transmission depends on exposure conditions.",
    ],
    document_ids=["medical", "finance", "transmission"],
)
for result in results:
    print(result)
