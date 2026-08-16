from imrnns import IMRNNAdapter

adapter = IMRNNAdapter.from_pretrained(encoder="minilm", dataset="scifact", device="cpu")
results = adapter.rank("What is scientific evidence?", ["Evidence supports a claim.", "A cooking recipe."])
for result in results:
    print(result.rank, result.adapted_score, result.text)
