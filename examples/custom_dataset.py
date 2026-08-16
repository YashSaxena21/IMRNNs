from imrnns import IMRNNDataset

dataset = IMRNNDataset.from_beir_directory("./my_dataset", split="test")
print(len(dataset.corpus), len(dataset.queries), len(dataset.qrels))
