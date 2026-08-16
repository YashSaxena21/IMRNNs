---
library_name: imrnns
tags:
  - dense-retrieval
  - information-retrieval
  - interpretability
  - beir
  - pytorch
pipeline_tag: sentence-similarity
license: cc-by-4.0
---

# IMRNNs

Interpretable Modular Retrieval Neural Networks rerank candidates from a frozen
dense retriever using lightweight query/document embedding modulation.

This repository provides the validated MiniLM–SciFact adapter at:

```text
checkpoints/validated/minilm/imrnns-minilm-scifact.pt
```

## Install

```bash
pip install imrnns
```

## Use

```python
from imrnns import IMRNNAdapter

adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="scifact",
    repo_id="yashsaxena21/IMRNNs",
    device="cpu",
)

results = adapter.rerank(
    query="What is scientific evidence?",
    documents=[
        "Evidence supports or refutes a scientific claim.",
        "A recipe describes how to prepare a meal.",
    ],
)

for result in results:
    print(result.rank, result.base_score, result.adapted_score, result.score_delta)
```

Use `rerank_embeddings()` to rerank NumPy arrays or PyTorch tensors without
loading the text encoder.

## SciFact test results

| Metric | Raw MiniLM | IMRNN | Delta |
| --- | ---: | ---: | ---: |
| nDCG@10 | 0.64508 | 0.69166 | +0.04658 |
| Recall@10 | 0.78333 | 0.84333 | +0.06000 |
| MRR@10 | 0.60472 | 0.64800 | +0.04328 |

Evaluation uses the complete official 300-query SciFact test set and identical
top-100 candidate sets for raw and adapted ranking.

## Training recipe

- same-dimensional identity-initialized projector;
- 128 hidden units, zero dropout;
- 63 dense hard negatives from top 100;
- improvement-margin objective with margin 0.05;
- Adam, learning rate `1e-4`, weight decay `1e-5`, batch size 32;
- 30 epochs maximum, patience 7, seed 42;
- best epoch selected on validation mean nDCG@10, Recall@10, and MRR@10.

## Citation

```bibtex
@inproceedings{saxena-etal-2026-imrnns,
  title = "{IMRNN}s: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation",
  author = "Saxena, Yash and Padia, Ankur and Gunaratna, Kalpa and Gaur, Manas",
  booktitle = "Findings of the Association for Computational Linguistics: EACL 2026",
  year = "2026",
  doi = "10.18653/v1/2026.findings-eacl.333",
  url = "https://aclanthology.org/2026.findings-eacl.333/"
}
```

## License

The code, scripts, documentation, and checkpoints are licensed under CC BY 4.0.
See `LICENSE` and `ATTRIBUTION.md` in the repository.
