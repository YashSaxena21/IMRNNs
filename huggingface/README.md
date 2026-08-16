---
library_name: imrnns
tags:
  - dense-retrieval
  - information-retrieval
  - interpretability
  - beir
  - pytorch
pipeline_tag: sentence-similarity
base_model: sentence-transformers/all-MiniLM-L6-v2
datasets:
  - allenai/scifact
license: cc-by-4.0
---

<p align="center">
  <a href="https://yashsaxena21.github.io/IMRNNs-web/">
    <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/imrnns-given-icon-tight.png" alt="IMRNNs" height="72">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://umbc.edu/">
    <img src="https://styleguide.umbc.edu/wp-content/uploads/sites/113/2019/01/UMBC-primary-logo-RGB-1024x236.png" alt="University of Maryland, Baltimore County" height="52">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://kai2.umbc.edu/">
    <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/kai2-logo.jpg" alt="KAI2 Lab" height="60">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/yashsaxena21/IMRNNs">
    <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.svg" alt="Hugging Face" height="52">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://2026.eacl.org/">
    <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/eacl2026-logo.png" alt="EACL 2026" height="60">
  </a>
</p>

<h1 align="center">IMRNNs</h1>

<p align="center">
  <strong>Interpretable Modular Retrieval Neural Networks</strong><br>
  Efficient, interpretable dense retrieval through dynamic embedding modulation.
</p>

<p align="center">
  <a href="https://huggingface.co/yashsaxena21/IMRNNs"><img src="https://visitor-badge.laobi.icu/badge?page_id=yashsaxena21.IMRNNs.huggingface&amp;left_text=model%20card%20views" alt="Model card views"></a>
  <a href="https://huggingface.co/yashsaxena21/IMRNNs"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fyashsaxena21%2FIMRNNs&amp;query=downloads&amp;label=downloads&amp;logo=huggingface&amp;color=FFD21E" alt="Hugging Face downloads"></a>
  <a href="https://pypi.org/project/imrnns/"><img src="https://img.shields.io/pypi/v/imrnns.svg?logo=pypi&amp;logoColor=white" alt="PyPI version"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/license-CC%20BY%204.0-blue.svg" alt="CC BY 4.0 license"></a>
  <a href="https://doi.org/10.18653/v1/2026.findings-eacl.333"><img src="https://img.shields.io/badge/EACL%202026-paper-7b1fa2.svg" alt="EACL 2026 paper"></a>
</p>

<p align="center">
  <a href="https://aclanthology.org/2026.findings-eacl.333/"><strong>Paper</strong></a>
  ·
  <a href="https://github.com/YashSaxena21/IMRNNs"><strong>GitHub</strong></a>
  ·
  <a href="https://pypi.org/project/imrnns/"><strong>PyPI</strong></a>
  ·
  <a href="https://yashsaxena21.github.io/IMRNNs-web/"><strong>Project website</strong></a>
  ·
  <a href="https://yashsaxena21.github.io/Portfolio/"><strong>Author portfolio</strong></a>
</p>

---

IMRNNs augments a frozen dense retriever with dynamic, bidirectional embedding
modulation. A Query Adapter conditions document embeddings on the query, while
a Document Adapter uses corpus-level feedback to adapt the query embedding.
Documents are scored with cosine similarity in the resulting embedding space,
and the base encoder remains unchanged.

This repository contains the MiniLM–SciFact adapter used automatically by the
`imrnns` package. You can also [browse the model files](https://huggingface.co/yashsaxena21/IMRNNs/tree/main/checkpoints).

## Installation

```bash
python -m pip install imrnns
```

For training and BEIR evaluation support:

```bash
python -m pip install "imrnns[train,eval]"
```

## Quick start

```python
from imrnns import IMRNNAdapter

adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="scifact",
    repo_id="yashsaxena21/IMRNNs",
    device="cpu",
)

results = adapter.rank(
    query="What is scientific evidence?",
    documents=[
        "Evidence supports or refutes a scientific claim.",
        "A recipe describes how to prepare a meal.",
    ],
)

for result in results:
    print(result.rank, result.adapted_score, result.score_delta, result.text)
```

Use `rank_embeddings()` to score and rank existing NumPy arrays or PyTorch
tensors without loading the text encoder.

## Model details

| Field | Value |
| --- | --- |
| Base encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Dataset | SciFact |
| Embedding dimension | 384 |
| Framework | PyTorch |
| Package | [`imrnns`](https://pypi.org/project/imrnns/) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

## Intended use

The adapter is intended for research and development involving dense retrieval,
embedding analysis, and retrieval-decision interpretability. Review retrieval
outputs before using them in high-impact or safety-critical systems. Performance
may vary with domain, corpus composition, query style, and base-encoder version.

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

The code, scripts, documentation, and checkpoints are licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Please preserve the attribution and citation information when redistributing
this work.
