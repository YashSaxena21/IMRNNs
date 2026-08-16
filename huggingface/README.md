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
  - BeIR/scifact
license: cc-by-4.0
---

<p align="center">
  <a href="https://yashsaxena21.github.io/IMRNNs-web/">
    <img src="https://huggingface.co/yashsaxena21/IMRNNs/resolve/main/assets/brand/imrnns-icon-card.png" alt="IMRNNs" height="92">
  </a>
</p>

<h1 align="center">IMRNNs</h1>

<p align="center">
  <strong>Interpretable Modular Retrieval Neural Networks</strong><br>
  Efficient, interpretable dense retrieval through dynamic embedding modulation.
</p>

<p align="center">
  <a href="https://umbc.edu/"><img src="https://huggingface.co/yashsaxena21/IMRNNs/resolve/main/assets/brand/umbc-logo-card.png" alt="University of Maryland, Baltimore County" height="48"></a>
  &nbsp;&nbsp;
  <a href="https://kai2.umbc.edu/"><img src="https://huggingface.co/yashsaxena21/IMRNNs/resolve/main/assets/brand/kai2-logo-card.jpg" alt="KAI² Lab" height="54"></a>
  &nbsp;&nbsp;
  <a href="https://huggingface.co/yashsaxena21/IMRNNs"><img src="https://huggingface.co/yashsaxena21/IMRNNs/resolve/main/assets/brand/hugging-face-logo-card.svg" alt="Hugging Face" height="48"></a>
  &nbsp;&nbsp;
  <a href="https://2026.eacl.org/"><img src="https://huggingface.co/yashsaxena21/IMRNNs/resolve/main/assets/brand/eacl-2026-logo-card.png" alt="EACL 2026" height="48"></a>
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
  <a href="https://github.com/YashSaxena21/IMRNNs"><strong>Documentation</strong></a>
  ·
  <a href="https://pypi.org/project/imrnns/"><strong>PyPI</strong></a>
  ·
  <a href="https://yashsaxena21.github.io/IMRNNs-web/"><strong>Project website</strong></a>
  ·
  <a href="https://yashsaxena21.github.io/Portfolio/"><strong>Author portfolio</strong></a>
</p>

---

Given a query and candidate documents, IMRNNs dynamically modulates both sides
of their dense embeddings before ranking the documents by cosine similarity. A
Query Adapter conditions each document representation on the query, while a
Document Adapter uses feedback from the candidate set to adapt the query
representation. The MiniLM base encoder stays frozen.

This repository contains the ready-to-use MiniLM–SciFact adapter. Browse the
[checkpoint files](https://huggingface.co/yashsaxena21/IMRNNs/tree/main/checkpoints)
or load the adapter automatically with the `imrnns` package.

## Installation

```bash
python -m pip install imrnns
```

## Rank SciFact documents

This executable example uses claim `130` and three genuine document titles
from the [BEIR SciFact corpus](https://huggingface.co/datasets/BeIR/scifact).

```python
from imrnns import IMRNNAdapter

CLAIM = (
    "Articles published in open access format are more likely to be cited "
    "than traditional journals."
)

DOCUMENTS = {
    "27768226": "Open Access Increases Citation Rate",
    "38180456": "Short-term medical service trips: a systematic review of the evidence.",
    "16979690": (
        "Effect on the quality of peer review of blinding reviewers and asking "
        "them to sign their reports: a randomized controlled trial."
    ),
}

adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="scifact",
    repo_id="yashsaxena21/IMRNNs",
    device="cpu",
)

ranked_documents = adapter.rank(
    query=CLAIM,
    documents=list(DOCUMENTS.values()),
    document_ids=list(DOCUMENTS),
    top_k=3,
)

for item in ranked_documents:
    print(
        item.rank,
        item.document_id,
        item.base_score,
        item.adapted_score,
        item.score_delta,
    )
```

Each returned item contains its new rank, original input position, optional
document ID and text, frozen-encoder score, modulated score, and score change.

## Explain a retrieval decision

Use the same claim and document to inspect vocabulary-level concepts and the
query/document modulation vectors:

```python
from pathlib import Path

explanation = adapter.explain(
    query=CLAIM,
    document=DOCUMENTS["27768226"],
    top_tokens=5,
)

print(explanation.top_query_tokens)
print(explanation.top_document_tokens)
print(explanation.query_modulation)
print(explanation.document_modulation)

Path("imrnns-explanation.html").write_text(
    explanation.to_html(),
    encoding="utf-8",
)
```

Vocabulary concepts are inspection aids derived through a Moore–Penrose
back-projection. They can contain WordPiece fragments and should not be treated
as causal natural-language rationales.

## Rank existing embeddings

Use vectors produced by the checkpoint's pinned MiniLM encoder:

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    device="cpu",
)
query_embedding = encoder.encode(CLAIM, convert_to_numpy=True)
document_embeddings = encoder.encode(
    list(DOCUMENTS.values()),
    convert_to_numpy=True,
)

embedding_adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="scifact",
    load_encoder=False,
    device="cpu",
)

ranked_documents = embedding_adapter.rank_embeddings(
    query_embedding=query_embedding,
    document_embeddings=document_embeddings,
    document_ids=list(DOCUMENTS),
    top_k=3,
)
```

NumPy arrays, PyTorch tensors, and numeric Python sequences are accepted. The
embedding dimension and base encoder must match the adapter checkpoint.

## Model details

| Field | Value |
| --- | --- |
| Base encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Base revision | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |
| Dataset | SciFact |
| Embedding dimension | 384 |
| Framework | PyTorch |
| Python package | [`imrnns`](https://pypi.org/project/imrnns/) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

For checkpoint downloading, offline loading, vector-store integration,
training, custom datasets, custom encoders, CLI commands, and the complete
Python API, see the
[project documentation](https://github.com/YashSaxena21/IMRNNs#readme).

## Intended use and limitations

The adapter is intended for research and development involving dense retrieval,
embedding analysis, and retrieval-decision interpretability. It is specialized
for the documented base encoder and domain. Behavior can change with corpus
composition, candidate-set size, query style, input truncation, or encoder
version. Review retrieved evidence before using it in high-impact or
safety-critical systems.

## Citation

```bibtex
@inproceedings{saxena-etal-2026-imrnns,
  title = "{IMRNN}s: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation",
  author = "Saxena, Yash and Padia, Ankur and Gunaratna, Kalpa and Gaur, Manas",
  booktitle = "Findings of the Association for Computational Linguistics: EACL 2026",
  year = "2026",
  pages = "6324--6337",
  doi = "10.18653/v1/2026.findings-eacl.333",
  url = "https://aclanthology.org/2026.findings-eacl.333/"
}
```

## License

The code, scripts, documentation, and checkpoints are licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Please preserve the attribution and citation information when redistributing
this work.
