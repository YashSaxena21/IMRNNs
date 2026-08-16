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
  <a href="https://pypi.org/project/imrnns/"><img src="https://img.shields.io/pypi/v/imrnns.svg?logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://pypi.org/project/imrnns/"><img src="https://img.shields.io/pypi/pyversions/imrnns.svg?logo=python&logoColor=white" alt="Supported Python versions"></a>
  <a href="https://github.com/YashSaxena21/IMRNNs/actions/workflows/test.yml"><img src="https://github.com/YashSaxena21/IMRNNs/actions/workflows/test.yml/badge.svg" alt="Build status"></a>
  <a href="https://github.com/YashSaxena21/IMRNNs/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-CC%20BY%204.0-blue.svg" alt="CC BY 4.0 license"></a>
  <a href="https://doi.org/10.18653/v1/2026.findings-eacl.333"><img src="https://img.shields.io/badge/EACL%202026-paper-7b1fa2.svg" alt="EACL 2026 paper"></a>
</p>

<p align="center">
  <a href="https://aclanthology.org/2026.findings-eacl.333/"><strong>Paper</strong></a>
  ·
  <a href="https://pypi.org/project/imrnns/"><strong>PyPI</strong></a>
  ·
  <a href="https://huggingface.co/yashsaxena21/IMRNNs"><strong>Model checkpoint</strong></a>
  ·
  <a href="https://yashsaxena21.github.io/IMRNNs-web/"><strong>Project website</strong></a>
  ·
  <a href="https://yashsaxena21.github.io/Portfolio/"><strong>Author portfolio</strong></a>
</p>

---

IMRNNs augments a frozen dense retriever with dynamic, bidirectional modulation
at inference time. A Query Adapter conditions document embeddings on the query,
while a Document Adapter uses corpus-level feedback to adapt the query
embedding. Documents are scored with cosine similarity in the resulting
modulated embedding space, and the base encoder remains unchanged. IMRNNs stays
within the initial dense-retrieval stage and does not add a downstream
cross-encoder stage.

The public release provides the installable Python package, command-line tools,
reproducible training and evaluation workflows, and a validated MiniLM–SciFact
checkpoint hosted on Hugging Face.

## Highlights

- **Lightweight adaptation** — train compact query and document adapters while
  keeping the base retriever frozen.
- **Drop-in retrieval adaptation** — rank text directly or pass existing NumPy and
  PyTorch embeddings.
- **Multi-level interpretability** — inspect explicit transformations,
  modulation vectors, score deltas, and vocabulary-level semantic concepts.
- **Strict evaluation** — compare raw and adapted nDCG@10, Recall@10, and
  MRR@10 over identical candidate sets.
- **Reproducible artifacts** — retain encoder revisions, data hashes, split
  metadata, training history, and evaluation results in each checkpoint.

## Installation

Install the stable package from PyPI:

```bash
python -m pip install imrnns
```

Install the optional training and BEIR evaluation dependencies:

```bash
python -m pip install "imrnns[train,eval]"
```

IMRNNs supports Python 3.10–3.12 and CPU inference. For local development, see
the [development guide](#development).

## Quick start

```python
from imrnns import IMRNNAdapter

adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="scifact",
    device="cpu",
)

results = adapter.rank(
    query="What is scientific evidence?",
    documents=[
        "Evidence is used to support or refute a scientific claim.",
        "The stock market closed higher today.",
        "A recipe describes how to prepare a meal.",
    ],
    top_k=3,
)

for result in results:
    print(
        result.rank,
        result.base_score,
        result.adapted_score,
        result.score_delta,
        result.text,
    )
```

`from_pretrained()` downloads the matching adapter checkpoint from
[Hugging Face](https://huggingface.co/yashsaxena21/IMRNNs). Each result contains
its rank, original index, optional document ID and text, base-retriever cosine
score, adapted score, and score delta.

## Validated results

The included checkpoint was selected on SciFact validation and evaluated once
on the complete official 300-query test set. Raw and adapted rankings use the
same top-100 MiniLM candidates.

| Metric | Raw MiniLM | IMRNN | Improvement |
| --- | ---: | ---: | ---: |
| nDCG@10 | 0.64508 | **0.69166** | **+0.04658** |
| Recall@10 | 0.78333 | **0.84333** | **+0.06000** |
| MRR@10 | 0.60472 | **0.64800** | **+0.04328** |

The same recipe passed all three validation metrics on NFCorpus and ArguAna.
Held-out testing passed the strict all-metrics rule on SciFact and ArguAna.
See the
[training study](https://github.com/YashSaxena21/IMRNNs/blob/main/TRAINING_STUDY.md)
for full metrics and split details.

## Rank existing embeddings

`rank_embeddings()` accepts NumPy arrays or PyTorch tensors and does not
invoke a text encoder:

```python
adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="scifact",
    load_encoder=False,
)

results = adapter.rank_embeddings(
    query_embedding=query_embedding,
    document_embeddings=document_embeddings,
    document_ids=document_ids,
    top_k=10,
)
```

## Explain a retrieval decision

```python
explanation = adapter.explain(
    query="What currency is used in Mexico?",
    document="The Mexican peso is the currency of Mexico.",
    top_tokens=10,
)

print(explanation.top_query_tokens)
print(explanation.top_document_tokens)
print(explanation.score_delta)
```

The explanation back-projects the query and document modulation vectors through
the Moore–Penrose pseudoinverse of the learned projector and reports aligned
encoder-vocabulary concepts.

## Training and evaluation

Build a shared embedding cache with dense hard negatives:

```bash
imrnns cache \
  --encoder minilm \
  --dataset scifact \
  --cache-dir ./cache/minilm-scifact \
  --device cpu
```

Train an adapter:

```bash
imrnns train \
  --encoder minilm \
  --dataset scifact \
  --cache-dir ./cache/minilm-scifact \
  --output-dir ./checkpoints
```

Evaluate it against the frozen base retriever:

```bash
imrnns evaluate \
  --encoder minilm \
  --dataset scifact \
  --datasets-dir ./datasets \
  --cache-dir ./cache/minilm-scifact \
  --checkpoint checkpoints/validated/minilm/imrnns-minilm-scifact.pt
```

The evaluation JSON contains `base_metrics`, adapted `metrics`,
`metric_delta`, and `beats_base_all_metrics`. A run passes only when every
adapted metric is greater than its corresponding base metric.

### Default training configuration

- same-dimensional, identity-initialized projector;
- 128 hidden units and zero dropout;
- one highest-relevance positive and 63 dense hard negatives from the raw
  retriever's top 100 results;
- improvement-margin objective with margin `0.05`;
- Adam with learning rate `1e-4`, weight decay `1e-5`, and batch size 32;
- up to 30 epochs with patience 7;
- best-epoch selection by validation mean nDCG@10, Recall@10, and MRR@10,
  including epoch 0 as a candidate.

## Local BEIR-format datasets

IMRNNs also accepts local datasets with the standard BEIR layout:

```text
my_dataset/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    ├── train.tsv
    └── test.tsv
```

For datasets with official train and test qrels, 15% of the official training
queries are reserved for validation and the test set remains untouched. A
dataset with one qrels split uses a deterministic 70/15/15 partition.

Runnable examples are available in the
[examples directory](https://github.com/YashSaxena21/IMRNNs/tree/main/examples).

## Command-line interface

| Command | Purpose |
| --- | --- |
| `imrnns info` | Show package, checkpoint, and training-recipe information |
| `imrnns download` | Download a released adapter checkpoint |
| `imrnns list-assets` | List supported encoders and checkpoints |
| `imrnns cache` | Prepare datasets, embeddings, and hard negatives |
| `imrnns train` | Train and validate an adapter |
| `imrnns evaluate` | Compare adapted and base-retriever metrics |
| `imrnns run` | Execute the cache, train, and evaluate pipeline |

Use `imrnns <command> --help` for complete options.

## Development

```bash
git clone https://github.com/YashSaxena21/IMRNNs.git
cd IMRNNs
python -m pip install -e ".[dev]"
pytest
ruff check src tests scripts examples
python -m build
twine check dist/*
```

Bug reports, feature proposals, and focused pull requests are welcome through
[GitHub Issues](https://github.com/YashSaxena21/IMRNNs/issues) and
[GitHub Pull Requests](https://github.com/YashSaxena21/IMRNNs/pulls).

## Citation

If IMRNNs supports your research, please cite the EACL 2026 paper:

```bibtex
@inproceedings{saxena-etal-2026-imrnns,
  title = {{IMRNN}s: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation},
  author = {Saxena, Yash and Padia, Ankur and Gunaratna, Kalpa and Gaur, Manas},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2026},
  year = {2026},
  pages = {6324--6337},
  doi = {10.18653/v1/2026.findings-eacl.333},
  url = {https://aclanthology.org/2026.findings-eacl.333/}
}
```

## Authors and acknowledgments

IMRNNs is authored by Yash Saxena, Ankur Padia, Kalpa Gunaratna, and Manas
Gaur. Visit
[Yash Saxena's portfolio](https://yashsaxena21.github.io/Portfolio/) for
additional projects and publications.

This work is associated with the
[University of Maryland, Baltimore County](https://umbc.edu/) and the
[KAI² Lab](https://kai2.umbc.edu/), and was published in the
[Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.333/).

## License

The code, scripts, documentation, and checkpoints are available under the
[Creative Commons Attribution 4.0 International License](https://github.com/YashSaxena21/IMRNNs/blob/main/LICENSE).
Attribution requirements are documented in
[ATTRIBUTION.md](https://github.com/YashSaxena21/IMRNNs/blob/main/ATTRIBUTION.md).
