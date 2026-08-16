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
  <a href="https://pypi.org/project/imrnns/"><img src="https://img.shields.io/pypi/v/imrnns.svg?logo=pypi&amp;logoColor=white" alt="PyPI version"></a>
  <a href="https://pypi.org/project/imrnns/"><img src="https://img.shields.io/pypi/pyversions/imrnns.svg?logo=python&amp;logoColor=white" alt="Supported Python versions"></a>
  <a href="https://github.com/YashSaxena21/IMRNNs/actions/workflows/test.yml"><img src="https://github.com/YashSaxena21/IMRNNs/actions/workflows/test.yml/badge.svg" alt="Build status"></a>
  <a href="https://huggingface.co/yashsaxena21/IMRNNs"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fyashsaxena21%2FIMRNNs&amp;query=downloads&amp;label=Hugging%20Face%20downloads&amp;logo=huggingface&amp;color=FFD21E" alt="Hugging Face downloads"></a>
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

Given a query and candidate documents, IMRNNs dynamically modulates both sides
of their dense embeddings before ranking the documents by cosine similarity. A
Query Adapter conditions each document representation on the query, while a
Document Adapter uses feedback from the candidate set to adapt the query
representation. The base encoder stays frozen.

IMRNNs operates inside the dense-retrieval stage. It does not add a downstream
cross-encoder stage.

## Choose your path

| I want to… | Start here |
| --- | --- |
| Rank text with the released model | [Five-minute quick start](#five-minute-quick-start) |
| Understand why a document received its score | [Explain a retrieval decision](#explain-a-retrieval-decision) |
| Use vectors from an existing retrieval system | [Rank existing embeddings](#rank-existing-embeddings) |
| Download or load a checkpoint explicitly | [Checkpoint loading](#checkpoint-loading) |
| Train and evaluate an adapter | [Training and evaluation](#training-and-evaluation) |
| Use my own BEIR-format data | [Custom datasets](#custom-datasets) |
| Look up a command or Python API | [Reference](#reference) |

## Installation

For text and embedding ranking with the released checkpoint:

```bash
python -m pip install imrnns
```

For training and BEIR evaluation:

```bash
python -m pip install "imrnns[train,eval]"
```

IMRNNs supports Python 3.10–3.12 and CPU inference. Set `device="cuda"` when a
CUDA-enabled PyTorch installation and compatible GPU are available. The first
text-ranking call downloads the pinned MiniLM encoder and the IMRNN checkpoint;
later calls use the local Hugging Face cache.

## Five-minute quick start

This walkthrough uses a genuine example from the
[BEIR SciFact corpus](https://huggingface.co/datasets/BeIR/scifact): claim
`130` and three official document titles. SciFact marks document `27768226` as
the relevant document in its
[test relevance judgments](https://huggingface.co/datasets/BeIR/scifact-qrels).

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
    device="cpu",
)

results = adapter.rank(
    query=CLAIM,
    documents=list(DOCUMENTS.values()),
    document_ids=list(DOCUMENTS),
    top_k=3,
)

for result in results:
    print(
        result.rank,
        result.document_id,
        f"base={result.base_score:.4f}",
        f"adapted={result.adapted_score:.4f}",
        f"delta={result.score_delta:+.4f}",
        sep=" | ",
    )
```

Observed with the released checkpoint on CPU:

```text
1 | 27768226 | base=0.6616 | adapted=0.6380 | delta=-0.0236
2 | 38180456 | base=0.1311 | adapted=0.1115 | delta=-0.0196
3 | 16979690 | base=0.1331 | adapted=0.1096 | delta=-0.0235
```

The gold SciFact document is ranked first. Each `RetrievalResult` contains:

| Field | Meaning |
| --- | --- |
| `rank` | Position after IMRNN modulation |
| `index` | Original position in the supplied candidate list |
| `document_id` | Optional identifier supplied by your application |
| `text` | Original document text when using `rank()` |
| `base_score` | Frozen encoder's cosine score |
| `adapted_score` | Cosine score in the modulated embedding space |
| `score_delta` | `adapted_score - base_score` for that query-document pair |

A negative `score_delta` does not mean the ranking became worse. It means that
one pair's cosine score moved downward. Ranking quality depends on the relative
ordering across all candidates; the corpus-level metrics later in this README
measure that ordering directly.

## Explain a retrieval decision

Use `explain()` with the same SciFact claim and its top document:

```python
from pathlib import Path

explanation = adapter.explain(
    query=CLAIM,
    document=DOCUMENTS["27768226"],
    top_tokens=5,
)

print("query concepts:", explanation.query_tokens)
print("document concepts:", explanation.document_tokens)
print(
    f"base={explanation.base_score:.4f}",
    f"adapted={explanation.adapted_score:.4f}",
    f"delta={explanation.score_delta:+.4f}",
)

Path("imrnns-explanation.html").write_text(
    explanation.to_html(),
    encoding="utf-8",
)
```

Observed concept labels include `bibliography`, `access`, `author`, and
`citation` for the query, and `access` and `citation` for the document.
`RetrievalExplanation` also exposes the complete query and document modulation
vectors through `query_modulation` and `document_modulation`.

The concepts are nearest encoder-vocabulary directions obtained through a
Moore–Penrose back-projection. WordPiece fragments such as `##wall` can appear,
and these labels should be interpreted as inspection aids rather than causal
natural-language rationales. `explain()` scores a single pair, whereas
`rank()` can use feedback from several candidates, so their adapted scores can
differ slightly.

## Rank existing embeddings

`rank_embeddings()` accepts NumPy arrays, PyTorch tensors, or numeric Python
sequences. The following example derives real embeddings from the same SciFact
inputs instead of using random vectors:

```python
from sentence_transformers import SentenceTransformer

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

encoder = SentenceTransformer(
    BASE_MODEL,
    revision=BASE_REVISION,
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
    device="cpu",
    load_encoder=False,
)

results = embedding_adapter.rank_embeddings(
    query_embedding=query_embedding,
    document_embeddings=document_embeddings,
    document_ids=list(DOCUMENTS),
    top_k=3,
)
```

This returns the same document order and scores as the text example because it
uses the same pinned encoder and input strings. In a production retrieval
system, replace these arrays with the vectors returned by your vector store.
The query and document dimensions must match the checkpoint's input dimension.

### Use a custom retrieval checkpoint

If you trained an adapter for another compatible encoder, load it without a
text encoder and pass vectors directly:

```python
custom_adapter = IMRNNAdapter.from_checkpoint(
    "./my-adapter.pt",
    encoder_model_name="my-organization/my-sentence-transformer",
    embedding_dim=768,
    load_encoder=False,
    device="cpu",
)

results = custom_adapter.rank_embeddings(
    query_embedding=query_vector,
    document_embeddings=document_vectors,
    document_ids=document_ids,
    top_k=10,
)
```

## Checkpoint loading

The public release currently contains one ready-to-use adapter:

| Encoder | Dataset | Dimension | Download |
| --- | --- | ---: | --- |
| MiniLM (`all-MiniLM-L6-v2`) | SciFact | 384 | [Hugging Face files](https://huggingface.co/yashsaxena21/IMRNNs/tree/main/checkpoints) |

`from_pretrained()` is the simplest loader. To download the file explicitly:

```bash
imrnns download --encoder minilm --dataset scifact
```

The equivalent Python helper is:

```python
from imrnns import download_checkpoint

downloaded = download_checkpoint(
    encoder="minilm",
    dataset="scifact",
)
print(downloaded.checkpoint_path)
```

Load a local copy directly:

```python
adapter = IMRNNAdapter.from_checkpoint(
    "./imrnns-minilm-scifact.pt",
    encoder="minilm",
    device="cpu",
)
```

For reproducible deployments, pass a Hugging Face commit hash through
`revision`. For offline execution after the files have been cached, pass
`local_files_only=True` and, if needed, the same `cache_dir` used during the
initial download.

## Training and evaluation

Install the optional dependencies before running this workflow:

```bash
python -m pip install "imrnns[train,eval]"
```

### One-command workflow

`imrnns run` downloads the BEIR dataset when necessary, builds the embedding
cache, mines dense hard negatives, trains the adapter, and evaluates the saved
checkpoint:

```bash
imrnns run \
  --encoder minilm \
  --dataset scifact \
  --datasets-dir ./datasets \
  --cache-dir ./cache/minilm-scifact \
  --output-dir ./checkpoints \
  --device cpu
```

The trained checkpoint is written to
`./checkpoints/imrnns-minilm-scifact.pt`, and the command prints a JSON report
containing the training history, base metrics, adapted metrics, metric deltas,
and the all-metrics pass indicator.

### Run each stage separately

Build a reusable embedding and hard-negative cache:

```bash
imrnns cache \
  --encoder minilm \
  --dataset scifact \
  --datasets-dir ./datasets \
  --cache-dir ./cache/minilm-scifact \
  --device cpu
```

Train from that cache:

```bash
imrnns train \
  --encoder minilm \
  --dataset scifact \
  --datasets-dir ./datasets \
  --cache-dir ./cache/minilm-scifact \
  --output-dir ./checkpoints \
  --device cpu
```

Evaluate the saved adapter against the frozen base retriever:

```bash
imrnns evaluate \
  --encoder minilm \
  --dataset scifact \
  --datasets-dir ./datasets \
  --cache-dir ./cache/minilm-scifact \
  --checkpoint ./checkpoints/imrnns-minilm-scifact.pt \
  --device cpu
```

The evaluation output includes `base_metrics`, adapted `metrics`,
`metric_delta`, and `beats_base_all_metrics`. The final field is true only when
every adapted metric is strictly greater than its corresponding base metric.

### Default training configuration

| Setting | Default |
| --- | --- |
| Projector | Same-dimensional, identity initialized |
| Hypernetwork hidden dimension | 128 |
| Dropout | 0 |
| Positive document | Highest-relevance document per query |
| Negatives | 63 dense hard negatives from the base top 100 |
| Objective | Improvement-margin loss, margin `0.05` |
| Optimizer | Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Batch size | 32 |
| Maximum epochs / patience | 30 / 7 |
| Seed | 42 |

Use `imrnns train --help` or `imrnns run --help` to inspect and override the
available hyperparameters.

## Custom datasets

IMRNNs accepts a local dataset in the standard BEIR layout:

```text
my_dataset/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    ├── train.tsv
    └── test.tsv
```

Minimal records using the same SciFact example look like this:

```json
{"_id":"27768226","title":"Open Access Increases Citation Rate","text":"Scientific abstract text goes here."}
```

```json
{"_id":"130","text":"Articles published in open access format are more likely to be cited than traditional journals."}
```

```text
query-id	corpus-id	score
130	27768226	1
```

Load and inspect a split in Python:

```python
from imrnns import IMRNNDataset

dataset = IMRNNDataset.from_beir_directory(
    "./my_dataset",
    split="test",
)
print(len(dataset.corpus), len(dataset.queries), len(dataset.qrels))
```

Use `--dataset-path ./my_dataset` instead of `--dataset scifact` with the
training commands. When official train and test qrels exist, IMRNNs reserves
15% of the training queries for validation and keeps the test set untouched. A
dataset with one qrels split uses a deterministic 70/15/15 partition.

## Encoders

The built-in encoder aliases can be used when caching and training:

| Alias | SentenceTransformers model | Dimension |
| --- | --- | ---: |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `e5` | `intfloat/e5-large-v2` | 1024 |
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` | 768 |

Only the MiniLM–SciFact adapter is distributed as a ready-to-use checkpoint.
The other aliases are available for creating your own caches and checkpoints.

For another SentenceTransformers encoder, provide its model name and dimension:

```bash
imrnns run \
  --encoder-model-name my-organization/my-model \
  --embedding-dim 768 \
  --encoder-revision COMMIT_HASH \
  --dataset-path ./my_dataset \
  --cache-dir ./cache/custom-model \
  --output-dir ./checkpoints
```

Use `--query-prefix` and `--passage-prefix` when required by the selected
encoder. Reuse the identical model, revision, prefixes, and embedding dimension
when loading its checkpoint.

## Published checkpoint evaluation

The released MiniLM–SciFact checkpoint was evaluated on the complete official
300-query SciFact test set. Base and adapted retrieval use the same top-100
candidate sets.

| Metric | What it measures | Base MiniLM | IMRNN |
| --- | --- | ---: | ---: |
| nDCG@10 | Quality and ordering of the first ten results | 0.64508 | **0.69166** |
| Recall@10 | Relevant documents found in the first ten results | 0.78333 | **0.84333** |
| MRR@10 | How early the first relevant document appears | 0.60472 | **0.64800** |

The [training study](https://github.com/YashSaxena21/IMRNNs/blob/main/TRAINING_STUDY.md)
contains dataset splits, configuration, additional datasets, and metric deltas.

## Reference

### Command-line interface

| Command | Purpose |
| --- | --- |
| `imrnns info` | Show the package version, release checkpoint, recipe, and supported encoders |
| `imrnns download` | Download the released MiniLM–SciFact checkpoint |
| `imrnns list-assets` | List cached embeddings and local or repository checkpoints |
| `imrnns cache` | Download/load data, encode it, and mine dense hard negatives |
| `imrnns train` | Train an adapter from a compatible cache |
| `imrnns evaluate` | Compare a checkpoint with its frozen base retriever |
| `imrnns run` | Execute cache, train, and evaluate stages end to end |

Run `imrnns <command> --help` for every available option.

### Public Python API

| API | Purpose |
| --- | --- |
| `IMRNNAdapter.from_pretrained()` | Load a released adapter from Hugging Face |
| `IMRNNAdapter.from_checkpoint()` | Load a local adapter checkpoint |
| `IMRNNAdapter.rank()` | Encode and rank text documents |
| `IMRNNAdapter.rank_embeddings()` | Rank existing document vectors |
| `IMRNNAdapter.explain()` | Inspect score changes, modulation vectors, and vocabulary concepts |
| `RetrievalExplanation.to_html()` | Render a dependency-free HTML explanation fragment |
| `download_checkpoint()` | Download a release checkpoint without constructing an adapter |
| `get_download_count()` | Read the public Hugging Face download count |
| `IMRNNDataset.from_beir_directory()` | Load and validate a local BEIR split |
| `cache_embeddings()` | Build a training cache programmatically |
| `train()` | Train, save, and evaluate an adapter programmatically |
| `evaluate()` | Evaluate a saved adapter programmatically |
| `run()` | Execute the complete programmatic workflow |

The package also exports `IMRNN`, `ModelConfig`, `EncoderSpec`,
`RetrievalResult`, `RetrievalExplanation`, and `TokenAttribution` for advanced
integration.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| `BEIR download support requires...` | Install `imrnns[train,eval]` |
| No released checkpoint for an encoder/dataset | Use `minilm` with `scifact`, or train a compatible checkpoint |
| Embedding-dimension error | Use vectors from the checkpoint's exact base encoder and dimension |
| Offline loading fails | Download once, then reuse the same `cache_dir` with `local_files_only=True` |
| A pair has a negative `score_delta` | Compare the full adapted ordering; a pairwise score change is not a ranking metric |
| Explanation tokens look fragmented | They are encoder-vocabulary directions and can include WordPiece fragments |
| Cache incompatibility error | Rebuild the cache with the same dataset, encoder revision, seed, and negative count |

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
