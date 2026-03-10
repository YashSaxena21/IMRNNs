# IMRNNs

Installable Python package for **IMRNNs: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation**.

Paper:
- arXiv: https://arxiv.org/abs/2601.20084

IMRNNs is a lightweight adapter for dense retrieval. It sits on top of a base retriever such as MiniLM or E5, projects embeddings into a shared 256-dimensional space, modulates query and document representations, and adapts retrieval scores over the top candidate set.

This repository provides:
- BEIR cache construction
- IMRNN training
- checkpoint evaluation with `MRR@10`, `Recall@10`, and `NDCG@10`
- pretrained adapter-only checkpoints
- Hugging Face loading helpers

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## CLI

Build caches:

```bash
python -m imrnns cache --encoder minilm --dataset trec-covid --device cpu
```

Train:

```bash
python -m imrnns train \
  --encoder minilm \
  --dataset trec-covid \
  --device cpu \
  --epochs 5 \
  --batch-size 8 \
  --k 10
```

Evaluate:

```bash
python -m imrnns evaluate \
  --encoder minilm \
  --dataset trec-covid \
  --device cpu \
  --checkpoint checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt \
  --k 10
```

Evaluate a custom retriever + custom IMRNN checkpoint:

```bash
python -m imrnns evaluate \
  --encoder-model-name my-org/my-retriever \
  --embedding-dim 768 \
  --query-prefix "query: " \
  --passage-prefix "passage: " \
  --dataset my-dataset \
  --cache-dir /path/to/my_cache \
  --datasets-dir /path/to/datasets \
  --checkpoint /path/to/my_imrnn_adapter.pt \
  --device cpu \
  --k 10
```

Run the full pipeline:

```bash
python -m imrnns run \
  --encoder minilm \
  --dataset trec-covid \
  --device cpu \
  --epochs 5 \
  --batch-size 8 \
  --k 10
```

## Python API

```python
from pathlib import Path

from imrnns import IMRNNAdapter, cache_embeddings, load_pretrained

# Load a released checkpoint from Hugging Face.
model, metadata, encoder_spec = load_pretrained(
    encoder="minilm",
    dataset="trec-covid",
    repo_id="yashsaxena21/IMRNNs",
    device="cpu",
)

# Single-query scoring with the packaged base retriever + IMRNN adapter checkpoint.
adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="trec-covid",
    repo_id="yashsaxena21/IMRNNs",
    device="cpu",
)
results = adapter.score(
    query="What is the incubation period of COVID-19?",
    documents=[
        "COVID-19 symptoms can appear 2 to 14 days after exposure.",
        "The stock market closed higher today.",
    ],
    top_k=2,
)

# Build caches locally for training and evaluation.
cache_embeddings(
    encoder="minilm",
    dataset="trec-covid",
    cache_dir=Path("cache_minilm_trec-covid"),
    datasets_dir=Path("datasets"),
    device="cpu",
)

# Custom retriever + custom IMRNN checkpoint.
custom_adapter = IMRNNAdapter.from_checkpoint(
    checkpoint_path="my_imrnn_adapter.pt",
    encoder_model_name="my-org/my-retriever",
    embedding_dim=768,
    query_prefix="query: ",
    passage_prefix="passage: ",
    device="cpu",
)
```

## Checkpoints

Released checkpoints are stored in adapter-only format. They contain the learned IMRNN projection and hypernetwork weights, while the base retriever is loaded separately by name.

Examples:
- `checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt`
- `checkpoints/pretrained/e5/imrnns-e5-fiqa.pt`

For the public checkpoint release and Hub demo, see the Hugging Face model page.

Legacy note: `BiHyperNetIR` is kept only as an internal compatibility alias for older checkpoint code paths. The public architecture name is `IMRNN`.
