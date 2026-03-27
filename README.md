# IMRNNs

<p align="center">
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/imrnns-given-wordmark-header.png" alt="IMRNNs" width="520" />
</p>

<p align="center">
  <strong>Interpretable Modular Retrieval Neural Networks (IMRNNs)</strong>
  <br />
  Official implementation for training, evaluation, and released adapter checkpoints
</p>

<p align="center">
  <a href="https://yashsaxena21.github.io/IMRNNs-web/">Website</a> ·
  <a href="https://aclanthology.org/2026.findings-eacl.333/">EACL 2026 Paper</a> ·
  <a href="https://huggingface.co/yashsaxena21/IMRNNs">Hugging Face</a>
</p>

<p align="center">
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/imrnns-given-icon-tight.png" alt="IMRNNs mark" height="54" />
  &nbsp;&nbsp;
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/umbc-shield.png" alt="UMBC" height="54" />
  &nbsp;&nbsp;
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/kai2-logo.jpg" alt="KAI2 Lab" height="54" />
  &nbsp;&nbsp;
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/hf-logo.svg" alt="Hugging Face" height="54" />
  &nbsp;&nbsp;
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/eacl2026-logo.png" alt="EACL 2026" height="54" />
</p>

IMRNNs keeps the base dense retriever frozen and learns two lightweight MLP adapters directly in embedding space:

- the query adapter modulates document embeddings using the current query
- the document adapter refines the query using initially retrieved documents

This repository contains the installable `imrnns` package, the CLI for data preparation, training, and evaluation, baseline code, and released adapter-only checkpoints.

## Project Links

- Website: `https://yashsaxena21.github.io/IMRNNs-web/`
- Published paper: `https://aclanthology.org/2026.findings-eacl.333/`
- Hugging Face checkpoints: `https://huggingface.co/yashsaxena21/IMRNNs`

## What This Repository Includes

- `src/imrnns`: package implementation
- `checkpoints/pretrained`: released adapter-only checkpoints
- `scripts`: minimal evaluation and Hub demos
- `baseline`: baseline research scripts kept alongside the package implementation

## Training and Evaluation Protocol

This point is easy to miss, so it is explicit here:

- The released checkpoints are dataset-specific adapters.
- They are not zero-shot checkpoints evaluated unchanged across all of BEIR.
- Each checkpoint name tells you the training dataset, for example `imrnns-minilm-trec-covid.pt` or `imrnns-e5-nq.pt`.
- The current package workflow downloads the selected BEIR dataset, loads its source split, and derives `train`, `val`, and `test` query splits from that dataset before training.
- For most datasets, the source split is BEIR `test`.
- For `msmarco`, the package uses BEIR `train` as the source split because the BEIR test setup differs there.

In other words, if you train with `--dataset trec-covid`, the adapter is trained and evaluated on splits derived from the `trec-covid` BEIR dataset.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## Step-by-Step: Data, Embeddings, Training, Evaluation

### 1. Choose a BEIR dataset

The package uses the official BEIR dataset archives hosted by the BEIR project:

- base URL pattern: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<dataset>.zip`
- example: `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip`

You have two options:

1. Let IMRNNs download the dataset automatically when you run `cache`.
2. Download the zip yourself from the BEIR URL above and unpack it into your `datasets` directory.

Expected layout after download:

```text
datasets/
└── trec-covid/
```

### 2. Build the embedding cache

This step downloads the base retriever, encodes the corpus and queries, creates BEIR-derived `train`, `val`, and `test` splits, and mines BM25 negatives for training.

```bash
python -m imrnns cache \
  --encoder minilm \
  --dataset trec-covid \
  --datasets-dir /path/to/datasets \
  --cache-dir /path/to/cache_minilm_trec-covid \
  --device cpu
```

What gets written:

```text
cache_minilm_trec-covid/
├── manifest.json
├── train/
│   ├── embeddings.pt
│   ├── query_embeddings_mini.pt
│   └── negatives.json
├── val/
│   ├── embeddings.pt
│   ├── query_embeddings_mini.pt
│   └── negatives.json
└── test/
    ├── embeddings.pt
    ├── query_embeddings_mini.pt
    └── negatives.json
```

For E5, the query file is `query_embeddings_e5.pt`. The document embeddings remain in `embeddings.pt`.

### 3. Train IMRNNs

```bash
python -m imrnns train \
  --encoder minilm \
  --dataset trec-covid \
  --cache-dir /path/to/cache_minilm_trec-covid \
  --datasets-dir /path/to/datasets \
  --output-dir /path/to/artifacts \
  --device cpu \
  --k 10
```

This writes a checkpoint such as:

```text
artifacts/imrnns-minilm-trec-covid.pt
```

### 4. Evaluate a checkpoint

```bash
python -m imrnns evaluate \
  --encoder minilm \
  --dataset trec-covid \
  --cache-dir /path/to/cache_minilm_trec-covid \
  --datasets-dir /path/to/datasets \
  --checkpoint /path/to/artifacts/imrnns-minilm-trec-covid.pt \
  --device cpu \
  --k 10
```

Reported metrics:

- `MRR@10`
- `Recall@10`
- `NDCG@10`

### 5. Run the full pipeline in one command

```bash
python -m imrnns run \
  --encoder minilm \
  --dataset trec-covid \
  --datasets-dir /path/to/datasets \
  --output-dir /path/to/artifacts \
  --device cpu \
  --k 10
```

## Quick Start with Released Checkpoints

Load a released adapter on top of the matching base retriever:

```python
from imrnns import IMRNNAdapter

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
        "Transmission risk depends on exposure setting and viral load.",
    ],
    top_k=3,
)

for item in results:
    print(item.rank, item.score, item.text)
```

## Custom Retriever Support

If you trained IMRNNs on your own dense retriever, provide the base model name, embedding size, and optional query or passage prefixes explicitly:

```bash
python -m imrnns evaluate \
  --encoder-model-name my-org/my-retriever \
  --embedding-dim 768 \
  --query-prefix "query: " \
  --passage-prefix "passage: " \
  --dataset my-dataset \
  --cache-dir /path/to/cache_my_retriever \
  --datasets-dir /path/to/datasets \
  --checkpoint /path/to/my_imrnn_adapter.pt \
  --device cpu \
  --k 10
```

Equivalent Python API:

```python
from imrnns import IMRNNAdapter

adapter = IMRNNAdapter.from_checkpoint(
    checkpoint_path="my_imrnn_adapter.pt",
    encoder_model_name="my-org/my-retriever",
    embedding_dim=768,
    query_prefix="query: ",
    passage_prefix="passage: ",
    device="cpu",
)
```

## Released Checkpoints

Released checkpoints are adapter-only. The base retriever is loaded separately by name, and the checkpoint contains the learned IMRNN projection and adapter weights.

Examples:

- `checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-webis-touche2020.pt`
- `checkpoints/pretrained/e5/imrnns-e5-nq.pt`
- `checkpoints/pretrained/e5/imrnns-e5-webis-touche2020.pt`

For the public checkpoint release and model card, see:

- `https://huggingface.co/yashsaxena21/IMRNNs`

## Citation

```bibtex
@inproceedings{saxena-etal-2026-imrnns,
  title = "{IMRNN}s: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation",
  author = "Saxena, Yash and
    Padia, Ankur and
    Gunaratna, Kalpa and
    Gaur, Manas",
  booktitle = "Findings of the Association for Computational Linguistics: EACL 2026",
  month = mar,
  year = "2026",
  address = "Rabat, Morocco",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2026.findings-eacl.333/",
  doi = "10.18653/v1/2026.findings-eacl.333",
  pages = "6324--6337"
}
```

Legacy note: `BiHyperNetIR` remains only as an internal compatibility alias for older checkpoints and code paths. The public architecture name is `IMRNN`.
