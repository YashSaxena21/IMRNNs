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
  <a href="https://arxiv.org/abs/2601.20084">Paper</a> ·
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

IMRNNs is a dense retriever adapter that keeps the base encoder frozen and learns two lightweight MLP modules in embedding space: one modulates document embeddings using the query, and one refines the query using retrieved documents. The repository provides the full package implementation, CLI workflows, baseline code, and released adapter-only checkpoints.

## Project Links

- Website: `https://yashsaxena21.github.io/IMRNNs-web/`
- Paper: `https://arxiv.org/abs/2601.20084`
- Hugging Face release: `https://huggingface.co/yashsaxena21/IMRNNs`

## What This Repository Includes

- `imrnns` Python package under `src/imrnns`
- End-to-end CLI for cache construction, training, evaluation, and full runs
- Adapter-only pretrained checkpoints under `checkpoints/pretrained`
- Baseline scripts under `baseline`
- Minimal evaluation and Hub demo scripts under `scripts`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Load a released adapter on top of the matching base retriever:

```python
from imrnns import IMRNNAdapter

adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="webis-touche2020",
    repo_id="yashsaxena21/IMRNNs",
    device="cpu",
)

results = adapter.score(
    query="Should social media platforms ban political advertising?",
    documents=[
        "Restricting political ads can reduce targeted misinformation.",
        "A recipe for roasted cauliflower with tahini sauce.",
        "Ad transparency archives improve auditing of campaign messaging.",
    ],
    top_k=3,
)

for item in results:
    print(item.rank, item.score, item.text)
```

## CLI Workflow

Build a BEIR cache:

```bash
python -m imrnns cache \
  --encoder minilm \
  --dataset webis-touche2020 \
  --datasets-dir /path/to/datasets \
  --output-dir /path/to/cache_minilm_webis-touche2020
```

Train IMRNNs:

```bash
python -m imrnns train \
  --encoder minilm \
  --dataset webis-touche2020 \
  --cache-dir /path/to/cache_minilm_webis-touche2020 \
  --datasets-dir /path/to/datasets \
  --output-dir /path/to/artifacts \
  --k 10
```

Evaluate a checkpoint:

```bash
python -m imrnns evaluate \
  --encoder minilm \
  --dataset webis-touche2020 \
  --cache-dir /path/to/cache_minilm_webis-touche2020 \
  --datasets-dir /path/to/datasets \
  --checkpoint checkpoints/pretrained/minilm/imrnns-minilm-webis-touche2020.pt \
  --device cpu \
  --k 10
```

Run the end-to-end flow:

```bash
python -m imrnns run \
  --encoder minilm \
  --dataset webis-touche2020 \
  --datasets-dir /path/to/datasets \
  --output-dir /path/to/artifacts \
  --device cpu \
  --k 10
```

## Custom Retriever Support

IMRNNs can also be trained or evaluated on top of a custom dense retriever by specifying the base model name, embedding size, and optional query or passage prefixes.

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

The same path is available through the Python API:

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

- `checkpoints/pretrained/minilm/imrnns-minilm-webis-touche2020.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt`
- `checkpoints/pretrained/e5/imrnns-e5-nq.pt`
- `checkpoints/pretrained/e5/imrnns-e5-webis-touche2020.pt`

For the public checkpoint release and model card, see Hugging Face:

- `https://huggingface.co/yashsaxena21/IMRNNs`

## Citation

```bibtex
@misc{saxena2026imrnns,
  title={IMRNNs: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation},
  author={Yash Saxena and Ankur Padia and Kalpa Gunaratna and Manas Gaur},
  year={2026},
  eprint={2601.20084},
  archivePrefix={arXiv},
  note={Accepted to EACL 2026}
}
```

Legacy note: `BiHyperNetIR` remains only as an internal compatibility alias for older checkpoints and code paths. The public architecture name is `IMRNN`.
