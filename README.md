# IMRNNs

Reference implementation of **IMRNNs: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation**.

Paper:
- arXiv: https://arxiv.org/abs/2601.20084

IMRNNs is a lightweight adapter for dense retrieval. It takes embeddings from a base retriever such as **MiniLM** or **E5**, projects them into a shared space, modulates query and document representations, and reranks candidates. The repository supports:

- building BEIR caches
- training IMRNN adapters
- evaluating checkpoints at `k = 10`
- using pretrained checkpoints

## What Matters In This Repo

- `src/imrnns/`: main IMRNN implementation
- `scripts/minimal_eval.py`: minimal checkpoint evaluator
- `scripts/hf_end_to_end_demo.py`: end-to-end demo from checkpoint to final metrics
- `checkpoints/pretrained/`: released adapter-only checkpoints
- `baseline/`: DIME, Hypencoder, and Search Adaptor baselines

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## End-to-End Usage

Build a cache from a BEIR dataset:

```bash
python -m imrnns cache --encoder minilm --dataset trec-covid --device cpu
```

Train IMRNNs:

```bash
python -m imrnns train \
  --encoder minilm \
  --dataset trec-covid \
  --device cpu \
  --epochs 5 \
  --batch-size 8 \
  --k 10
```

Evaluate a checkpoint:

```bash
python -m imrnns evaluate \
  --encoder minilm \
  --dataset trec-covid \
  --device cpu \
  --checkpoint checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt \
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

Reported metrics:

- `MRR@10`
- `Recall@10`
- `NDCG@10`

## Pretrained Checkpoints

Released checkpoints are stored in adapter-only format, so they contain the learned IMRNN projection and adapter weights without bundling the full base retriever.

Examples:

- `checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt`
- `checkpoints/pretrained/e5/imrnns-e5-fiqa.pt`

## Notes

- Checkpoints are dataset-specific.
- The matching base retriever must be used with each checkpoint.
- For quick public usage, see the Hugging Face model page and the bundled demo scripts.
