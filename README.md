# IMRNNs

Reference implementation of **IMRNNs** for dense retrieval over BEIR datasets, with support for:

- `all-MiniLM-L6-v2`
- `intfloat/e5-large-v2`

The repository supports the full workflow:

1. download a BEIR dataset
2. cache dense embeddings and mined negatives
3. train the IMRNN adapters
4. evaluate retrieval quality at `k = 10`

Primary metrics:

- `MRR@10`
- `Recall@10`
- `nDCG@10`

## Paper

**IMRNNs: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation**  
arXiv: https://arxiv.org/abs/2601.20084

## Repository Layout

- `src/imrnns/model.py`: IMRNN adapter architecture
- `src/imrnns/caching.py`: BEIR download, embedding cache construction, BM25 negative mining
- `src/imrnns/training.py`: training loop
- `src/imrnns/evaluation.py`: reranking and metric computation
- `src/imrnns/cli.py`: command-line entry points
- `baseline/`: baseline comparison implementations kept separate from the main IMRNN package
- `checkpoints/pretrained`: neatly named pretrained checkpoints exposed inside the repo

The package under `src/imrnns` is the canonical implementation. Baseline code is isolated under `baseline/`.

## Checkpoints

Pretrained checkpoints from the parent `models` workspace are exposed inside this repository with clearer names:

- `checkpoints/pretrained/minilm/imrnns-minilm-<dataset>.pt`
- `checkpoints/pretrained/e5/imrnns-e5-<dataset>.pt`

Examples:

- `checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt`
- `checkpoints/pretrained/e5/imrnns-e5-fiqa.pt`

These are now real checkpoint files stored inside the repository, not symlinks to an external workspace.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

## Commands

List available caches and checkpoints:

```bash
python -m imrnns list-assets
```

Build a fresh cache from BEIR using MiniLM or E5:

```bash
python -m imrnns cache --encoder minilm --dataset trec-covid --device cpu
python -m imrnns cache --encoder e5 --dataset fiqa --device cuda
```

Train IMRNNs from a cached dataset:

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

Minimal single-checkpoint evaluation:

```bash
python scripts/minimal_eval.py \
  --checkpoint checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt \
  --encoder minilm \
  --dataset trec-covid \
  --cache-dir ../cache_mini_trec-covid \
  --datasets-dir ../datasets \
  --k 10
```

Run the end-to-end pipeline:

```bash
python -m imrnns run \
  --encoder minilm \
  --dataset trec-covid \
  --device cpu \
  --epochs 5 \
  --batch-size 8 \
  --k 10
```

By default, `evaluate` and `run` report:

- `MRR@10`
- `Recall@10`
- `NDCG@10`

## Verified Local Runs

The current environment was validated with:

```bash
python -m imrnns evaluate \
  --encoder minilm \
  --dataset trec-covid \
  --device cpu \
  --checkpoint checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt \
  --k 10

python -m imrnns train \
  --encoder minilm \
  --dataset trec-covid \
  --device cpu \
  --epochs 1 \
  --batch-size 8 \
  --output-dir artifacts \
  --k 10
```

That training run produced:

- `artifacts/imrnns-minilm-trec-covid.pt`

## Notes

- Existing workspace caches under `--assets-root` are discovered automatically. By default, the CLI uses the parent workspace of this repository.
- Fresh caches created by `cache` or `run` are written under the selected cache directory and include:
  - document embeddings
  - query embeddings
  - mined negatives
  - a cache manifest
- Pretrained checkpoint files are stored under `checkpoints/pretrained/<encoder>/`.
