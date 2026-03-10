---
library_name: pytorch
tags:
  - dense-retrieval
  - information-retrieval
  - reranking
  - beir
  - pytorch
  - retrieval-augmented-generation
pipeline_tag: sentence-similarity
license: mit
---

# IMRNNs

Pretrained **IMRNNs** checkpoints for dense retrieval with **MiniLM** and **E5** backbones across multiple **BEIR** datasets.

This model repository accompanies the paper:

**IMRNNs: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation**  
Yash Saxena, Ankur Padia, Kalpa Gunaratna, Manas Gaur  
arXiv: https://arxiv.org/abs/2601.20084

## Overview

IMRNNs is a lightweight retrieval adapter designed for dense retrieval and retrieval-augmented generation workflows. Instead of treating query and document embeddings as fixed representations, IMRNNs applies **bidirectional modulation** before ranking:

- a **query-conditioned adapter** modulates candidate document embeddings
- a **document-informed adapter** refines the query representation using retrieved evidence

The goal is to improve retrieval quality while preserving a more interpretable adaptation mechanism than black-box rerankers.

According to the paper, IMRNNs improves standard dense retrievers on BEIR with average gains of:

- `+6.35%` nDCG
- `+7.14%` Recall
- `+7.04%` MRR

## What Is In This Repository

This Hugging Face model repository contains:

- pretrained IMRNN checkpoint files
- a minimal evaluation script
- the `src/imrnns` package required to load and evaluate the checkpoints
- dependency metadata for local use

This is a **checkpoint release**, not a hosted inference model. The files here are meant to be used with the included codebase.
The released checkpoints use a compact adapter-only format: the base retriever is loaded separately, while the checkpoint stores the learned IMRNN projector and adapter weights.

## Available Checkpoints

### MiniLM Backbones

- `checkpoints/pretrained/minilm/imrnns-minilm-msmarco.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-fiqa.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-hotpotqa.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-nq.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-scifact.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-webis-touche2020.pt`

### E5 Backbones

- `checkpoints/pretrained/e5/imrnns-e5-msmarco.pt`
- `checkpoints/pretrained/e5/imrnns-e5-fiqa.pt`
- `checkpoints/pretrained/e5/imrnns-e5-hotpotqa.pt`
- `checkpoints/pretrained/e5/imrnns-e5-nq.pt`
- `checkpoints/pretrained/e5/imrnns-e5-scifact.pt`
- `checkpoints/pretrained/e5/imrnns-e5-webis-touche2020.pt`

Each checkpoint is **dataset-specific** and should be evaluated with the matching dataset and encoder family.

## Intended Use

Use these checkpoints if you want to:

- evaluate IMRNNs on BEIR-style retrieval tasks
- reproduce or inspect the adapter behavior reported in the paper
- compare IMRNNs against baseline dense retrievers or adapter methods
- build retrieval pipelines where query/document modulation is explicitly part of the ranking stage

## Not Intended Use

This repository is **not** intended for:

- direct `transformers.from_pretrained(...)` loading
- browser-based hosted inference from the Hugging Face model page
- zero-code use without the included package

## Quick Start

Clone the model repo or download its contents locally, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the minimal evaluator:

```bash
python scripts/minimal_eval.py \
  --checkpoint checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt \
  --encoder minilm \
  --dataset trec-covid \
  --cache-dir /path/to/cache_mini_trec-covid \
  --datasets-dir /path/to/datasets \
  --device cpu \
  --k 10
```

The evaluator reports:

- `MRR@10`
- `Recall@10`
- `NDCG@10`

## End-to-End Demo

If you want a single script that downloads a checkpoint from Hugging Face, uses the matching base retriever, builds the required cache for the chosen BEIR dataset, and evaluates the final IMRNN model, use:

```bash
python scripts/hf_end_to_end_demo.py \
  --repo-id yashsaxena21/IMRNNs \
  --encoder minilm \
  --dataset trec-covid \
  --device cpu \
  --k 10
```

This script will:

1. download the selected IMRNN checkpoint from Hugging Face
2. load the matching base encoder (`all-MiniLM-L6-v2` or `intfloat/e5-large-v2`)
3. build the dense embedding cache if it does not already exist locally
4. run IMRNN evaluation on the selected BEIR dataset
5. print `MRR@10`, `Recall@10`, and `NDCG@10`

This is the recommended public demo path for loading a released IMRNN checkpoint and seeing the final retrieval result end to end.

### Python Demo

The following example shows the full flow explicitly in Python:

```python
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

# Make the local IMRNN package importable from this downloaded model repo.
repo_root = Path(".").resolve()
src_root = repo_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from imrnns.beir_data import load_beir_source
from imrnns.caching import build_cache
from imrnns.checkpoints import load_model
from imrnns.data import load_cached_split
from imrnns.encoders import get_encoder_spec
from imrnns.evaluation import evaluate_model
from imrnns.model import ModelConfig


# Step 1: choose the released checkpoint you want to evaluate.
repo_id = "yashsaxena21/IMRNNs"
encoder_name = "minilm"
dataset_name = "trec-covid"
device = "cpu"
k = 10

# Step 2: download the checkpoint from Hugging Face.
checkpoint_path = hf_hub_download(
    repo_id=repo_id,
    filename="checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt",
    repo_type="model",
)

# Step 3: resolve the matching base retriever family.
# For `minilm`, this uses `all-MiniLM-L6-v2`.
# For `e5`, this uses `intfloat/e5-large-v2`.
encoder_spec = get_encoder_spec(encoder_name)

# Step 4: build the BEIR cache locally if it does not already exist.
# The cache stores:
# - document embeddings
# - query embeddings
# - mined negatives
cache_dir = Path("demo_cache") / "cache_minilm_trec-covid"
datasets_dir = Path("datasets")

if not (cache_dir / "test" / "embeddings.pt").exists():
    build_cache(
        dataset_name=dataset_name,
        encoder_spec=encoder_spec,
        cache_dir=cache_dir,
        datasets_dir=datasets_dir,
        device=device,
        batch_size=64,
        num_negatives=20,
        negative_pool=200,
    )

# Step 5: load the BEIR source split and align it with the cached split.
source = load_beir_source(dataset_name, datasets_dir=datasets_dir)
cached_test = load_cached_split(cache_dir, "test", source, encoder_spec, device)

# Step 6: load the IMRNN checkpoint on top of the matching base retriever.
model, metadata, missing, unexpected = load_model(
    checkpoint_path=Path(checkpoint_path),
    model_config=ModelConfig(input_dim=encoder_spec.embedding_dim),
    device=device,
)

# Step 7: evaluate the final IMRNN model at k = 10.
metrics = evaluate_model(
    model=model,
    cached_split=cached_test,
    device=device,
    feedback_k=100,
    ranking_k=k,
    k_values=[k],
)

print(
    json.dumps(
        {
            "checkpoint": checkpoint_path,
            "metrics": metrics,
            "metadata": metadata,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        },
        indent=2,
    )
)
```

This example is equivalent in spirit to the bundled `scripts/hf_end_to_end_demo.py` script, but presented inline for direct copy-paste use from the model card.

For smaller or faster public demos, you can also limit the number of queries:

```bash
python scripts/hf_end_to_end_demo.py \
  --repo-id yashsaxena21/IMRNNs \
  --encoder minilm \
  --dataset trec-covid \
  --device cpu \
  --k 10 \
  --max-queries 20
```

## End-to-End Usage

The accompanying implementation also supports:

- downloading a BEIR dataset
- building dense embedding caches
- mining negatives
- training IMRNN adapters
- evaluating trained checkpoints

For the full package implementation and end-to-end CLI workflow, see the companion GitHub repository:

- `YashSaxena21/IMRNNs`

If you want to train IMRNNs for additional retrievers, new datasets, or custom cache configurations, use the GitHub implementation rather than this model page bundle.

## Model Details

The released checkpoints are PyTorch state dictionaries for the custom IMRNN implementation in `src/imrnns`.
They use the `imrnns-adapter-only-v1` format.

They are compatible with:

- `src/imrnns/model.py`
- `src/imrnns/checkpoints.py`
- `scripts/minimal_eval.py`

They are **not** packaged as standard Hugging Face `transformers` model weights.
Instead, they are loaded on top of the matching base encoder family at runtime.

## Evaluation Setting

The minimal public evaluator uses:

- BEIR datasets
- cached embeddings from the matching encoder family
- top-k evaluation with `k = 10`

Reported public metrics from the evaluator are:

- `MRR@10`
- `Recall@10`
- `NDCG@10`

## Limitations

- BEIR datasets are not bundled in this model repository.
- Embedding caches are not bundled in this model repository.
- Checkpoints are tied to specific encoder and dataset combinations.
- The model page itself is not a deployable inference endpoint.

## Citation

If you use IMRNNs in research, please cite the paper:

```bibtex
@article{saxena2026imrnns,
  title={IMRNNs: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation},
  author={Saxena, Yash and Padia, Ankur and Gunaratna, Kalpa and Gaur, Manas},
  journal={arXiv preprint arXiv:2601.20084},
  year={2026}
}
```
