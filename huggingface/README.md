---
library_name: imrnns
tags:
  - dense-retrieval
  - information-retrieval
  - beir
  - pytorch
pipeline_tag: sentence-similarity
license: mit
---

# IMRNNs

Adapter-only IMRNN checkpoints for the paper **IMRNNs: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation**.

Paper:
- arXiv: https://arxiv.org/abs/2601.20084

IMRNNs is trained on top of a base dense retriever such as MiniLM or E5. The checkpoint projects the original embedding space into a shared 256-dimensional adapter space, applies bidirectional modulation, and adapts retrieval scores over the top candidate set.

This model repo is the public checkpoint release for the `imrnns` Python package. It includes:
- adapter-only checkpoints
- the `src/imrnns` package
- minimal evaluation and end-to-end demo scripts

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Adapter Demo

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

## End-to-End Evaluation Demo

```python
from pathlib import Path

from imrnns import cache_embeddings, load_pretrained
from imrnns.beir_data import load_beir_source
from imrnns.data import load_cached_split
from imrnns.evaluation import evaluate_model

repo_id = "yashsaxena21/IMRNNs"
encoder = "minilm"
dataset = "trec-covid"
device = "cpu"
k = 10

# Step 1: load the released IMRNN checkpoint from the Hub.
model, metadata, encoder_spec = load_pretrained(
    encoder=encoder,
    dataset=dataset,
    repo_id=repo_id,
    device=device,
)

# Step 2: build the matching BEIR cache if it does not exist yet.
cache_dir = Path("demo_cache") / "cache_minilm_trec-covid"
datasets_dir = Path("datasets")
if not (cache_dir / "test" / "embeddings.pt").exists():
    cache_embeddings(
        encoder=encoder,
        dataset=dataset,
        cache_dir=cache_dir,
        datasets_dir=datasets_dir,
        device=device,
    )

# Step 3: evaluate the checkpoint and print the final top-k retrieval metrics.
source = load_beir_source(dataset, datasets_dir=datasets_dir)
cached_test = load_cached_split(cache_dir, "test", source, encoder_spec, device)
metrics = evaluate_model(
    model=model,
    cached_split=cached_test,
    device=device,
    feedback_k=100,
    ranking_k=k,
    k_values=[k],
)
print({"metrics": metrics, "metadata": metadata})
```

Reported metrics:
- `MRR@10`
- `Recall@10`
- `NDCG@10`

## Released Checkpoints

MiniLM:
- `checkpoints/pretrained/minilm/imrnns-minilm-msmarco.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-fiqa.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-hotpotqa.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-nq.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-scifact.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt`
- `checkpoints/pretrained/minilm/imrnns-minilm-webis-touche2020.pt`

E5:
- `checkpoints/pretrained/e5/imrnns-e5-msmarco.pt`
- `checkpoints/pretrained/e5/imrnns-e5-fiqa.pt`
- `checkpoints/pretrained/e5/imrnns-e5-hotpotqa.pt`
- `checkpoints/pretrained/e5/imrnns-e5-nq.pt`
- `checkpoints/pretrained/e5/imrnns-e5-scifact.pt`
- `checkpoints/pretrained/e5/imrnns-e5-webis-touche2020.pt`

For training IMRNNs on additional retrievers or datasets, use the full GitHub implementation:
- https://github.com/YashSaxena21/IMRNNs
