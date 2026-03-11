---
library_name: imrnns
tags:
  - dense-retrieval
  - information-retrieval
  - interpretability
  - beir
  - pytorch
pipeline_tag: sentence-similarity
license: mit
---

# IMRNNs

<p align="center">
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/imrnns-given-wordmark-header.png" alt="IMRNNs" width="520" />
</p>

<p align="center">
  <strong>Interpretable Modular Retrieval Neural Networks (IMRNNs)</strong>
  <br />
  Adapter-only checkpoint release for dense retriever adaptation
</p>

<p align="center">
  <a href="https://yashsaxena21.github.io/IMRNNs-web/">Website</a> ·
  <a href="https://github.com/YashSaxena21/IMRNNs">GitHub</a> ·
  <a href="https://arxiv.org/abs/2601.20084">Paper</a>
</p>

<p align="center">
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/imrnns-given-icon-tight.png" alt="IMRNNs mark" height="54" />
  &nbsp;&nbsp;
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/hf-logo.svg" alt="Hugging Face" height="54" />
  &nbsp;&nbsp;
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/eacl2026-logo.png" alt="EACL 2026" height="54" />
</p>

This repository is the public checkpoint release for the `imrnns` Python package. IMRNNs keeps the base dense retriever frozen and learns lightweight embedding-space adapters on top of it: one modulates documents using the query, and one refines the query using retrieved documents.

## What Is Released Here

- Adapter-only IMRNN checkpoints
- The `imrnns` package source needed to load them
- Minimal evaluation and end-to-end demo scripts

For full training and cache-building workflows, use the main GitHub repository:

- `https://github.com/YashSaxena21/IMRNNs`

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Load the matching base retriever first, then apply the IMRNN adapter checkpoint:

```python
from imrnns import IMRNNAdapter

adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="webis-touche2020",
    repo_id="yashsaxena21/IMRNNs",
    device="cpu",
)

scores = adapter.score(
    query="Should social media platforms ban political advertising?",
    documents=[
        "Restricting political ads can reduce targeted misinformation.",
        "A recipe for roasted cauliflower with tahini sauce.",
        "Ad transparency archives improve auditing of campaign messaging.",
    ],
    top_k=3,
)

for item in scores:
    print(item.rank, item.score, item.text)
```

`IMRNNAdapter.from_pretrained(...)` loads the corresponding base retriever and then applies the released IMRNN adapter on top of it.

## End-to-End Evaluation

```python
from pathlib import Path

from imrnns import cache_embeddings, load_pretrained
from imrnns.beir_data import load_beir_source
from imrnns.data import load_cached_split
from imrnns.evaluation import evaluate_model

encoder = "minilm"
dataset = "webis-touche2020"
repo_id = "yashsaxena21/IMRNNs"
device = "cpu"
k = 10

model, metadata, encoder_spec = load_pretrained(
    encoder=encoder,
    dataset=dataset,
    repo_id=repo_id,
    device=device,
)

cache_dir = Path("demo_cache") / "cache_minilm_webis-touche2020"
datasets_dir = Path("datasets")

if not (cache_dir / "test" / "embeddings.pt").exists():
    cache_embeddings(
        encoder=encoder,
        dataset=dataset,
        cache_dir=cache_dir,
        datasets_dir=datasets_dir,
        device=device,
    )

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

## Your Own Retriever

If you trained IMRNNs on a different dense retriever, provide your own base model and checkpoint directly:

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

For training IMRNNs on additional retrievers or datasets, use the full GitHub implementation:

- `https://github.com/YashSaxena21/IMRNNs`

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
