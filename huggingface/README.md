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
  <a href="https://aclanthology.org/2026.findings-eacl.333/">EACL 2026 Paper</a>
</p>

<p align="center">
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/imrnns-given-icon-tight.png" alt="IMRNNs mark" height="54" />
  &nbsp;&nbsp;
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/hf-logo.svg" alt="Hugging Face" height="54" />
  &nbsp;&nbsp;
  <img src="https://yashsaxena21.github.io/IMRNNs-web/assets/eacl2026-logo.png" alt="EACL 2026" height="54" />
</p>

This repository hosts the released IMRNN checkpoints. Each checkpoint is adapter-only, and the corresponding base dense retriever is loaded separately by the `imrnns` package.

## Important Training Note

The released checkpoints are dataset-specific. They are not zero-shot checkpoints evaluated unchanged across all of BEIR.

Examples:

- `imrnns-minilm-webis-touche2020.pt` is trained for `webis-touche2020`
- `imrnns-e5-nq.pt` is trained for `nq`

If you want to train IMRNNs on another dataset or on another retriever, use the full GitHub implementation:

- `https://github.com/YashSaxena21/IMRNNs`

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## How to Get the Dataset and Embeddings

The full data-preparation flow lives in the GitHub repository. The package downloads official BEIR datasets from:

- `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<dataset>.zip`

Example:

- `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip`

To build the cache for a dataset and retriever:

```bash
python -m imrnns cache \
  --encoder minilm \
  --dataset webis-touche2020 \
  --datasets-dir /path/to/datasets \
  --cache-dir /path/to/cache_minilm_webis-touche2020 \
  --device cpu
```

That command downloads the dataset if needed, embeds the corpus and queries, and writes the training files required by IMRNNs.

## Quick Start

Load the matching base retriever and the released IMRNN adapter checkpoint:

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
