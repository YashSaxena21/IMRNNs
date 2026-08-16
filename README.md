# IMRNNs

Interpretable Modular Retrieval Neural Networks (IMRNNs) rerank candidates from
a frozen dense retriever using lightweight query/document embedding modulation.

[EACL 2026 paper](https://aclanthology.org/2026.findings-eacl.333/) ·
[Hugging Face checkpoint](https://huggingface.co/yashsaxena21/IMRNNs) ·
[project website](https://yashsaxena21.github.io/IMRNNs-web/)

This repository contains one implementation and one validated checkpoint:
MiniLM trained for SciFact at
`checkpoints/validated/minilm/imrnns-minilm-scifact.pt`.

## Install

```bash
pip install imrnns
```

From this checkout:

```bash
python -m pip install .
```

Training and BEIR evaluation dependencies are optional:

```bash
python -m pip install ".[train,eval]"
```

Python 3.10–3.12 and CPU inference are supported.

## Quickstart

```python
from imrnns import IMRNNAdapter

adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="scifact",
    device="cpu",
)

results = adapter.rerank(
    query="What is scientific evidence?",
    documents=[
        "Evidence is used to support or refute a scientific claim.",
        "The stock market closed higher today.",
        "A recipe describes how to prepare a meal.",
    ],
    top_k=3,
)

for result in results:
    print(result.rank, result.base_score, result.adapted_score, result.score_delta, result.text)
```

Each result includes its rank, original index, optional document ID/text, raw
base-retriever cosine score, adapted score, and score delta.

## Rerank existing embeddings

`rerank_embeddings()` does not invoke a text encoder:

```python
adapter = IMRNNAdapter.from_pretrained(
    encoder="minilm",
    dataset="scifact",
    load_encoder=False,
)

results = adapter.rerank_embeddings(
    query_embedding=query_embedding,
    document_embeddings=document_embeddings,
    document_ids=document_ids,
    top_k=10,
)
```

## Interpret a retrieval decision

```python
explanation = adapter.explain(
    query="What currency is used in Mexico?",
    document="The Mexican peso is the currency of Mexico.",
    top_tokens=10,
)

print(explanation.top_query_tokens)
print(explanation.top_document_tokens)
print(explanation.score_delta)
```

The explanation back-projects query/document modulation vectors through the
Moore–Penrose pseudoinverse of the learned projector and reports aligned
encoder-vocabulary concepts.

## Training recipe

Prepare a shared embedding and dense-negative cache:

```bash
imrnns cache \
  --encoder minilm \
  --dataset scifact \
  --cache-dir ./cache/minilm-scifact \
  --device cpu
```

Train and evaluate:

```bash
imrnns train \
  --encoder minilm \
  --dataset scifact \
  --cache-dir ./cache/minilm-scifact \
  --output-dir ./checkpoints
```

The fixed recipe uses:

- a same-dimensional projector initialized to identity;
- 128 hidden units and zero dropout;
- one highest-relevance positive and 63 dense hard negatives from the raw
  retriever's top 100 per query;
- improvement-margin loss
  `mean(max(0, 0.05 + base_margin - adapted_margin))`;
- Adam with learning rate `1e-4`, weight decay `1e-5`, and batch size 32;
- up to 30 epochs with patience 7;
- best-epoch selection by validation mean nDCG@10, Recall@10, and MRR@10,
  including epoch 0 as a candidate.

Training checkpoints store the exact encoder revision, cache manifest, split
hashes, training history, base/adapted test metrics, metric deltas, and strict
pass status. A dataset passes only when every adapted metric is greater than
the corresponding base metric.

## Validated results

The included checkpoint was selected on SciFact validation and evaluated once
on the complete official 300-query test set with the same top-100 candidates
for the raw and adapted rankings.

| Metric | Raw MiniLM | IMRNN | Delta |
| --- | ---: | ---: | ---: |
| nDCG@10 | 0.64508 | 0.69166 | +0.04658 |
| Recall@10 | 0.78333 | 0.84333 | +0.06000 |
| MRR@10 | 0.60472 | 0.64800 | +0.04328 |

The recipe also passed all three validation metrics on NFCorpus and ArguAna.
Held-out testing passed the strict all-metrics rule on SciFact and ArguAna;
NFCorpus improved nDCG@10 and Recall@10 but missed MRR@10 by 0.00014. Full
numbers and split details are in [TRAINING_STUDY.md](TRAINING_STUDY.md).

## Evaluate a checkpoint

```bash
imrnns evaluate \
  --encoder minilm \
  --dataset scifact \
  --datasets-dir ./datasets \
  --cache-dir ./cache/minilm-scifact \
  --checkpoint checkpoints/validated/minilm/imrnns-minilm-scifact.pt
```

The JSON output includes `base_metrics`, adapted `metrics`, `metric_delta`, and
`beats_base_all_metrics`.

## Local BEIR-format datasets

The loader accepts:

```text
my_dataset/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    ├── train.tsv
    └── test.tsv
```

For official train/test datasets, 15% of official train is reserved for
validation and official test remains untouched. A dataset with one qrels split
uses a deterministic 70/15/15 partition.

## CLI

```text
imrnns info
imrnns download --encoder minilm --dataset scifact
imrnns list-assets
imrnns cache ...
imrnns train ...
imrnns evaluate ...
imrnns run ...
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests scripts examples
python -m build
twine check dist/*
```

## Citation

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

## License

The code, scripts, documentation, and checkpoints are licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
See [LICENSE](LICENSE) and [ATTRIBUTION.md](ATTRIBUTION.md).
