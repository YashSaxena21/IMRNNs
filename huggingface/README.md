---
library_name: pytorch
tags:
  - dense-retrieval
  - information-retrieval
  - beir
  - pytorch
pipeline_tag: sentence-similarity
license: mit
---

# IMRNNs

Pretrained **IMRNNs** checkpoints for dense retrieval with **MiniLM** and **E5** backbones.

Paper:
- **IMRNNs: An Efficient Method for Interpretable Dense Retrieval via Embedding Modulation**
- arXiv: https://arxiv.org/abs/2601.20084

IMRNNs is a lightweight adapter that sits on top of a base dense retriever, projects embeddings into a shared space, modulates query and document representations, and reranks candidates.

## What This Model Repo Contains

- adapter-only IMRNN checkpoints
- minimal evaluation script
- end-to-end demo script
- `src/imrnns` package required to load and run the checkpoints

This is a **checkpoint release**, not a hosted inference model.

## Available Checkpoints

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

Each checkpoint is specific to its encoder family and dataset.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Minimal evaluation:

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

## End-to-End Python Demo

```python
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

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

repo_id = "yashsaxena21/IMRNNs"
encoder_name = "minilm"
dataset_name = "trec-covid"
device = "cpu"
k = 10

checkpoint_path = hf_hub_download(
    repo_id=repo_id,
    filename="checkpoints/pretrained/minilm/imrnns-minilm-trec-covid.pt",
    repo_type="model",
)

encoder_spec = get_encoder_spec(encoder_name)
cache_dir = Path("demo_cache") / "cache_minilm_trec-covid"
datasets_dir = Path("datasets")

if not (cache_dir / "test" / "embeddings.pt").exists():
    build_cache(
        dataset_name=dataset_name,
        encoder_spec=encoder_spec,
        cache_dir=cache_dir,
        datasets_dir=datasets_dir,
        device=device,
    )

source = load_beir_source(dataset_name, datasets_dir=datasets_dir)
cached_test = load_cached_split(cache_dir, "test", source, encoder_spec, device)

model, metadata, missing, unexpected = load_model(
    checkpoint_path=Path(checkpoint_path),
    model_config=ModelConfig(input_dim=encoder_spec.embedding_dim),
    device=device,
)

metrics = evaluate_model(
    model=model,
    cached_split=cached_test,
    device=device,
    feedback_k=100,
    ranking_k=k,
    k_values=[k],
)

print(json.dumps({"metrics": metrics, "metadata": metadata}, indent=2))
```

This evaluates the final model and reports:

- `MRR@10`
- `Recall@10`
- `NDCG@10`

## Notes

- BEIR datasets and caches are not bundled here.
- The checkpoints are not standard `transformers.from_pretrained(...)` weights.
- For training IMRNNs on additional retrievers or datasets, use the full GitHub implementation.
