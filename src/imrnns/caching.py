from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .beir_data import load_beir_splits
from .encoders import EncoderSpec


def _document_text(document: Dict[str, str], encoder_spec: EncoderSpec) -> str:
    title = (document.get("title") or "").strip()
    text = (document.get("text") or "").strip()
    combined = f"{title}\n{text}".strip() if title else text
    return f"{encoder_spec.passage_prefix}{combined}" if encoder_spec.passage_prefix else combined


def _query_text(query: str, encoder_spec: EncoderSpec) -> str:
    return f"{encoder_spec.query_prefix}{query}" if encoder_spec.query_prefix else query


def _encode_texts(
    model: SentenceTransformer,
    items: Iterable[tuple[str, str]],
    batch_size: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    item_list = list(items)
    ids = [item_id for item_id, _ in item_list]
    texts = [text for _, text in item_list]
    outputs: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="encode", leave=False):
            batch_ids = ids[start : start + batch_size]
            batch_texts = texts[start : start + batch_size]
            embeddings = model.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=device,
            )
            for item_id, embedding in zip(batch_ids, embeddings):
                outputs[item_id] = embedding.cpu()
    return outputs


def mine_dense_negatives(
    query_embeddings: Dict[str, torch.Tensor],
    document_embeddings: Dict[str, torch.Tensor],
    qrels: Dict[str, Dict[str, int]],
    *,
    num_negatives: int,
    top_k: int,
    device: str,
    batch_size: int = 128,
) -> Dict[str, List[str]]:
    """Mine hard negatives using raw cosine similarity from the base encoder."""

    if num_negatives <= 0 or top_k <= 0 or batch_size <= 0:
        raise ValueError("num_negatives, top_k, and batch_size must be positive.")
    document_ids = sorted(document_embeddings)
    if not document_ids:
        return {}
    document_matrix = F.normalize(
        torch.stack([document_embeddings[document_id].float() for document_id in document_ids]),
        p=2,
        dim=-1,
    ).to(device)
    query_ids = sorted(query_id for query_id in query_embeddings if query_id in qrels)
    negatives: Dict[str, List[str]] = {}
    retrieval_k = min(top_k, len(document_ids))
    with torch.no_grad():
        for start in tqdm(range(0, len(query_ids), batch_size), desc="mine dense negatives", leave=False):
            batch_ids = query_ids[start : start + batch_size]
            query_matrix = F.normalize(
                torch.stack([query_embeddings[query_id].float() for query_id in batch_ids]),
                p=2,
                dim=-1,
            ).to(device)
            ranked_indices = torch.topk(query_matrix @ document_matrix.T, k=retrieval_k, dim=1).indices.cpu()
            for query_id, indices in zip(batch_ids, ranked_indices):
                positive_ids = {document_id for document_id, relevance in qrels[query_id].items() if relevance > 0}
                mined = [document_ids[index] for index in indices.tolist() if document_ids[index] not in positive_ids]
                negatives[query_id] = mined[:num_negatives]
    return negatives


def _cache_is_reusable(
    cache_dir: Path,
    *,
    dataset_name: str,
    encoder_spec: EncoderSpec,
    num_negatives: int,
    negative_pool: int,
    max_queries: Optional[int],
    seed: int,
) -> bool:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(manifest, dict):
        return False
    expected = {
        "schema_version": 2,
        "dataset": dataset_name,
        "encoder": encoder_spec.key,
        "model_name": encoder_spec.model_name,
        "model_revision": encoder_spec.revision,
        "num_negatives": num_negatives,
        "negative_pool": negative_pool,
        "negative_method": "dense",
        "max_queries": max_queries,
        "seed": seed,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    required = [
        cache_dir / "documents" / "ids.json",
        cache_dir / "documents" / "embeddings.pt",
    ]
    for split_name in ("train", "validation", "test"):
        required.extend(
            [
                cache_dir / split_name / "query_ids.json",
                cache_dir / split_name / "query_embeddings.pt",
                cache_dir / split_name / "negatives.json",
            ]
        )
    return all(path.is_file() for path in required)


def build_cache(
    dataset_name: str,
    encoder_spec: EncoderSpec,
    cache_dir: Path,
    datasets_dir: Path,
    device: str,
    batch_size: int = 64,
    num_negatives: int = 63,
    negative_pool: int = 100,
    max_queries: Optional[int] = None,
    seed: int = 42,
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_negatives <= 0:
        raise ValueError("num_negatives must be positive.")
    if negative_pool < num_negatives:
        raise ValueError("negative_pool must be at least num_negatives.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if _cache_is_reusable(
        cache_dir,
        dataset_name=dataset_name,
        encoder_spec=encoder_spec,
        num_negatives=num_negatives,
        negative_pool=negative_pool,
        max_queries=max_queries,
        seed=seed,
    ):
        return cache_dir
    splits = load_beir_splits(
        dataset_name,
        datasets_dir=datasets_dir,
        max_queries=max_queries,
        seed=seed,
    )
    model = SentenceTransformer(
        encoder_spec.model_name,
        device=device,
        revision=encoder_spec.revision,
    )
    corpus = splits["train"].corpus

    document_texts = [(doc_id, _document_text(corpus[doc_id], encoder_spec)) for doc_id in sorted(corpus)]
    document_embeddings = _encode_texts(model, document_texts, batch_size=batch_size, device=device)

    documents_dir = cache_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    torch.save(document_embeddings, documents_dir / "embeddings.pt")
    with open(documents_dir / "ids.json", "w", encoding="utf-8") as handle:
        json.dump(list(document_embeddings), handle)

    split_manifest: dict[str, dict[str, object]] = {}
    for split_name, split in splits.items():
        split_dir = cache_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        query_texts = [(qid, _query_text(split.queries[qid], encoder_spec)) for qid in sorted(split.queries)]
        query_embeddings = _encode_texts(model, query_texts, batch_size=batch_size, device=device)
        negatives = mine_dense_negatives(
            query_embeddings,
            document_embeddings,
            split.qrels,
            num_negatives=num_negatives,
            top_k=negative_pool,
            device=device,
        )

        torch.save(query_embeddings, split_dir / "query_embeddings.pt")
        with open(split_dir / "negatives.json", "w", encoding="utf-8") as handle:
            json.dump(negatives, handle)
        query_ids = sorted(split.queries)
        with open(split_dir / "query_ids.json", "w", encoding="utf-8") as handle:
            json.dump(query_ids, handle, indent=2)
        split_manifest[split_name] = {
            "query_count": len(query_ids),
            "query_ids_sha256": hashlib.sha256("\n".join(query_ids).encode()).hexdigest(),
        }

    manifest = {
        "dataset": dataset_name,
        "encoder": encoder_spec.key,
        "model_name": encoder_spec.model_name,
        "model_revision": encoder_spec.revision,
        "cache_dir": str(cache_dir),
        "num_negatives": num_negatives,
        "negative_pool": negative_pool,
        "negative_method": "dense",
        "max_queries": max_queries,
        "seed": seed,
        "schema_version": 2,
        "documents": {
            "count": len(document_embeddings),
            "path": "documents/embeddings.pt",
            "ids_sha256": hashlib.sha256("\n".join(document_embeddings).encode()).hexdigest(),
        },
        "splits": split_manifest,
    }
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return cache_dir
