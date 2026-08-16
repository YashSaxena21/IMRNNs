from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import CachedSplit
from .model import IMRNN

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def _build_search_index(doc_embeddings: np.ndarray):
    if faiss is None:
        return None
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings.astype("float32"))
    return index


def _search(
    index,
    all_document_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, len(all_document_embeddings))
    if k <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    if index is None:
        scores = all_document_embeddings @ query_embedding
        if k == len(scores):
            top_indices = np.arange(len(scores))
        else:
            top_indices = np.argpartition(-scores, k - 1)[:k]
        top_scores = scores[top_indices]
        order = np.argsort(-top_scores)
        return top_scores[order], top_indices[order]
    query_embedding = query_embedding.reshape(1, -1).astype("float32")
    scores, indices = index.search(query_embedding, k)
    return scores[0], indices[0]


def compute_mrr(ranked_doc_ids: Sequence[str], qrel: dict[str, int], k: int = 10) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if qrel.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def compute_recall(ranked_doc_ids: Sequence[str], qrel: dict[str, int], k: int = 10) -> float:
    total_relevant = sum(1 for relevance in qrel.values() if relevance > 0)
    if total_relevant == 0:
        return 0.0
    retrieved = sum(1 for doc_id in ranked_doc_ids[:k] if qrel.get(doc_id, 0) > 0)
    return retrieved / total_relevant


def compute_ndcg(ranked_doc_ids: Sequence[str], qrel: dict[str, int], k: int = 10) -> float:
    dcg = sum(
        (2 ** qrel.get(doc_id, 0) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1)
        if qrel.get(doc_id, 0) > 0
    )
    ideal_relevances = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
        if relevance > 0
    )
    return dcg / idcg if idcg else 0.0


def compute_metrics(ranked_doc_ids: list[str], qrel: dict[str, int], k_values: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"MRR@{k}"] = compute_mrr(ranked_doc_ids, qrel, k)
        metrics[f"Recall@{k}"] = compute_recall(ranked_doc_ids, qrel, k)
        metrics[f"NDCG@{k}"] = compute_ndcg(ranked_doc_ids, qrel, k)
    return metrics


_compute_metrics = compute_metrics


def evaluate_model(
    model: IMRNN,
    cached_split: CachedSplit,
    device: str,
    feedback_k: int = 100,
    ranking_k: int = 100,
    k_values: Optional[List[int]] = None,
    projection_batch_size: int = 4096,
) -> dict[str, float]:
    adapted, _ = evaluate_model_with_baseline(
        model=model,
        cached_split=cached_split,
        device=device,
        feedback_k=feedback_k,
        ranking_k=ranking_k,
        k_values=k_values,
        projection_batch_size=projection_batch_size,
    )
    return adapted


def evaluate_model_with_baseline(
    model: IMRNN,
    cached_split: CachedSplit,
    device: str,
    feedback_k: int = 100,
    ranking_k: int = 100,
    k_values: Optional[List[int]] = None,
    projection_batch_size: int = 4096,
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate adapted and base retrieval under exactly the same protocol.

    The base ranking is raw cosine retrieval from the frozen base-encoder
    embeddings and is used to select candidates. The adapted ranking reorders
    that same candidate set, making
    per-metric comparisons meaningful even when ``feedback_k`` is bounded.
    """
    if k_values is None:
        k_values = [10]
    if feedback_k <= 0 or ranking_k <= 0 or projection_batch_size <= 0:
        raise ValueError("feedback_k, ranking_k, and projection_batch_size must be positive.")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain at least one positive cutoff.")

    model.eval()

    document_ids = sorted(
        doc_id for doc_id in cached_split.split.corpus.keys() if doc_id in cached_split.document_embeddings
    )
    if not document_ids:
        raise ValueError("The evaluation split contains no cached corpus embeddings.")
    document_tensor = torch.stack([cached_split.document_embeddings[doc_id].float() for doc_id in document_ids], dim=0)
    normalized_batches = []
    for start in range(0, len(document_tensor), projection_batch_size):
        normalized_batches.append(F.normalize(document_tensor[start : start + projection_batch_size], p=2, dim=-1))
    base_documents = torch.cat(normalized_batches, dim=0).numpy()
    index = _build_search_index(base_documents)

    adapted_values = defaultdict(list)
    base_values = defaultdict(list)

    with torch.no_grad():
        for qid, query_embedding in tqdm(cached_split.query_embeddings.items(), desc="evaluate", leave=False):
            if qid not in cached_split.split.qrels:
                continue

            base_query = F.normalize(query_embedding.float().unsqueeze(0), p=2, dim=-1)
            _, indices = _search(
                index=index,
                all_document_embeddings=base_documents,
                query_embedding=base_query.squeeze(0).detach().cpu().numpy(),
                k=min(feedback_k, len(document_ids)),
            )

            candidate_ids = [document_ids[idx] for idx in indices if 0 <= idx < len(document_ids)]
            if not candidate_ids:
                continue

            base_ranked = candidate_ids[:ranking_k]

            candidate_embeddings = torch.stack(
                [cached_split.document_embeddings[doc_id].float() for doc_id in candidate_ids],
                dim=0,
            ).to(device)

            _, _, adapted_scores = model.score_candidates(query_embedding.float().to(device), candidate_embeddings)
            adapted_scores = adapted_scores.cpu().tolist()
            reranked = [
                doc_id
                for doc_id, _ in sorted(zip(candidate_ids, adapted_scores), key=lambda item: item[1], reverse=True)
            ][:ranking_k]

            adapted_metrics = _compute_metrics(reranked, cached_split.split.qrels[qid], k_values)
            base_metrics = _compute_metrics(base_ranked, cached_split.split.qrels[qid], k_values)
            for name, value in adapted_metrics.items():
                adapted_values[name].append(value)
            for name, value in base_metrics.items():
                base_values[name].append(value)

    if not adapted_values:
        raise ValueError("The evaluation split contains no queries with relevance judgments.")
    return (
        {metric: float(np.mean(values)) for metric, values in adapted_values.items()},
        {metric: float(np.mean(values)) for metric, values in base_values.items()},
    )
