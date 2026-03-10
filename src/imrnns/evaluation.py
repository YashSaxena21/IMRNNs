from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import CachedSplit
from .model import BiHyperNetIR

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
    if index is None:
        scores = all_document_embeddings @ query_embedding
        top_indices = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top_scores = scores[top_indices]
        order = np.argsort(-top_scores)
        return top_scores[order], top_indices[order]
    query_embedding = query_embedding.reshape(1, -1).astype("float32")
    scores, indices = index.search(query_embedding, k)
    return scores[0], indices[0]


def _compute_metrics(ranked_doc_ids: list[str], qrel: dict[str, int], k_values: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        top_docs = ranked_doc_ids[:k]
        mrr = 0.0
        for rank, doc_id in enumerate(top_docs, start=1):
            if qrel.get(doc_id, 0) > 0:
                mrr = 1.0 / rank
                break
        metrics[f"MRR@{k}"] = mrr

        total_relevant = sum(1 for rel in qrel.values() if rel > 0)
        retrieved_relevant = sum(1 for doc_id in top_docs if qrel.get(doc_id, 0) > 0)
        metrics[f"Recall@{k}"] = retrieved_relevant / total_relevant if total_relevant else 0.0

        dcg = 0.0
        ideal_relevances = sorted(qrel.values(), reverse=True)[:k]
        for rank, doc_id in enumerate(top_docs, start=1):
            relevance = qrel.get(doc_id, 0)
            if relevance > 0:
                dcg += (2**relevance - 1) / math.log2(rank + 1)
        idcg = 0.0
        for rank, relevance in enumerate(ideal_relevances, start=1):
            if relevance > 0:
                idcg += (2**relevance - 1) / math.log2(rank + 1)
        metrics[f"NDCG@{k}"] = dcg / idcg if idcg else 0.0
    return metrics


def evaluate_model(
    model: BiHyperNetIR,
    cached_split: CachedSplit,
    device: str,
    feedback_k: int = 100,
    ranking_k: int = 100,
    k_values: Optional[List[int]] = None,
) -> dict[str, float]:
    if k_values is None:
        k_values = [10]

    model.eval()

    document_ids = sorted(
        doc_id for doc_id in cached_split.split.corpus.keys() if doc_id in cached_split.document_embeddings
    )
    document_tensor = torch.stack(
        [cached_split.document_embeddings[doc_id].float() for doc_id in document_ids], dim=0
    ).to(device)
    projected_documents = (
        F.normalize(model.project(document_tensor), p=2, dim=-1).detach().cpu().numpy()
    )
    index = _build_search_index(projected_documents)

    aggregated = defaultdict(list)

    with torch.no_grad():
        for qid, query_embedding in tqdm(cached_split.query_embeddings.items(), desc="evaluate", leave=False):
            if qid not in cached_split.split.qrels:
                continue

            base_query = F.normalize(model.project(query_embedding.float().unsqueeze(0).to(device)), p=2, dim=-1)
            scores, indices = _search(
                index=index,
                all_document_embeddings=projected_documents,
                query_embedding=base_query.squeeze(0).detach().cpu().numpy(),
                k=min(feedback_k, len(document_ids)),
            )

            candidate_ids = [document_ids[idx] for idx in indices if 0 <= idx < len(document_ids)]
            if not candidate_ids:
                continue

            candidate_embeddings = torch.stack(
                [cached_split.document_embeddings[doc_id].float() for doc_id in candidate_ids],
                dim=0,
            ).to(device)

            _, _, rerank_scores = model.encode_candidates(query_embedding.float().to(device), candidate_embeddings)
            rerank_scores = rerank_scores.cpu().tolist()
            reranked = [
                doc_id for doc_id, _ in sorted(zip(candidate_ids, rerank_scores), key=lambda item: item[1], reverse=True)
            ][:ranking_k]

            metrics = _compute_metrics(reranked, cached_split.split.qrels[qid], k_values)
            for name, value in metrics.items():
                aggregated[name].append(value)

    return {metric: float(np.mean(values)) for metric, values in aggregated.items()}
