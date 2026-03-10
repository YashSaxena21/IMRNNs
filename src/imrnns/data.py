from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset

from .beir_data import DatasetSplit
from .encoders import EncoderSpec


@dataclass(frozen=True)
class CachedSplit:
    split: DatasetSplit
    document_embeddings: dict[str, torch.Tensor]
    query_embeddings: dict[str, torch.Tensor]
    negatives: dict[str, list[str]]


def _query_cache_path(cache_dir: Path, split_name: str, encoder_key: str) -> Path:
    return cache_dir / split_name / f"query_embeddings_{encoder_key}.pt"


def load_document_embeddings(cache_dir: Path, split_name: str) -> dict[str, torch.Tensor]:
    return torch.load(cache_dir / split_name / "embeddings.pt", map_location="cpu", weights_only=True)


def load_negatives(cache_dir: Path, split_name: str) -> dict[str, list[str]]:
    with open(cache_dir / split_name / "negatives.json") as handle:
        return json.load(handle)


def encode_queries(
    queries: dict[str, str],
    encoder_spec: EncoderSpec,
    cache_dir: Path,
    split_name: str,
    device: str,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    cache_path = _query_cache_path(cache_dir, split_name, encoder_spec.key)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(encoder_spec.model_name, device=device)
    query_ids = list(queries.keys())
    texts = [encoder_spec.query_prefix + queries[qid] for qid in query_ids]
    encoded = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device,
    )
    query_embeddings = {qid: embedding.cpu() for qid, embedding in zip(query_ids, encoded)}
    torch.save(query_embeddings, cache_path)
    return query_embeddings


def load_cached_split(
    cache_dir: Path,
    split_name: str,
    dataset_source: DatasetSplit,
    encoder_spec: EncoderSpec,
    device: str,
) -> CachedSplit:
    negatives = load_negatives(cache_dir, split_name)
    cached_qids = list(negatives.keys())
    filtered_queries = {
        qid: dataset_source.queries[qid]
        for qid in cached_qids
        if qid in dataset_source.queries and qid in dataset_source.qrels
    }
    filtered_qrels = {qid: dataset_source.qrels[qid] for qid in filtered_queries}
    filtered_split = DatasetSplit(
        corpus=dataset_source.corpus,
        queries=filtered_queries,
        qrels=filtered_qrels,
    )
    return CachedSplit(
        split=filtered_split,
        document_embeddings=load_document_embeddings(cache_dir, split_name),
        query_embeddings=encode_queries(filtered_split.queries, encoder_spec, cache_dir, split_name, device),
        negatives=negatives,
    )


class ContrastiveCachedDataset(Dataset):
    def __init__(
        self,
        cached_split: CachedSplit,
        num_negatives: int,
    ) -> None:
        self.cached_split = cached_split
        self.num_negatives = num_negatives
        self.examples: list[tuple[str, str, list[str]]] = []

        for qid, qrel in cached_split.split.qrels.items():
            if qid not in cached_split.query_embeddings:
                continue
            positives = [doc_id for doc_id, rel in qrel.items() if rel > 0 and doc_id in cached_split.document_embeddings]
            negatives = [doc_id for doc_id in cached_split.negatives.get(qid, []) if doc_id in cached_split.document_embeddings]
            if not positives or not negatives:
                continue
            self.examples.append((qid, positives[0], negatives[:num_negatives]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        qid, positive_id, negative_ids = self.examples[index]
        query_embedding = self.cached_split.query_embeddings[qid].float()
        positive_embedding = self.cached_split.document_embeddings[positive_id].float()
        normalized_negative_ids = list(negative_ids[: self.num_negatives])
        if not normalized_negative_ids:
            normalized_negative_ids = [positive_id] * self.num_negatives
        while len(normalized_negative_ids) < self.num_negatives:
            normalized_negative_ids.append(normalized_negative_ids[-1])

        negative_embeddings = [
            self.cached_split.document_embeddings[doc_id].float() for doc_id in normalized_negative_ids
        ]
        documents = torch.stack([positive_embedding, *negative_embeddings], dim=0)
        return {
            "qid": qid,
            "query_embedding": query_embedding,
            "documents": documents,
        }


def collate_contrastive_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    return {
        "qids": [item["qid"] for item in batch],
        "query_embeddings": torch.stack([item["query_embedding"] for item in batch], dim=0),
        "documents": torch.stack([item["documents"] for item in batch], dim=0),
    }
