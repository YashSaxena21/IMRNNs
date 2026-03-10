from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .beir_data import DatasetSplit, load_beir_splits
from .encoders import EncoderSpec


def _document_text(document: Dict[str, str], encoder_spec: EncoderSpec) -> str:
    title = (document.get("title") or "").strip()
    text = (document.get("text") or "").strip()
    combined = f"{title}\n{text}".strip() if title else text
    return f"{encoder_spec.passage_prefix}{combined}" if encoder_spec.passage_prefix else combined


def _query_text(query: str, encoder_spec: EncoderSpec) -> str:
    return f"{encoder_spec.query_prefix}{query}" if encoder_spec.query_prefix else query


class BM25NegativeMiner:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "by", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "can", "this", "that", "these", "those",
        }
        self.postings: Dict[str, List[tuple[int, int]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self.doc_lengths: Dict[int, int] = {}
        self.doc_ids: List[str] = []
        self.avgdl = 0.0

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        return [token for token in tokens if token not in self.stopwords and len(token) > 2]

    def fit(self, corpus: Dict[str, Dict[str, str]]) -> None:
        self.doc_ids = list(corpus.keys())
        term_doc_freq: Dict[str, set[int]] = defaultdict(set)
        total_length = 0

        for doc_idx, doc_id in enumerate(tqdm(self.doc_ids, desc="index bm25", leave=False)):
            text = f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}".strip()
            tokens = self._tokenize(text)
            total_length += len(tokens)
            self.doc_lengths[doc_idx] = len(tokens)

            term_freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            for token, tf in term_freq.items():
                self.postings[token].append((doc_idx, tf))
                term_doc_freq[token].add(doc_idx)

        self.avgdl = total_length / max(len(self.doc_ids), 1)
        total_docs = len(self.doc_ids)
        for token, doc_set in term_doc_freq.items():
            df = len(doc_set)
            self.idf[token] = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

    def mine(
        self,
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        num_negatives: int,
        top_k: int,
    ) -> Dict[str, List[str]]:
        negatives: Dict[str, List[str]] = {}
        for qid, query in tqdm(queries.items(), desc="mine negatives", leave=False):
            if qid not in qrels:
                continue
            scores = [0.0] * len(self.doc_ids)
            for token in self._tokenize(query):
                if token not in self.postings:
                    continue
                idf = self.idf[token]
                for doc_idx, tf in self.postings[token]:
                    dl = self.doc_lengths[doc_idx]
                    norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                    scores[doc_idx] += idf * ((tf * (self.k1 + 1)) / (tf + norm))

            positive_doc_ids = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
            ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
            mined: List[str] = []
            for idx in ranked:
                doc_id = self.doc_ids[idx]
                if doc_id in positive_doc_ids:
                    continue
                mined.append(doc_id)
                if len(mined) >= num_negatives:
                    break
            negatives[qid] = mined
        return negatives


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


def build_cache(
    dataset_name: str,
    encoder_spec: EncoderSpec,
    cache_dir: Path,
    datasets_dir: Path,
    device: str,
    batch_size: int = 64,
    num_negatives: int = 20,
    negative_pool: int = 200,
    max_queries: Optional[int] = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits = load_beir_splits(dataset_name, datasets_dir=datasets_dir, max_queries=max_queries)
    model = SentenceTransformer(encoder_spec.model_name, device=device)
    negative_miner = BM25NegativeMiner()
    corpus = splits["train"].corpus
    negative_miner.fit(corpus)

    document_texts = [(doc_id, _document_text(document, encoder_spec)) for doc_id, document in corpus.items()]
    document_embeddings = _encode_texts(model, document_texts, batch_size=batch_size, device=device)

    for split_name, split in splits.items():
        split_dir = cache_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        query_texts = [(qid, _query_text(query, encoder_spec)) for qid, query in split.queries.items()]
        query_embeddings = _encode_texts(model, query_texts, batch_size=batch_size, device=device)
        negatives = negative_miner.mine(
            split.queries,
            split.qrels,
            num_negatives=num_negatives,
            top_k=negative_pool,
        )

        torch.save(document_embeddings, split_dir / "embeddings.pt")
        torch.save(query_embeddings, split_dir / f"query_embeddings_{encoder_spec.key}.pt")
        with open(split_dir / "negatives.json", "w") as handle:
            json.dump(negatives, handle)

    manifest = {
        "dataset": dataset_name,
        "encoder": encoder_spec.key,
        "model_name": encoder_spec.model_name,
        "cache_dir": str(cache_dir),
        "num_negatives": num_negatives,
        "negative_pool": negative_pool,
        "max_queries": max_queries,
    }
    with open(cache_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return cache_dir
