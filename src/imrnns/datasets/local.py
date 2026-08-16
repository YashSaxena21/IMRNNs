from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IMRNNDataset:
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def __post_init__(self) -> None:
        missing_queries = set(self.qrels) - set(self.queries)
        if missing_queries:
            sample = sorted(missing_queries)[:3]
            raise ValueError(f"qrels reference missing query IDs, for example: {sample}")
        missing_documents = {
            document_id
            for query_qrels in self.qrels.values()
            for document_id in query_qrels
            if document_id not in self.corpus
        }
        if missing_documents:
            sample = sorted(missing_documents)[:3]
            raise ValueError(f"qrels reference missing document IDs, for example: {sample}")

    @classmethod
    def from_beir_directory(cls, path: str | Path, *, split: str = "test") -> "IMRNNDataset":
        return load_beir_directory(path, split=split)


def _load_jsonl(path: Path, *, id_field: str = "_id") -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required BEIR file is missing: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
    for row in rows:
        if id_field not in row:
            raise ValueError(f"Every row in {path} must contain '{id_field}'.")
    return rows


def load_beir_directory(path: str | Path, *, split: str = "test") -> IMRNNDataset:
    root = Path(path)
    corpus_rows = _load_jsonl(root / "corpus.jsonl")
    query_rows = _load_jsonl(root / "queries.jsonl")
    corpus = {
        str(row["_id"]): {"title": str(row.get("title") or ""), "text": str(row.get("text") or "")}
        for row in corpus_rows
    }
    queries = {str(row["_id"]): str(row.get("text") or "") for row in query_rows}
    qrels_path = root / "qrels" / f"{split}.tsv"
    if not qrels_path.is_file():
        available = sorted(item.stem for item in (root / "qrels").glob("*.tsv"))
        raise FileNotFoundError(
            f"Qrels split '{split}' is missing at {qrels_path}. Available splits: {available or 'none'}."
        )
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{qrels_path} must have tab-separated columns: {sorted(required)}")
        for row in reader:
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    used_queries = {query_id: queries[query_id] for query_id in qrels if query_id in queries}
    return IMRNNDataset(corpus=corpus, queries=used_queries, qrels=qrels)
