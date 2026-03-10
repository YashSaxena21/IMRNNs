from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    corpus: dict
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def download_beir_dataset(dataset_name: str, datasets_dir: Path) -> Path:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, str(datasets_dir))
    return Path(data_path)


def load_beir_source(
    dataset_name: str,
    datasets_dir: Path,
    max_queries: Optional[int] = None,
    source_split: str = "test",
) -> DatasetSplit:
    if dataset_name.lower() == "msmarco" and source_split == "test":
        source_split = "train"
    data_path = download_beir_dataset(dataset_name, datasets_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split=source_split)

    qids = list(queries.keys())
    if max_queries is not None:
        qids = qids[:max_queries]
        queries = {qid: queries[qid] for qid in qids}
        qrels = {qid: qrels[qid] for qid in qids if qid in qrels}
    return DatasetSplit(corpus=corpus, queries=queries, qrels=qrels)

def load_beir_splits(
    dataset_name: str,
    datasets_dir: Path,
    max_queries: Optional[int] = None,
    source_split: str = "test",
) -> dict[str, DatasetSplit]:
    base = load_beir_source(dataset_name, datasets_dir, max_queries=max_queries, source_split=source_split)
    qids = list(base.queries.keys())
    train_ids, temp_ids = train_test_split(qids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    splits: dict[str, DatasetSplit] = {}
    for split_name, split_ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        split_queries = {qid: base.queries[qid] for qid in split_ids if qid in base.qrels}
        split_qrels = {qid: base.qrels[qid] for qid in split_ids if qid in base.qrels}
        splits[split_name] = DatasetSplit(corpus=base.corpus, queries=split_queries, qrels=split_qrels)
    return splits
