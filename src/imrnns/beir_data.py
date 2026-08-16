from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from .datasets.local import IMRNNDataset, load_beir_directory

DatasetSplit = IMRNNDataset


def download_beir_dataset(dataset_name: str, datasets_dir: Path) -> Path:
    try:
        from beir import util
    except ImportError as exc:  # pragma: no cover - exercised in minimal-install smoke tests
        raise ImportError('BEIR download support requires `pip install "imrnns[eval]"`.') from exc
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    return Path(util.download_and_unzip(url, str(datasets_dir)))


def _dataset_path(dataset_name: str, datasets_dir: Path) -> Path:
    candidate = datasets_dir / dataset_name
    return candidate if candidate.is_dir() else download_beir_dataset(dataset_name, datasets_dir)


def _limit(dataset: DatasetSplit, max_queries: Optional[int]) -> DatasetSplit:
    if max_queries is None:
        return dataset
    query_ids = sorted(dataset.queries)[:max_queries]
    return DatasetSplit(
        corpus=dataset.corpus,
        queries={query_id: dataset.queries[query_id] for query_id in query_ids},
        qrels={query_id: dataset.qrels[query_id] for query_id in query_ids},
    )


def load_beir_source(
    dataset_name: str,
    datasets_dir: Path,
    max_queries: Optional[int] = None,
    source_split: str = "test",
) -> DatasetSplit:
    data_path = _dataset_path(dataset_name, datasets_dir)
    if dataset_name.lower() == "msmarco" and source_split == "test":
        source_split = "train"
    return _limit(load_beir_directory(data_path, split=source_split), max_queries)


def _subset(base: DatasetSplit, query_ids: list[str]) -> DatasetSplit:
    return DatasetSplit(
        corpus=base.corpus,
        queries={query_id: base.queries[query_id] for query_id in query_ids},
        qrels={query_id: base.qrels[query_id] for query_id in query_ids},
    )


def _partition(query_ids: list[str], fractions: tuple[float, ...], seed: int) -> list[list[str]]:
    shuffled = sorted(query_ids)
    random.Random(seed).shuffle(shuffled)
    boundaries: list[int] = []
    running = 0.0
    for fraction in fractions[:-1]:
        running += fraction
        boundaries.append(round(len(shuffled) * running))
    starts = [0, *boundaries]
    ends = [*boundaries, len(shuffled)]
    return [shuffled[start:end] for start, end in zip(starts, ends)]


def load_beir_splits(
    dataset_name: str,
    datasets_dir: Path,
    max_queries: Optional[int] = None,
    source_split: str = "test",
    seed: int = 42,
) -> dict[str, DatasetSplit]:
    """Load deterministic train/validation/test splits.

    When official train and test qrels both exist, train is split 85/15 for
    training/validation and the official test set is kept intact. Datasets with
    only one qrels split use a 70/15/15 partition based on sorted IDs and an
    explicit seed.
    """

    data_path = _dataset_path(dataset_name, datasets_dir)
    train_qrels = data_path / "qrels" / "train.tsv"
    test_qrels = data_path / "qrels" / "test.tsv"
    if train_qrels.is_file() and test_qrels.is_file() and dataset_name.lower() != "msmarco":
        official_train = _limit(load_beir_directory(data_path, split="train"), max_queries)
        official_test = _limit(load_beir_directory(data_path, split="test"), max_queries)
        train_ids, validation_ids = _partition(list(official_train.queries), (0.85, 0.15), seed)
        return {
            "train": _subset(official_train, train_ids),
            "validation": _subset(official_train, validation_ids),
            "test": official_test,
        }

    base = load_beir_source(
        dataset_name,
        datasets_dir=datasets_dir,
        max_queries=max_queries,
        source_split=source_split,
    )
    train_ids, validation_ids, test_ids = _partition(list(base.queries), (0.70, 0.15, 0.15), seed)
    return {
        "train": _subset(base, train_ids),
        "validation": _subset(base, validation_ids),
        "test": _subset(base, test_ids),
    }
