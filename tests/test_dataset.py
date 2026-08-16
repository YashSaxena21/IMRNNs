import json
from pathlib import Path

import pytest

from imrnns.beir_data import _partition
from imrnns.datasets import IMRNNDataset, load_beir_directory


def _write_dataset(root: Path):
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        "\n".join(
            json.dumps(row) for row in [{"_id": "d1", "title": "T", "text": "one"}, {"_id": "d2", "text": "two"}]
        ),
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(json.dumps({"_id": "q1", "text": "query"}) + "\n", encoding="utf-8")
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")


def test_local_beir_loader(tmp_path: Path):
    _write_dataset(tmp_path)
    dataset = load_beir_directory(tmp_path)
    assert isinstance(dataset, IMRNNDataset)
    assert dataset.corpus["d1"]["title"] == "T"
    assert dataset.queries == {"q1": "query"}
    assert dataset.qrels == {"q1": {"d1": 1}}


def test_local_loader_reports_missing_split(tmp_path: Path):
    _write_dataset(tmp_path)
    with pytest.raises(FileNotFoundError, match="Available splits"):
        load_beir_directory(tmp_path, split="train")


def test_split_partition_is_seeded_and_independent_of_input_order():
    query_ids = [f"q{index}" for index in range(20)]
    first = _partition(query_ids, (0.70, 0.15, 0.15), seed=42)
    second = _partition(list(reversed(query_ids)), (0.70, 0.15, 0.15), seed=42)
    different_seed = _partition(query_ids, (0.70, 0.15, 0.15), seed=7)
    assert first == second
    assert first != different_seed
    assert [len(split) for split in first] == [14, 3, 3]
