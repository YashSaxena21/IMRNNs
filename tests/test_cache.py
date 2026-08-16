import json
from pathlib import Path

import pytest
import torch

from imrnns.api import _validate_cache
from imrnns.caching import _cache_is_reusable, mine_dense_negatives
from imrnns.data import load_document_embeddings, load_negatives
from imrnns.encoders import EncoderSpec


def test_shared_document_cache_is_reused_across_splits(tmp_path: Path):
    (tmp_path / "documents").mkdir()
    (tmp_path / "train").mkdir()
    (tmp_path / "validation").mkdir()
    embeddings = {"d1": torch.ones(3)}
    torch.save(embeddings, tmp_path / "documents" / "embeddings.pt")
    for split in ("train", "validation"):
        (tmp_path / split / "negatives.json").write_text(json.dumps({"q": ["d1"]}), encoding="utf-8")
    assert torch.equal(load_document_embeddings(tmp_path)["d1"], torch.ones(3))
    assert not (tmp_path / "train" / "embeddings.pt").exists()
    assert load_negatives(tmp_path, "validation") == {"q": ["d1"]}


def test_complete_matching_cache_is_reusable(tmp_path: Path):
    spec = EncoderSpec("fake", "fake/model", 3, revision="abc")
    manifest = {
        "schema_version": 2,
        "dataset": "tiny",
        "encoder": "fake",
        "model_name": "fake/model",
        "model_revision": "abc",
        "num_negatives": 2,
        "negative_pool": 10,
        "negative_method": "dense",
        "max_queries": None,
        "seed": 42,
    }
    (tmp_path / "documents").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "documents" / "ids.json").touch()
    (tmp_path / "documents" / "embeddings.pt").touch()
    for split in ("train", "validation", "test"):
        (tmp_path / split).mkdir()
        for name in ("query_ids.json", "query_embeddings.pt", "negatives.json"):
            (tmp_path / split / name).touch()

    assert _cache_is_reusable(
        tmp_path,
        dataset_name="tiny",
        encoder_spec=spec,
        num_negatives=2,
        negative_pool=10,
        max_queries=None,
        seed=42,
    )
    assert not _cache_is_reusable(
        tmp_path,
        dataset_name="tiny",
        encoder_spec=spec,
        num_negatives=2,
        negative_pool=10,
        max_queries=None,
        seed=7,
    )
    assert (
        _validate_cache(
            tmp_path,
            dataset="tiny",
            model_name="fake/model",
            model_revision="abc",
            max_queries=None,
            seed=42,
            num_negatives=2,
        )
        == manifest
    )
    with pytest.raises(ValueError, match="seed"):
        _validate_cache(
            tmp_path,
            dataset="tiny",
            model_name="fake/model",
            model_revision="abc",
            max_queries=None,
            seed=7,
            num_negatives=2,
        )


def test_dense_negative_mining_excludes_relevant_documents():
    queries = {"q": torch.tensor([1.0, 0.0])}
    documents = {
        "positive": torch.tensor([1.0, 0.0]),
        "hard": torch.tensor([0.9, 0.1]),
        "easy": torch.tensor([-1.0, 0.0]),
    }
    negatives = mine_dense_negatives(
        queries,
        documents,
        {"q": {"positive": 1}},
        num_negatives=2,
        top_k=3,
        device="cpu",
    )
    assert negatives == {"q": ["hard", "easy"]}


def test_cache_manifest_must_be_a_json_object(tmp_path: Path):
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        _validate_cache(
            tmp_path,
            dataset="tiny",
            model_name="fake/model",
            model_revision=None,
            max_queries=None,
            seed=42,
        )
