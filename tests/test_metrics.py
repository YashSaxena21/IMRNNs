import math

import pytest

from imrnns.beir_data import DatasetSplit
from imrnns.data import CachedSplit
from imrnns.evaluation import (
    _search,
    compute_metrics,
    compute_mrr,
    compute_ndcg,
    compute_recall,
    evaluate_model_with_baseline,
)
from imrnns.model import IMRNN, ModelConfig


def test_hand_computable_binary_metrics():
    ranking = ["d0", "d1", "d2", "d3"]
    qrel = {"d1": 1, "d3": 1}
    assert compute_mrr(ranking, qrel, 3) == 0.5
    assert compute_recall(ranking, qrel, 3) == 0.5
    expected_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert math.isclose(compute_ndcg(ranking, qrel, 3), expected_ndcg)
    metrics = compute_metrics(ranking, qrel, [3])
    assert metrics == {"MRR@3": 0.5, "Recall@3": 0.5, "NDCG@3": expected_ndcg}


def test_graded_ndcg_and_empty_qrels():
    assert compute_ndcg(["high", "low"], {"high": 2, "low": 1}, 2) == 1.0
    assert compute_recall(["anything"], {}, 10) == 0.0


def test_search_rejects_non_positive_k():
    import numpy as np

    scores, indices = _search(None, np.eye(2, dtype=np.float32), np.ones(2, dtype=np.float32), 0)
    assert scores.size == indices.size == 0


def test_evaluation_fails_closed_on_empty_corpus():
    empty = DatasetSplit(corpus={}, queries={}, qrels={})
    cached = CachedSplit(split=empty, document_embeddings={}, query_embeddings={}, negatives={})
    with pytest.raises(ValueError, match="no cached corpus"):
        evaluate_model_with_baseline(IMRNN(ModelConfig(input_dim=4)), cached, device="cpu")
