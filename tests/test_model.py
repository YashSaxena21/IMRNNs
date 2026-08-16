import pytest
import torch

from imrnns.model import HIDDEN_DIM, IMRNN, ModelConfig


def test_model_shapes_finite_gradients_and_eval_determinism():
    torch.manual_seed(1)
    model = IMRNN(ModelConfig(input_dim=12))
    queries = torch.randn(3, 12, requires_grad=True)
    documents = torch.randn(3, 4, 12, requires_grad=True)
    modulated_queries, modulated_documents, scores = model(queries, documents)
    assert modulated_queries.shape == (3, 12)
    assert modulated_documents.shape == (3, 4, 12)
    assert scores.shape == (3, 4)
    assert model.config.hidden_dim == HIDDEN_DIM
    assert torch.isfinite(scores).all()
    scores.sum().backward()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert documents.grad is not None and torch.isfinite(documents.grad).all()

    model.eval()
    with torch.no_grad():
        first = model(queries.detach(), documents.detach())[2]
        second = model(queries.detach(), documents.detach())[2]
    torch.testing.assert_close(first, second)


def test_candidate_order_only_permutes_scores():
    torch.manual_seed(2)
    model = IMRNN(ModelConfig(input_dim=8)).eval()
    query = torch.randn(1, 8)
    documents = torch.randn(1, 4, 8)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        original = model(query, documents)[2]
        permuted = model(query, documents[:, permutation])[2]
    torch.testing.assert_close(permuted, original[:, permutation])


def test_model_rejects_wrong_embedding_dimension():
    model = IMRNN(ModelConfig(input_dim=8))
    with pytest.raises(ValueError, match="Expected embeddings"):
        model.project(torch.randn(2, 7))


@pytest.mark.parametrize("input_dim", [0, -1, 3.5, True])
def test_model_config_rejects_non_positive_integer_dimensions(input_dim):
    with pytest.raises(ValueError, match="positive integer"):
        ModelConfig(input_dim=input_dim)
