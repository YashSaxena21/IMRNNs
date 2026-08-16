import pytest
import torch

from imrnns.model import IMRNN, ModelConfig
from imrnns.training import ImprovementMarginObjective, TrainingConfig, initialize_projector, set_seed


def test_improvement_loss_compares_adapted_and_base_margins():
    objective = ImprovementMarginObjective(margin=0.05)
    base = torch.tensor([[0.8, 0.7]])
    better = objective(torch.tensor([[0.9, 0.7]]), base)
    worse = objective(torch.tensor([[0.7, 0.8]]), base)
    assert better == 0
    assert worse > better


def test_improvement_loss_validates_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        ImprovementMarginObjective(-0.1)
    objective = ImprovementMarginObjective()
    with pytest.raises(ValueError, match="identical"):
        objective(torch.ones(1, 2), torch.ones(1, 3))
    with pytest.raises(ValueError, match="at least one negative"):
        objective(torch.ones(1, 1), torch.ones(1, 1))
    with pytest.raises(ValueError, match="must be positive"):
        TrainingConfig(epochs=0)


def test_identity_projector_initialization():
    model = IMRNN(ModelConfig(input_dim=4))
    initialize_projector(model)
    torch.testing.assert_close(model.projector.weight, torch.eye(4))
    torch.testing.assert_close(model.projector.bias, torch.zeros(4))


def test_seed_is_applied_before_model_initialization():
    set_seed(42)
    first = IMRNN(ModelConfig(input_dim=4))
    set_seed(42)
    second = IMRNN(ModelConfig(input_dim=4))
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name])
