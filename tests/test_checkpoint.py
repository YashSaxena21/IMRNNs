from pathlib import Path

import pytest
import torch

from imrnns.checkpoints import CHECKPOINT_FORMAT, CheckpointCompatibilityError, load_model, save_checkpoint
from imrnns.hub import checkpoint_repo_path
from imrnns.model import IMRNN, ModelConfig


def test_checkpoint_roundtrip_preserves_final_config(tmp_path: Path):
    config = ModelConfig(input_dim=11)
    model = IMRNN(config)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {"encoder_model_name": "example/model", "custom": 3})
    loaded, metadata, missing, unexpected = load_model(path)
    assert loaded.config == config
    assert metadata["custom"] == 3
    assert metadata["checkpoint_format"] == CHECKPOINT_FORMAT
    assert missing == unexpected == []
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])


def test_checkpoint_metadata_is_safe_for_weights_only_loading(tmp_path: Path):
    class VersionLike:
        def __str__(self):
            return "2.13.0"

    path = tmp_path / "safe.pt"
    save_checkpoint(path, IMRNN(ModelConfig(input_dim=4)), {"environment": {"torch": VersionLike()}})
    _, metadata, _, _ = load_model(path)
    assert metadata["environment"]["torch"] == "2.13.0"


def test_non_final_checkpoint_is_rejected(tmp_path: Path):
    path = tmp_path / "unsupported.pt"
    torch.save(IMRNN(ModelConfig(input_dim=8)).state_dict(), path)
    with pytest.raises(CheckpointCompatibilityError, match="Unsupported checkpoint format"):
        load_model(path)


def test_hub_exposes_only_the_validated_checkpoint():
    assert checkpoint_repo_path("minilm", "scifact") == ("checkpoints/validated/minilm/imrnns-minilm-scifact.pt")
    with pytest.raises(ValueError, match="currently provides only"):
        checkpoint_repo_path("e5", "scifact")


def test_checkpoint_shape_mismatch_is_not_silently_ignored(tmp_path: Path):
    model = IMRNN(ModelConfig(input_dim=8))
    path = tmp_path / "broken.pt"
    save_checkpoint(path, model)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload["model_state"]["doc_norm.bias"]
    torch.save(payload, path)
    with pytest.raises(CheckpointCompatibilityError, match="incompatible"):
        load_model(path)


def test_validated_scifact_checkpoint_records_strict_base_improvement():
    path = Path("checkpoints/validated/minilm/imrnns-minilm-scifact.pt")
    if not path.is_file():
        pytest.skip("validated checkpoint not present in source archive")
    model, metadata, missing, unexpected = load_model(path)
    assert model.config.input_dim == model.config.output_dim == 384
    assert metadata["objective"] == "improvement-margin"
    assert metadata["beats_base_all_metrics"] is True
    assert all(delta > 0 for delta in metadata["evaluation_delta"].values())
    assert missing == unexpected == []
