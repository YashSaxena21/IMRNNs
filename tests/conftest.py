from __future__ import annotations

import torch

from imrnns.adapter import IMRNNAdapter
from imrnns.encoders import EncoderSpec
from imrnns.model import IMRNN, ModelConfig


class FakeTokenizer:
    all_special_ids = [0]

    def convert_ids_to_tokens(self, ids):
        return [f"tok_{item}" for item in ids]


class FakeAutoModel:
    def __init__(self, dimension: int):
        torch.manual_seed(9)
        self.embedding = torch.nn.Embedding(32, dimension)

    def get_input_embeddings(self):
        return self.embedding


class FakeModule:
    def __init__(self, dimension: int):
        self.auto_model = FakeAutoModel(dimension)


class FakeEncoder:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.tokenizer = FakeTokenizer()
        self.module = FakeModule(dimension)
        self.calls = 0

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self.module

    def encode(self, texts, **kwargs):
        self.calls += 1
        rows = []
        for text in texts:
            seed = sum(ord(character) for character in text) % 997
            generator = torch.Generator().manual_seed(seed)
            rows.append(torch.randn(self.dimension, generator=generator))
        return torch.stack(rows)


def make_adapter(dimension: int = 8) -> tuple[IMRNNAdapter, FakeEncoder]:
    torch.manual_seed(7)
    model = IMRNN(ModelConfig(input_dim=dimension))
    model.eval()
    encoder = FakeEncoder(dimension)
    adapter = IMRNNAdapter(
        model=model,
        encoder=encoder,
        encoder_spec=EncoderSpec("fake", "fake/model", dimension),
        metadata={},
        device="cpu",
    )
    return adapter, encoder
