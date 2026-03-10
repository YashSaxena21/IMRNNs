from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ContrastiveCachedDataset, collate_contrastive_batch
from .model import BiHyperNetIR


class MultipleNegativesRankingLoss(torch.nn.Module):
    def __init__(self, scale: float = 10.0):
        super().__init__()
        self.scale = scale
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, query_embeddings: torch.Tensor, positive_embeddings: torch.Tensor) -> torch.Tensor:
        similarities = torch.matmul(
            F.normalize(query_embeddings, p=2, dim=-1),
            F.normalize(positive_embeddings, p=2, dim=-1).transpose(0, 1),
        ) * self.scale
        labels = torch.arange(similarities.size(0), device=similarities.device)
        return self.cross_entropy(similarities, labels)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_negatives: int = 20


def build_dataloader(dataset: ContrastiveCachedDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_contrastive_batch,
    )


def evaluate_loss(
    model: BiHyperNetIR,
    dataloader: DataLoader,
    device: str,
    loss_fn: MultipleNegativesRankingLoss,
) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            query_embeddings = batch["query_embeddings"].to(device)
            documents = batch["documents"].to(device)
            modulated_queries, modulated_documents, _ = model(query_embeddings, documents)
            positive_documents = modulated_documents[:, 0, :]
            loss = loss_fn(modulated_queries, positive_documents)
            total_loss += loss.item()
            steps += 1
    model.train()
    return total_loss / max(steps, 1)


def train_model(
    model: BiHyperNetIR,
    train_dataset: ContrastiveCachedDataset,
    val_dataset: ContrastiveCachedDataset,
    config: TrainingConfig,
    device: str,
) -> dict[str, float]:
    train_loader = build_dataloader(train_dataset, config.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, config.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = MultipleNegativesRankingLoss()
    model.to(device)
    best_val_loss = float("inf")
    history: dict[str, float] = {}

    for epoch in range(1, config.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        running_loss = 0.0
        steps = 0
        for batch in progress:
            optimizer.zero_grad()
            query_embeddings = batch["query_embeddings"].to(device)
            documents = batch["documents"].to(device)
            modulated_queries, modulated_documents, _ = model(query_embeddings, documents)
            positive_documents = modulated_documents[:, 0, :]
            loss = loss_fn(modulated_queries, positive_documents)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(steps, 1)
        val_loss = evaluate_loss(model, val_loader, device, loss_fn)
        best_val_loss = min(best_val_loss, val_loss)
        history = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "epochs_completed": float(epoch),
        }

    return history
