from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ContrastiveCachedDataset, collate_contrastive_batch
from .model import IMRNN


class ImprovementMarginObjective(torch.nn.Module):
    """Require adapted positive-negative margins to exceed base margins."""

    def __init__(self, margin: float = 0.05):
        super().__init__()
        if margin < 0:
            raise ValueError("improvement_margin must be non-negative.")
        self.margin = margin

    def forward(self, scores: torch.Tensor, base_scores: torch.Tensor) -> torch.Tensor:
        if scores.shape != base_scores.shape:
            raise ValueError("Adapted and base score tensors must have identical shapes.")
        if scores.dim() != 2 or scores.shape[1] < 2:
            raise ValueError("Improvement-margin training requires one positive and at least one negative.")
        adapted_margin = scores[:, :1] - scores[:, 1:]
        base_margin = base_scores[:, :1] - base_scores[:, 1:]
        return torch.relu(self.margin + base_margin - adapted_margin).mean()


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_negatives: int = 63
    patience: int = 7
    seed: int = 42
    improvement_margin: float = 0.05

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.epochs <= 0 or self.num_negatives <= 0:
            raise ValueError("batch_size, epochs, and num_negatives must be positive.")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("lr must be positive and weight_decay must be non-negative.")
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")
        if self.improvement_margin < 0:
            raise ValueError("improvement_margin must be non-negative.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
    dataset: ContrastiveCachedDataset,
    batch_size: int,
    shuffle: bool,
    *,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_contrastive_batch,
        generator=generator,
    )


def _loss_for_batch(
    model: IMRNN,
    query_embeddings: torch.Tensor,
    documents: torch.Tensor,
    objective: ImprovementMarginObjective,
) -> torch.Tensor:
    _, _, adapted_scores = model(query_embeddings, documents)
    base_scores = F.cosine_similarity(documents, query_embeddings.unsqueeze(1), dim=-1)
    return objective(adapted_scores, base_scores)


def evaluate_loss(
    model: IMRNN,
    dataloader: DataLoader,
    device: str,
    objective: ImprovementMarginObjective,
) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in dataloader:
            loss = _loss_for_batch(
                model,
                batch["query_embeddings"].to(device),
                batch["documents"].to(device),
                objective,
            )
            total_loss += loss.item()
            steps += 1
    model.train()
    return total_loss / max(steps, 1)


def train_model(
    model: IMRNN,
    train_dataset: ContrastiveCachedDataset,
    val_dataset: ContrastiveCachedDataset,
    config: TrainingConfig,
    device: str,
    validation_metric_fn: Callable[[IMRNN], float] | None = None,
    validation_metric_name: str = "validation_metric",
) -> dict[str, Any]:
    """Train with improvement margin and restore the best validation epoch."""

    set_seed(config.seed)
    train_loader = build_dataloader(train_dataset, config.batch_size, shuffle=True, seed=config.seed)
    val_loader = build_dataloader(val_dataset, config.batch_size, shuffle=False, seed=config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    objective = ImprovementMarginObjective(config.improvement_margin)
    model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float | None]] = []

    initial_val_loss = evaluate_loss(model, val_loader, device, objective)
    best_value = float(validation_metric_fn(model)) if validation_metric_fn else initial_val_loss
    history.append(
        {
            "epoch": 0.0,
            "train_loss": None,
            "val_loss": initial_val_loss,
            "selection_value": best_value,
        }
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        running_loss = 0.0
        steps = 0
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_batch(
                model,
                batch["query_embeddings"].to(device),
                batch["documents"].to(device),
                objective,
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(steps, 1)
        val_loss = evaluate_loss(model, val_loader, device, objective)
        selection_value = float(validation_metric_fn(model)) if validation_metric_fn else val_loss
        improved = selection_value > best_value if validation_metric_fn else selection_value < best_value
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "selection_value": selection_value,
            }
        )
        if improved:
            best_value = selection_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if config.patience > 0 and patience_counter >= config.patience:
            break

    model.load_state_dict(best_state, strict=True)
    model.eval()
    return {
        "objective": "improvement-margin",
        "best_epoch": best_epoch,
        "best_validation_value": best_value,
        "selection_metric": validation_metric_name if validation_metric_fn else "validation_loss",
        "epochs_completed": len(history) - 1,
        "history": history,
        "seed": config.seed,
        "optimizer": "adam",
        "learning_rate": config.lr,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "configured_epochs": config.epochs,
        "patience": config.patience,
        "number_of_negatives": config.num_negatives,
        "improvement_margin": config.improvement_margin,
    }


def initialize_projector(model: IMRNN) -> None:
    """Initialize the square projector to the identity transform."""

    with torch.no_grad():
        model.projector.weight.copy_(torch.eye(model.config.input_dim))
        model.projector.bias.zero_()
