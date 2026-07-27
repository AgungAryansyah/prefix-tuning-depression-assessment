"""Training loop for depression severity models."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from prefix_tuning_depression.metrics import compute_mae, compute_rmse


class EarlyStopping:
    """Stop training if validation loss does not improve for N epochs."""

    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        labels = batch["labels"].to(device)
        interview_lengths = batch["interview_lengths"].to(device)

        optimizer.zero_grad()
        logits = _forward_batch(model, batch, interview_lengths, model_type, device)
        loss = criterion(logits.view(-1), labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
) -> dict[str, float]:
    """Evaluate model and return loss + metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in dataloader:
        labels = batch["labels"].to(device)
        interview_lengths = batch["interview_lengths"].to(device)

        logits = _forward_batch(model, batch, interview_lengths, model_type, device)
        loss = criterion(logits.view(-1), labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.view(-1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        all_preds.append(np.atleast_1d(preds))
        all_labels.append(np.atleast_1d(labels_np))

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    return {
        "loss": total_loss / total_samples,
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
    }


def _forward_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    interview_lengths: torch.Tensor,
    model_type: str,
    device: torch.device,
) -> torch.Tensor:
    """Dispatch forward call based on model type."""
    match model_type:
        case "st-only" | "dual-encoder":
            st_inputs = batch["st_inputs"].to(device)
            if model_type == "st-only":
                return model(st_inputs, interview_lengths)
            prefix_inputs = batch["prefix_inputs"].to(device)
            return model(st_inputs, prefix_inputs, interview_lengths)
        case "prefix-only" | "bert-pt" | "bert-ft1" | "bert-ft2" | "roberta-pt" | "roberta-ft1" | "roberta-ft2":
            inputs = batch["inputs"].to(device)
            return model(inputs, interview_lengths)
        case _:
            raise ValueError(f"Unknown model type in forward: {model_type}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    model_type: str,
    num_epochs: int = 200,
    patience: int = 20,
    learning_rate: float = 3e-4,
    device: torch.device | None = None,
    device_ids: list[int] | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train a model with early stopping and return the best model + history.

    If ``device_ids`` contains more than one GPU id, the model is wrapped in
    ``nn.DataParallel`` for the duration of training and unwrapped before it
    is returned, so the saved state dicts are compatible with single-GPU
    checkpoints.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    use_dp = device_ids is not None and len(device_ids) > 1
    if use_dp:
        model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "dev_loss": [],
        "dev_rmse": [],
        "dev_mae": [],
    }

    best_state: dict | None = None

    pbar = tqdm(total=num_epochs, desc="Training") if verbose else None

    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, model_type
        )
        dev_results = evaluate(model, dev_loader, criterion, device, model_type)

        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_results["loss"])
        history["dev_rmse"].append(dev_results["rmse"])
        history["dev_mae"].append(dev_results["mae"])

        improved = early_stopping.step(dev_results["loss"])
        if improved:
            source = model.module if use_dp else model
            best_state = {k: v.cpu().clone() for k, v in source.state_dict().items()}

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "train_loss": f"{train_loss:.3f}",
                    "dev_loss": f"{dev_results['loss']:.3f}",
                    "dev_rmse": f"{dev_results['rmse']:.3f}",
                }
            )

        if early_stopping.should_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if use_dp:
        model = model.module

    return model, history
