"""Selection helpers for paper-style development and test evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from prefix_tuning_depression.artifacts import run_name
from prefix_tuning_depression.config import ModelConfig


@dataclass(frozen=True)
class BestCheckpoint:
    """One checkpoint selected by minimum development loss."""

    seed: int
    epoch: int
    checkpoint_path: Path
    metadata_path: Path
    dev_loss: float
    dev_rmse: float
    dev_mae: float


def select_best_checkpoint(
    checkpoint_dir: Path | str,
    model_type: str,
    model_config: ModelConfig,
    seeds: list[int],
) -> BestCheckpoint:
    """Select one run across seeds without inspecting test data."""
    checkpoint_dir = Path(checkpoint_dir)
    artifact_name = run_name(model_type, model_config)
    candidates: list[BestCheckpoint] = []
    for seed in seeds:
        history_path = checkpoint_dir / f"{artifact_name}_seed{seed}_history.json"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing training history: {history_path}")
        history = json.loads(history_path.read_text())
        if not history.get("dev_loss"):
            raise ValueError(f"Training history has no development results: {history_path}")
        epoch = min(range(len(history["dev_loss"])), key=history["dev_loss"].__getitem__)
        checkpoint_path = checkpoint_dir / f"{artifact_name}_seed{seed}.pt"
        metadata_path = checkpoint_dir / f"{artifact_name}_seed{seed}_metadata.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
        candidates.append(
            BestCheckpoint(
                seed=seed,
                epoch=epoch,
                checkpoint_path=checkpoint_path,
                metadata_path=metadata_path,
                dev_loss=float(history["dev_loss"][epoch]),
                dev_rmse=float(history["dev_rmse"][epoch]),
                dev_mae=float(history["dev_mae"][epoch]),
            )
        )
    return min(candidates, key=lambda candidate: (candidate.dev_loss, candidate.seed))
