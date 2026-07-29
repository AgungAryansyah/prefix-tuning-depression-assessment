"""Reproducibility metadata for training and evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from prefix_tuning_depression.config import ModelConfig, TrainingConfig
from prefix_tuning_depression.data import Interview
from prefix_tuning_depression.splits import OfficialDaicWozContract

MANIFEST_FILES = (
    "train_split_Depression_AVEC2017.csv",
    "dev_split_Depression_AVEC2017.csv",
    "test_split_Depression_AVEC2017.csv",
    "full_test_split.csv",
)


def run_name(model_type: str, model_config: ModelConfig) -> str:
    """Return the stable artifact prefix used by training and evaluation."""
    if model_type == "dual-encoder":
        return f"{model_type}_{model_config.fusion_method}"
    return model_type


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_metadata(
    manifest_dir: Path,
    contract: OfficialDaicWozContract,
    model_type: str,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    seed: int,
    train_interviews: list[Interview],
    dev_interviews: list[Interview],
    device: str,
) -> dict[str, Any]:
    """Build a self-contained record for one training run."""
    manifest_dir = Path(manifest_dir)
    transcript_ids = sorted(contract.split_map)
    return {
        "model_type": model_type,
        "run_name": run_name(model_type, model_config),
        "seed": seed,
        "device": device,
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "manifest_sha256": {
            name: sha256_file(manifest_dir / name) for name in MANIFEST_FILES
        },
        "transcript_sha256": {
            str(subject_id): sha256_file(
                manifest_dir / f"{subject_id}_TRANSCRIPT.csv"
            )
            for subject_id in transcript_ids
        },
        "split_subject_ids": {
            split: sorted(
                subject_id
                for subject_id, subject_split in contract.split_map.items()
                if subject_split == split
            )
            for split in ("train", "dev", "test")
        },
        "train_subject_ids": [interview.subject_id for interview in train_interviews],
        "dev_subject_ids": [interview.subject_id for interview in dev_interviews],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write an indented JSON artifact."""
    path.write_text(json.dumps(value, indent=2) + "\n")
