"""Train a depression severity model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prefix_tuning_depression.config import ModelConfig, TrainingConfig
from prefix_tuning_depression.data import load_interviews
from prefix_tuning_depression.dataset import InterviewDataset, build_collator
from prefix_tuning_depression.metrics import aggregate_run_results
from prefix_tuning_depression.models.depression_model import build_depression_model
from prefix_tuning_depression.training import evaluate, train_model


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(
    model_type: str,
    data_root: Path,
    output_dir: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Train one run and return dev metrics."""
    set_seed(seed)

    train_interviews = load_interviews(data_root, split="train")
    dev_interviews = load_interviews(data_root, split="dev")

    train_dataset = InterviewDataset(train_interviews)
    dev_dataset = InterviewDataset(dev_interviews)
    collator = build_collator(model_config, model_type)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = build_depression_model(model_config, model_type)
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        model_type=model_type,
        num_epochs=training_config.num_epochs,
        patience=training_config.es_patience,
        learning_rate=training_config.learning_rate,
        device=device,
        verbose=True,
    )

    # Final dev evaluation.
    criterion = torch.nn.MSELoss()
    dev_results = evaluate(model, dev_loader, criterion, device, model_type)

    # Save checkpoint and history.
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_type}_seed{seed}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    history_path = output_dir / f"{model_type}_seed{seed}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Seed {seed}: RMSE={dev_results['rmse']:.3f}, MAE={dev_results['mae']:.3f}")
    return dev_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a depression severity model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "prefix-only",
            "st-only",
            "dual-encoder",
            "bert-pt",
            "bert-ft1",
            "bert-ft2",
            "roberta-pt",
            "roberta-ft1",
            "roberta-ft2",
        ],
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-epochs", type=int, default=TrainingConfig().num_epochs)
    parser.add_argument("--es-patience", type=int, default=TrainingConfig().es_patience)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig().batch_size)
    args = parser.parse_args()

    device = torch.device(args.device)
    model_config = ModelConfig()
    training_config = TrainingConfig().replace(
        num_epochs=args.num_epochs,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
    )

    results: list[dict[str, float]] = []
    for seed in args.seeds:
        print(f"\n=== Run {args.model} with seed {seed} ===")
        run_result = run_training(
            model_type=args.model,
            data_root=args.data_root,
            output_dir=args.output_dir,
            model_config=model_config,
            training_config=training_config,
            device=device,
            seed=seed,
        )
        results.append(run_result)

    aggregated = aggregate_run_results(results)
    print("\n=== Aggregated dev results ===")
    print(f"RMSE: {aggregated['rmse_mean']:.3f} ± {aggregated['rmse_std']:.3f}")
    print(f"MAE:  {aggregated['mae_mean']:.3f} ± {aggregated['mae_std']:.3f}")


if __name__ == "__main__":
    main()
