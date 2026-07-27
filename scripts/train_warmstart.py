"""Train a warm-started dual encoder model.

First trains a prefix-only model, then initializes a dual encoder from it and
continues training with the frozen prefix branch plus trainable ST projection
and interview-level layers.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prefix_tuning_depression.config import ModelConfig, TrainingConfig
from prefix_tuning_depression.data import load_interviews
from prefix_tuning_depression.dataset import InterviewDataset, build_collator
from prefix_tuning_depression.metrics import aggregate_run_results
from prefix_tuning_depression.models.depression_model import (
    build_depression_model,
    build_warmstarted_dual_encoder,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train warm-started dual encoder")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-epochs", type=int, default=TrainingConfig().num_epochs)
    parser.add_argument("--es-patience", type=int, default=TrainingConfig().es_patience)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig().batch_size)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig().num_workers)
    parser.add_argument("--device-ids", type=int, nargs="+", default=None,
                        help="GPU device IDs for DataParallel; single id means no parallelism")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_config = ModelConfig()
    training_config = TrainingConfig().replace(
        num_epochs=args.num_epochs,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results: list[dict[str, float]] = []
    for seed in args.seeds:
        print(f"\n=== Warm-start run with seed {seed} ===")
        set_seed(seed)

        train_interviews = load_interviews(args.data_root, split="train")
        dev_interviews = load_interviews(args.data_root, split="dev")

        train_dataset = InterviewDataset(train_interviews)
        dev_dataset = InterviewDataset(dev_interviews)

        prefix_collator = build_collator(model_config, "prefix-only")
        prefix_train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            collate_fn=prefix_collator,
            num_workers=training_config.num_workers,
        )
        prefix_dev_loader = DataLoader(
            dev_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            collate_fn=prefix_collator,
            num_workers=training_config.num_workers,
        )

        dual_collator = build_collator(model_config, "dual-encoder")
        dual_train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            collate_fn=dual_collator,
            num_workers=training_config.num_workers,
        )
        dual_dev_loader = DataLoader(
            dev_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            collate_fn=dual_collator,
            num_workers=training_config.num_workers,
        )

        # Step 1: train prefix-only model.
        print("Training prefix-only model...")
        prefix_model = build_depression_model(model_config, "prefix-only").to(device)
        prefix_model, _ = train_model(
            model=prefix_model,
            train_loader=prefix_train_loader,
            dev_loader=prefix_dev_loader,
            model_type="prefix-only",
            num_epochs=training_config.num_epochs,
            patience=training_config.es_patience,
            learning_rate=training_config.learning_rate,
            device=device,
            device_ids=args.device_ids,
            verbose=True,
        )
        prefix_checkpoint = args.output_dir / f"warmstart_prefix_seed{seed}.pt"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(prefix_model.state_dict(), prefix_checkpoint)

        # Step 2: initialize dual encoder from prefix model and train.
        print("Training dual encoder from warm-start...")
        dual_model = build_warmstarted_dual_encoder(
            model_config, str(prefix_checkpoint), device
        )
        dual_model, _ = train_model(
            model=dual_model,
            train_loader=dual_train_loader,
            dev_loader=dual_dev_loader,
            model_type="dual-encoder",
            num_epochs=training_config.num_epochs,
            patience=training_config.es_patience,
            learning_rate=training_config.learning_rate,
            device=device,
            device_ids=args.device_ids,
            verbose=True,
        )
        dual_checkpoint = args.output_dir / f"dual_encoder_warmstart_seed{seed}.pt"
        torch.save(dual_model.state_dict(), dual_checkpoint)

        criterion = torch.nn.MSELoss()
        dev_results = evaluate(dual_model, dual_dev_loader, criterion, device, "dual-encoder")
        print(
            f"Seed {seed}: RMSE={dev_results['rmse']:.3f}, MAE={dev_results['mae']:.3f}"
        )
        results.append(dev_results)

    aggregated = aggregate_run_results(results)
    print("\n=== Aggregated warm-start dual encoder dev results ===")
    print(f"RMSE: {aggregated['rmse_mean']:.3f} ± {aggregated['rmse_std']:.3f}")
    print(f"MAE:  {aggregated['mae_mean']:.3f} ± {aggregated['mae_std']:.3f}")


if __name__ == "__main__":
    main()
