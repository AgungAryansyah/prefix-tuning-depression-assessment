"""Evaluate a saved checkpoint on the dev set."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prefix_tuning_depression.config import ModelConfig, TrainingConfig
from prefix_tuning_depression.data import load_interviews
from prefix_tuning_depression.dataset import InterviewDataset, build_collator
from prefix_tuning_depression.models.depression_model import build_depression_model
from prefix_tuning_depression.training import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=TrainingConfig().num_workers)
    parser.add_argument("--device-ids", type=int, nargs="+", default=None,
                        help="GPU device IDs for DataParallel; single id means no parallelism")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_config = ModelConfig()

    interviews = load_interviews(args.data_root, split=args.split)
    dataset = InterviewDataset(interviews)
    collator = build_collator(model_config, args.model)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    model = build_depression_model(model_config, args.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.device_ids is not None and len(args.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    criterion = torch.nn.MSELoss()
    results = evaluate(model, dataloader, criterion, device, args.model)

    print(f"{args.model} on {args.split}:")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE:  {results['mae']:.4f}")


if __name__ == "__main__":
    main()
