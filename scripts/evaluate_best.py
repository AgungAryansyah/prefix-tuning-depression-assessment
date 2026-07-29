"""Select the best development run and evaluate it once on official test data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prefix_tuning_depression.artifacts import write_json
from prefix_tuning_depression.config import ModelConfig, TrainingConfig
from prefix_tuning_depression.data import load_interviews
from prefix_tuning_depression.dataset import InterviewDataset, build_collator
from prefix_tuning_depression.evaluation import select_best_checkpoint
from prefix_tuning_depression.models.depression_model import FUSION_METHODS, build_depression_model
from prefix_tuning_depression.splits import (
    load_official_avec2017_contract,
    require_complete_official_transcript_coverage,
)
from prefix_tuning_depression.training import collect_predictions, evaluate

MODEL_TYPES = (
    "prefix-only",
    "st-only",
    "dual-encoder",
    "bert-pt",
    "bert-ft1",
    "bert-ft2",
    "roberta-pt",
    "roberta-ft1",
    "roberta-ft2",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the best development checkpoint")
    parser.add_argument("--model", choices=MODEL_TYPES, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete-coverage", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--fusion-method", choices=FUSION_METHODS, default="average")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=TrainingConfig().num_workers)
    parser.add_argument("--device-ids", type=int, nargs="+", default=None)
    args = parser.parse_args()

    model_config = replace(ModelConfig(), fusion_method=args.fusion_method)
    contract = load_official_avec2017_contract(args.manifest_dir)
    if not args.allow_incomplete_coverage:
        require_complete_official_transcript_coverage(contract, args.manifest_dir)
    selected = select_best_checkpoint(
        args.checkpoint_dir, args.model, model_config, args.seeds
    )

    device = torch.device(args.device)
    test_interviews = load_interviews(
        args.manifest_dir,
        split="test",
        contract=contract,
        allow_incomplete_coverage=args.allow_incomplete_coverage,
    )
    dataloader = DataLoader(
        InterviewDataset(test_interviews),
        batch_size=2,
        shuffle=False,
        collate_fn=build_collator(model_config, args.model),
        num_workers=args.num_workers,
    )
    model = build_depression_model(model_config, args.model).to(device)
    model.load_state_dict(
        torch.load(selected.checkpoint_path, map_location=device, weights_only=True)
    )
    if args.device_ids is not None and len(args.device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    test_results = evaluate(model, dataloader, torch.nn.MSELoss(), device, args.model)
    labels, predictions = collect_predictions(model, dataloader, device, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{args.model}_{args.fusion_method}" if args.model == "dual-encoder" else args.model
    prediction_path = args.output_dir / f"{artifact_name}_best_test_predictions.npz"
    np.savez(prediction_path, labels=labels, predictions=predictions)

    metadata = {}
    if selected.metadata_path.exists():
        metadata = json.loads(selected.metadata_path.read_text())
    result = {
        "model_type": args.model,
        "selected_seed": selected.seed,
        "selected_epoch": selected.epoch,
        "selected_checkpoint": str(selected.checkpoint_path),
        "selected_metadata": str(selected.metadata_path),
        "development": {
            "loss": selected.dev_loss,
            "rmse": selected.dev_rmse,
            "mae": selected.dev_mae,
        },
        "test": test_results,
        "predictions": str(prediction_path),
        "run_metadata": metadata,
        "model_config": asdict(model_config),
    }
    write_json(args.output_dir / f"{artifact_name}_best_test.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
