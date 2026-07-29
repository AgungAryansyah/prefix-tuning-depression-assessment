"""Run the paper's low-data comparison experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prefix_tuning_depression.config import ModelConfig, TrainingConfig
from prefix_tuning_depression.data import Interview, load_interviews
from prefix_tuning_depression.dataset import InterviewDataset, build_collator
from prefix_tuning_depression.experiments import best_run, stratified_train_subset
from prefix_tuning_depression.models.depression_model import build_depression_model
from prefix_tuning_depression.splits import (
    load_official_avec2017_contract,
    require_complete_official_transcript_coverage,
)
from prefix_tuning_depression.training import evaluate
from train import run_training

MODELS = ("prefix-only", "bert-ft1", "roberta-ft1")


def evaluate_checkpoint(
    checkpoint_path: Path,
    model_type: str,
    model_config: ModelConfig,
    test_interviews: list[Interview],
    device: torch.device,
    num_workers: int,
) -> dict[str, float]:
    """Evaluate one selected checkpoint on the official test partition."""
    dataset = InterviewDataset(test_interviews)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=build_collator(model_config, model_type),
        num_workers=num_workers,
    )
    model = build_depression_model(model_config, model_type).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return evaluate(model, dataloader, torch.nn.MSELoss(), device, model_type)


def select_prefix_length(
    prefix_lengths: list[int],
    train_interviews: list[Interview],
    dev_interviews: list[Interview],
    manifest_dir: Path,
    output_dir: Path,
    training_config: TrainingConfig,
    contract,
    device: torch.device,
    seed: int,
    device_ids: list[int] | None,
) -> int:
    """Select prefix length with one development run, as in the paper."""
    candidates: list[tuple[int, dict[str, float]]] = []
    for prefix_length in prefix_lengths:
        model_config = replace(ModelConfig(), pre_seq_len=prefix_length)
        result = run_training(
            model_type="prefix-only",
            manifest_dir=manifest_dir,
            output_dir=output_dir / f"prefix_length_{prefix_length}",
            model_config=model_config,
            training_config=training_config,
            device=device,
            seed=seed,
            contract=contract,
            train_interviews=train_interviews,
            dev_interviews=dev_interviews,
            device_ids=device_ids,
        )
        candidates.append((prefix_length, result))
    return min(candidates, key=lambda candidate: candidate[1]["loss"])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DAIC-WOZ low-data experiment")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("low_data_results"))
    parser.add_argument("--percentages", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-epochs", type=int, default=TrainingConfig().num_epochs)
    parser.add_argument("--es-patience", type=int, default=TrainingConfig().es_patience)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig().batch_size)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig().num_workers)
    parser.add_argument("--device-ids", type=int, nargs="+", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    contract = load_official_avec2017_contract(args.manifest_dir)
    require_complete_official_transcript_coverage(contract, args.manifest_dir)
    train_interviews = load_interviews(args.manifest_dir, split="train", contract=contract)
    dev_interviews = load_interviews(args.manifest_dir, split="dev", contract=contract)
    test_interviews = load_interviews(args.manifest_dir, split="test", contract=contract)
    training_config = TrainingConfig().replace(
        num_epochs=args.num_epochs,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    all_results: dict[str, dict] = {}
    for percentage in args.percentages:
        subset = stratified_train_subset(train_interviews, percentage, args.subset_seed)
        percentage_dir = args.output_dir / f"{percentage}pct"
        prefix_length = select_prefix_length(
            args.prefix_lengths,
            subset,
            dev_interviews,
            args.manifest_dir,
            percentage_dir / "selection",
            training_config,
            contract,
            device,
            args.selection_seed,
            args.device_ids,
        )
        percentage_results: dict[str, object] = {"prefix_length": prefix_length}

        for model_type in MODELS:
            model_config = replace(
                ModelConfig(),
                pre_seq_len=prefix_length if model_type == "prefix-only" else 10,
            )
            model_dir = percentage_dir / model_type
            runs: list[dict[str, float]] = []
            for seed in args.seeds:
                result = run_training(
                    model_type=model_type,
                    manifest_dir=args.manifest_dir,
                    output_dir=model_dir,
                    model_config=model_config,
                    training_config=training_config,
                    device=device,
                    seed=seed,
                    contract=contract,
                    train_interviews=subset,
                    dev_interviews=dev_interviews,
                    device_ids=args.device_ids,
                )
                runs.append({**result, "seed": seed})

            selected = best_run(runs)
            checkpoint = model_dir / f"{model_type}_seed{int(selected['seed'])}.pt"
            percentage_results[model_type] = {
                "dev_runs": runs,
                "test": evaluate_checkpoint(
                    checkpoint,
                    model_type,
                    model_config,
                    test_interviews,
                    device,
                    args.num_workers,
                ),
            }
        all_results[str(percentage)] = percentage_results

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "low_data_results.json").open("w") as file:
        json.dump(all_results, file, indent=2)


if __name__ == "__main__":
    main()
