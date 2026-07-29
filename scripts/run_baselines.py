"""Run all baseline models and report aggregated dev results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from prefix_tuning_depression.config import TrainingConfig

MODELS = [
    "bert-pt",
    "roberta-pt",
    "st-only",
    "bert-ft1",
    "bert-ft2",
    "roberta-ft1",
    "roberta-ft2",
    "prefix-only",
]


def run_model(
    model: str,
    seeds: list[int],
    manifest_dir: Path,
    output_dir: Path,
    device: str,
    num_workers: int,
    allow_incomplete_coverage: bool,
) -> dict:
    """Train one model across seeds and load aggregated results."""
    print(f"\n=== Running {model} ===")
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--model",
        model,
        "--manifest-dir",
        str(manifest_dir),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
        "--seeds",
    ] + [str(s) for s in seeds]
    if allow_incomplete_coverage:
        cmd.append("--allow-incomplete-coverage")

    subprocess.run(cmd, check=True)

    # Load per-seed results from history files.
    results = []
    for seed in seeds:
        history_path = output_dir / f"{model}_seed{seed}_history.json"
        with open(history_path) as f:
            history = json.load(f)
        # Best dev result is the minimum loss entry.
        best_idx = min(range(len(history["dev_loss"])), key=lambda i: history["dev_loss"][i])
        results.append(
            {
                "rmse": history["dev_rmse"][best_idx],
                "mae": history["dev_mae"][best_idx],
            }
        )

    rmse_values = [r["rmse"] for r in results]
    mae_values = [r["mae"] for r in results]
    return {
        "rmse_mean": sum(rmse_values) / len(rmse_values),
        "rmse_std": (sum((x - sum(rmse_values) / len(rmse_values)) ** 2 for x in rmse_values) / len(rmse_values)) ** 0.5,
        "mae_mean": sum(mae_values) / len(mae_values),
        "mae_std": (sum((x - sum(mae_values) / len(mae_values)) ** 2 for x in mae_values) / len(mae_values)) ** 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all baseline models")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete-coverage", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--models", type=str, nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig().num_workers)
    args = parser.parse_args()

    all_results: dict[str, dict] = {}
    for model in args.models:
        all_results[model] = run_model(
            model,
            args.seeds,
            args.manifest_dir,
            args.output_dir,
            args.device,
            args.num_workers,
            args.allow_incomplete_coverage,
        )

    print("\n=== Aggregated dev results ===")
    print(f"{'Model':<16} {'RMSE':>12} {'MAE':>12}")
    print("-" * 42)
    for model, res in all_results.items():
        print(
            f"{model:<16} {res['rmse_mean']:>5.2f}±{res['rmse_std']:<4.2f} "
            f"{res['mae_mean']:>5.2f}±{res['mae_std']:<4.2f}"
        )

    summary_path = args.output_dir / "baseline_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
