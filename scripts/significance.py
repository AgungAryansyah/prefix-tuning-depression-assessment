"""Compare saved model predictions with the paper's ANOVA protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prefix_tuning_depression.metrics import pairwise_error_anova


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["labels"], data["predictions"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pairwise error ANOVA")
    parser.add_argument("--reference", required=True, type=_named_path)
    parser.add_argument("--candidate", required=True, type=_named_path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference_name, reference_path = args.reference
    labels, reference_predictions = _load_predictions(reference_path)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for candidate_name, candidate_path in args.candidate:
        candidate_labels, candidate_predictions = _load_predictions(candidate_path)
        if not np.array_equal(labels, candidate_labels):
            raise ValueError(f"labels differ between {reference_name} and {candidate_name}")
        results[candidate_name] = pairwise_error_anova(
            labels, reference_predictions, candidate_predictions
        )

    output = json.dumps({"reference": reference_name, "comparisons": results}, indent=2)
    if args.output is None:
        print(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
