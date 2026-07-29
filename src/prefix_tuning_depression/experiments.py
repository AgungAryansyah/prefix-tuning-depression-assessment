"""Utilities shared by paper-reproduction experiment runners."""

from __future__ import annotations

from typing import cast

from sklearn.model_selection import train_test_split

from prefix_tuning_depression.data import Interview


def stratified_train_subset(
    interviews: list[Interview],
    percentage: int,
    random_state: int,
) -> list[Interview]:
    """Sample a PHQ-binary-stratified training subset at ``percentage``."""
    if not 0 < percentage <= 100:
        raise ValueError("percentage must be in 1..100")
    if percentage == 100:
        return list(interviews)
    subset, _ = train_test_split(
        interviews,
        train_size=percentage / 100,
        stratify=[interview.phq_binary for interview in interviews],
        random_state=random_state,
    )
    return cast(list[Interview], subset)


def best_run(results: list[dict[str, float]]) -> dict[str, float]:
    """Return the run selected by minimum development loss."""
    if not results:
        raise ValueError("at least one run is required")
    return min(results, key=lambda result: result["loss"])
