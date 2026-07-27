"""DAIC-WOZ train/dev/test split handling.

The paper uses the AVEC 2017 split (107/35/47) over the 189 DAIC-WOZ sessions.
Transcripts are read from ``data/transcript/``.

This module first looks for an official AVEC 2017 split file, and falls back to
a deterministic stratified split that preserves the original 57/19/25 ratio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import pandas as pd
from sklearn.model_selection import train_test_split

SplitName = Literal["train", "dev", "test"]

# Sessions present in DAIC-WOZ (300-492) but omitted by the upstream cleaner.
CLEANER_EXCLUDED_SESSIONS: frozenset[int] = frozenset(
    {342, 394, 398, 460, 373, 444, 451, 458, 480, 402}
)

# Technical exclusions from the original DAIC-WOZ documentation.
TECHNICAL_EXCLUSIONS: frozenset[int] = frozenset({342, 394, 398, 460})


def _default_daicwoz_session_ids() -> list[int]:
    """Return the 189 DAIC-WOZ session IDs (300-492 minus technical exclusions)."""
    return [sid for sid in range(300, 493) if sid not in TECHNICAL_EXCLUSIONS]


def _available_session_ids(data_root: Path) -> list[int]:
    """Return session IDs that have transcript files on disk."""
    transcript_dir = data_root / "transcript"
    if not transcript_dir.exists():
        return []
    ids: set[int] = set()
    for path in transcript_dir.glob("*_TRANSCRIPT.csv"):
        sid = int(path.stem.split("_")[0])
        if 300 <= sid <= 492:
            ids.add(sid)
    return sorted(ids)


def _load_official_split(path: Path) -> dict[int, SplitName]:
    """Load an official split CSV with columns Participant_ID and split."""
    df = pd.read_csv(path)
    return {int(row["Participant_ID"]): row["split"] for _, row in df.iterrows()}


def _make_stratified_split(session_ids: list[int], labels: pd.DataFrame) -> dict[int, SplitName]:
    """Create a deterministic stratified split matching the 57/19/25 ratio."""
    labels = labels.set_index("participant_id").loc[session_ids].reset_index()
    stratify = labels["phq_binary"].astype(str)

    n_total = len(session_ids)
    test_size = max(1, round(n_total * 0.25))
    train_dev_size = n_total - test_size
    dev_size = max(1, round(train_dev_size * 0.25))
    train_size = train_dev_size - dev_size

    train_dev_ids, test_ids = train_test_split(
        session_ids,
        train_size=train_dev_size,
        test_size=test_size,
        stratify=stratify,
        random_state=42,
    )
    train_ids, dev_ids = train_test_split(
        train_dev_ids,
        train_size=train_size,
        test_size=dev_size,
        stratify=stratify.loc[labels["participant_id"].isin(train_dev_ids)],
        random_state=42,
    )

    split_map: dict[int, SplitName] = {}
    for sid in train_ids:
        split_map[sid] = "train"
    for sid in dev_ids:
        split_map[sid] = "dev"
    for sid in test_ids:
        split_map[sid] = "test"
    return split_map


def load_split(
    data_root: Path | str,
    labels_path: Path | str | None = None,
    official_split_path: Path | str | None = None,
) -> dict[int, SplitName]:
    """Return {session_id: split} for DAIC-WOZ sessions.

    Args:
        data_root: Path to the cleaned transcript data directory.
        labels_path: Optional path to a CSV with columns
            participant_id, phq_score, phq_binary. Defaults to
            data_root/../cleaning_report_Transcript.csv.
        official_split_path: Optional path to the official AVEC 2017 split file
            with columns Participant_ID and split. If provided, it is used
            directly for sessions that exist in the cleaned data.
    """
    data_root = Path(data_root)
    if labels_path is None:
        labels_path = data_root / "cleaning_report_Transcript.csv"
    labels_path = Path(labels_path)

    available_ids = _available_session_ids(data_root)
    if not available_ids:
        raise FileNotFoundError(f"No cleaned DAIC-WOZ transcripts found in {data_root}")

    labels = pd.read_csv(labels_path)
    labels = labels[labels["participant_id"].isin(available_ids)].copy()

    if official_split_path is not None:
        official_split_path = Path(official_split_path)
        if official_split_path.exists():
            official = _load_official_split(official_split_path)
            split_map = {sid: official[sid] for sid in available_ids if sid in official}
            if len(split_map) == len(available_ids):
                return cast(dict[int, SplitName], split_map)

    return _make_stratified_split(available_ids, labels)


def print_split_summary(split_map: dict[int, SplitName], labels_path: Path | str) -> None:
    """Print a short summary of the split distribution."""
    labels = pd.read_csv(labels_path)
    labels = labels[labels["participant_id"].isin(split_map)].copy()
    labels["split"] = labels["participant_id"].map(split_map).astype(str)

    summary = labels.groupby("split").agg(
        n=("participant_id", "count"),
        depressed=("phq_binary", "sum"),
        mean_phq=("phq_score", "mean"),
    )
    print(summary)
