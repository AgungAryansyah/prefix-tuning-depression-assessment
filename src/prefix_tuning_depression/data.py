"""Data loading utilities for DAIC-WOZ text interviews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from prefix_tuning_depression.preprocessing import preprocess_transcript
from prefix_tuning_depression.splits import SplitName, load_split


@dataclass
class Interview:
    """A single interview with its QR pairs and PHQ-8 label."""

    subject_id: int
    qr_pairs: list[str]
    phq_score: float
    phq_binary: int
    split: SplitName

    def __len__(self) -> int:
        return len(self.qr_pairs)


def _load_labels(labels_path: Path) -> dict[int, tuple[float, int]]:
    """Load PHQ labels from the cleaning report."""
    df = pd.read_csv(labels_path)
    df = df.drop_duplicates("participant_id")
    df = df.dropna(subset=["phq_score", "phq_binary"])
    return {
        int(row["participant_id"]): (float(row["phq_score"]), int(row["phq_binary"]))
        for _, row in df.iterrows()
    }


def load_interviews(
    data_root: Path | str,
    split: SplitName | None = None,
    split_map: dict[int, SplitName] | None = None,
) -> list[Interview]:
    """Load interviews from official DAIC-WOZ transcript CSVs.

    Args:
        data_root: Path to the data directory containing the ``transcript/``
            folder and ``cleaning_report_Transcript.csv``.
        split: If provided, only return interviews from this split.
        split_map: Optional pre-computed split mapping. If None, it is loaded
            from the cleaning report.

    Returns:
        List of Interview objects.
    """
    data_root = Path(data_root)
    labels_path = data_root / "cleaning_report_Transcript.csv"

    if split_map is None:
        split_map = load_split(data_root, labels_path=labels_path)

    labels = _load_labels(labels_path)

    transcript_dir = data_root / "transcript"

    interviews: list[Interview] = []
    for subject_id, subject_split in split_map.items():
        if split is not None and subject_split != split:
            continue

        csv_path = transcript_dir / f"{subject_id}_TRANSCRIPT.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, sep="\t")
        phq_score, phq_binary = labels[subject_id]
        qr_pairs = preprocess_transcript(df, subject_id=subject_id)

        if not qr_pairs:
            continue

        interviews.append(
            Interview(
                subject_id=subject_id,
                qr_pairs=qr_pairs,
                phq_score=phq_score,
                phq_binary=phq_binary,
                split=subject_split,
            )
        )

    return interviews


def print_interview_summary(interviews: list[Interview]) -> None:
    """Print summary statistics for a list of interviews."""
    if not interviews:
        print("No interviews loaded.")
        return

    total_pairs = sum(len(i) for i in interviews)
    avg_pairs = total_pairs / len(interviews)
    print(f"Interviews: {len(interviews)}")
    print(f"Total QR pairs: {total_pairs}")
    print(f"Avg QR pairs per interview: {avg_pairs:.1f}")
    print(f"Min/Max QR pairs: {min(len(i) for i in interviews)}/{max(len(i) for i in interviews)}")
    print(f"Mean PHQ score: {sum(i.phq_score for i in interviews) / len(interviews):.2f}")
