"""Data loading utilities for DAIC-WOZ text interviews."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from prefix_tuning_depression.preprocessing import preprocess_transcript
from prefix_tuning_depression.splits import (
    OfficialDaicWozContract,
    SplitName,
    load_official_avec2017_contract,
    require_complete_official_transcript_coverage,
)


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


def load_interviews(
    manifest_dir: Path | str,
    split: SplitName | None = None,
    contract: OfficialDaicWozContract | None = None,
) -> list[Interview]:
    """Load official DAIC-WOZ interviews from an AVEC manifest directory.

    Args:
        manifest_dir: Directory containing raw transcript CSVs and the four
            validated AVEC manifest files.
        split: If provided, only return interviews from this split.
        contract: Optional preloaded official manifest contract.

    Returns:
        List of Interview objects.
    """
    manifest_dir = Path(manifest_dir)
    if contract is None:
        contract = load_official_avec2017_contract(manifest_dir)
        require_complete_official_transcript_coverage(contract, manifest_dir)

    interviews: list[Interview] = []
    for subject_id, subject_split in contract.split_map.items():
        if split is not None and subject_split != split:
            continue

        csv_path = manifest_dir / f"{subject_id}_TRANSCRIPT.csv"

        df = pd.read_csv(csv_path, sep="\t")
        phq_score, phq_binary = contract.labels[subject_id]
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
