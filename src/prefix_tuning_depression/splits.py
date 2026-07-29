"""Official DAIC-WOZ AVEC 2017 manifests and transcript coverage checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

SplitName = Literal["train", "dev", "test"]

# Technical exclusions from the original DAIC-WOZ documentation.
TECHNICAL_EXCLUSIONS: frozenset[int] = frozenset({342, 394, 398, 460})
CANONICAL_DAIC_WOZ_IDS: frozenset[int] = frozenset(
    sid for sid in range(300, 493) if sid not in TECHNICAL_EXCLUSIONS
)


class OfficialManifestError(ValueError):
    """Raised when the AVEC 2017 manifests do not define DAIC-WOZ exactly."""


class OfficialTranscriptCoverageError(ValueError):
    """Raised when official subjects cannot form the required QR inputs."""


@dataclass(frozen=True)
class OfficialDaicWozContract:
    """Validated AVEC 2017 split membership and PHQ-8 labels."""

    split_map: dict[int, SplitName]
    labels: dict[int, tuple[float, int]]


@dataclass(frozen=True)
class OfficialTranscriptCoverage:
    """Raw transcript availability and Ellie-turn coverage for official subjects."""

    missing_transcript_ids: frozenset[int]
    missing_ellie_ids: frozenset[int]

    @property
    def is_complete(self) -> bool:
        return not self.missing_transcript_ids and not self.missing_ellie_ids


def _read_manifest(path: Path, id_column: str, label_columns: tuple[str, str] | None = None) -> tuple[set[int], dict[int, tuple[float, int]]]:
    df = pd.read_csv(path)
    required = [id_column]
    if label_columns is not None:
        required.extend(label_columns)
    missing = set(required) - set(df.columns)
    if missing:
        raise OfficialManifestError(f"{path.name} is missing columns: {sorted(missing)}")
    if df[id_column].isna().any() or df[id_column].duplicated().any():
        raise OfficialManifestError(f"{path.name} has missing or duplicate participant IDs")

    ids = set(df[id_column].astype(int))
    if label_columns is None:
        return ids, {}

    score_column, binary_column = label_columns
    if df[[score_column, binary_column]].isna().any().any():
        raise OfficialManifestError(f"{path.name} has missing PHQ-8 labels")
    labels = {
        int(row[id_column]): (float(row[score_column]), int(row[binary_column]))
        for _, row in df.iterrows()
    }
    return ids, labels


def load_official_avec2017_contract(manifest_dir: Path | str) -> OfficialDaicWozContract:
    """Load and validate the official DAIC-WOZ AVEC 2017 manifests.

    ``manifest_dir`` must contain the three official split manifests and the
    labelled test sidecar. This loader intentionally does not consult the
    mixed E-DAIC cleaning report.
    """
    manifest_dir = Path(manifest_dir)
    train_ids, train_labels = _read_manifest(
        manifest_dir / "train_split_Depression_AVEC2017.csv",
        "Participant_ID",
        ("PHQ8_Score", "PHQ8_Binary"),
    )
    dev_ids, dev_labels = _read_manifest(
        manifest_dir / "dev_split_Depression_AVEC2017.csv",
        "Participant_ID",
        ("PHQ8_Score", "PHQ8_Binary"),
    )
    test_ids, _ = _read_manifest(
        manifest_dir / "test_split_Depression_AVEC2017.csv", "participant_ID"
    )
    test_label_ids, test_labels = _read_manifest(
        manifest_dir / "full_test_split.csv",
        "Participant_ID",
        ("PHQ_Score", "PHQ_Binary"),
    )

    splits = {"train": train_ids, "dev": dev_ids, "test": test_ids}
    expected_counts = {"train": 107, "dev": 35, "test": 47}
    actual_counts = {name: len(ids) for name, ids in splits.items()}
    if actual_counts != expected_counts:
        raise OfficialManifestError(
            f"Expected AVEC split counts {expected_counts}, found {actual_counts}"
        )
    if len(train_ids | dev_ids | test_ids) != sum(actual_counts.values()):
        raise OfficialManifestError("AVEC split manifests overlap")
    if train_ids | dev_ids | test_ids != CANONICAL_DAIC_WOZ_IDS:
        raise OfficialManifestError("AVEC split manifests do not cover canonical DAIC-WOZ IDs")
    if test_label_ids != test_ids:
        raise OfficialManifestError("full_test_split.csv does not match the official test manifest")

    split_map: dict[int, SplitName] = {
        **{subject_id: "train" for subject_id in train_ids},
        **{subject_id: "dev" for subject_id in dev_ids},
        **{subject_id: "test" for subject_id in test_ids},
    }
    return OfficialDaicWozContract(
        split_map=split_map,
        labels={**train_labels, **dev_labels, **test_labels},
    )


def inspect_official_transcript_coverage(
    contract: OfficialDaicWozContract,
    transcript_dir: Path | str,
) -> OfficialTranscriptCoverage:
    """Report whether every official subject can form Ellie-participant QR pairs."""
    transcript_dir = Path(transcript_dir)
    available_ids = {
        int(path.stem.split("_")[0])
        for path in transcript_dir.glob("*_TRANSCRIPT.csv")
    }
    expected_ids = set(contract.split_map)
    missing_transcript_ids = expected_ids - available_ids
    missing_ellie_ids = {
        subject_id
        for subject_id in expected_ids & available_ids
        if not pd.read_csv(
            transcript_dir / f"{subject_id}_TRANSCRIPT.csv",
            sep="\t",
            usecols=["speaker"],
        )["speaker"].eq("Ellie").any()
    }
    return OfficialTranscriptCoverage(
        missing_transcript_ids=frozenset(missing_transcript_ids),
        missing_ellie_ids=frozenset(missing_ellie_ids),
    )


def require_complete_official_transcript_coverage(
    contract: OfficialDaicWozContract,
    transcript_dir: Path | str,
) -> None:
    """Reject a QR-only reproduction with incomplete official transcript coverage."""
    coverage = inspect_official_transcript_coverage(contract, transcript_dir)
    if coverage.is_complete:
        return

    issues = []
    if coverage.missing_transcript_ids:
        issues.append(f"missing transcripts {sorted(coverage.missing_transcript_ids)}")
    if coverage.missing_ellie_ids:
        issues.append(f"missing Ellie turns {sorted(coverage.missing_ellie_ids)}")
    raise OfficialTranscriptCoverageError(
        "Official QR coverage is incomplete: " + "; ".join(issues)
    )

