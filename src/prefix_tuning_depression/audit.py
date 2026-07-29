"""Read-only readiness audit for the official reproduction inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prefix_tuning_depression.splits import (
    OfficialDaicWozContract,
    inspect_official_transcript_coverage,
)

QR_POLICY = "strict_ellie_participant_adjacent"


def build_reproduction_audit(
    contract: OfficialDaicWozContract,
    transcript_dir: Path | str,
) -> dict[str, Any]:
    """Return a machine-readable readiness report without changing data."""
    coverage = inspect_official_transcript_coverage(contract, transcript_dir)
    issues: list[str] = []
    if coverage.missing_transcript_ids:
        issues.append("recover missing official transcripts")
    if coverage.missing_ellie_ids:
        issues.append("resolve official sessions without Ellie turns")
    return {
        "status": "ready" if not issues else "blocked",
        "qr_policy": QR_POLICY,
        "split_counts": {
            split: sum(value == split for value in contract.split_map.values())
            for split in ("train", "dev", "test")
        },
        "missing_transcript_ids": sorted(coverage.missing_transcript_ids),
        "missing_ellie_ids": sorted(coverage.missing_ellie_ids),
        "blocking_issues": issues,
    }
