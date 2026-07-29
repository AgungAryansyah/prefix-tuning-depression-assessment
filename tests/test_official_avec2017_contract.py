import csv
import tempfile
import unittest
from pathlib import Path

from prefix_tuning_depression.data import load_interviews
from prefix_tuning_depression.audit import build_reproduction_audit
from prefix_tuning_depression.splits import (
    CANONICAL_DAIC_WOZ_IDS,
    OfficialTranscriptCoverageError,
    inspect_official_transcript_coverage,
    load_official_avec2017_contract,
    require_complete_official_transcript_coverage,
)


class OfficialAvec2017ContractTests(unittest.TestCase):
    def test_loads_the_complete_canonical_contract(self) -> None:
        session_ids = sorted(CANONICAL_DAIC_WOZ_IDS)
        with tempfile.TemporaryDirectory() as directory:
            manifest_dir = Path(directory)
            contract = self._write_contract(manifest_dir, session_ids)

        self.assertEqual(len(contract.split_map), 189)
        self.assertEqual(len(contract.labels), 189)
        self.assertEqual(contract.split_map[session_ids[0]], "train")
        self.assertEqual(contract.split_map[session_ids[-1]], "test")

    def test_reports_incomplete_qr_coverage(self) -> None:
        session_ids = sorted(CANONICAL_DAIC_WOZ_IDS)

        with tempfile.TemporaryDirectory() as directory:
            manifest_dir = Path(directory)
            contract = self._write_contract(manifest_dir, session_ids)
            for subject_id in session_ids:
                if subject_id != 458:
                    self._write_transcript(
                        manifest_dir / f"{subject_id}_TRANSCRIPT.csv",
                        has_ellie=subject_id != 451,
                    )

            coverage = inspect_official_transcript_coverage(contract, manifest_dir)

            self.assertEqual(coverage.missing_transcript_ids, frozenset({458}))
            self.assertEqual(coverage.missing_ellie_ids, frozenset({451}))
            with self.assertRaises(OfficialTranscriptCoverageError):
                require_complete_official_transcript_coverage(contract, manifest_dir)

            self._write_transcript(manifest_dir / "458_TRANSCRIPT.csv", has_ellie=True)
            self._write_transcript(manifest_dir / "451_TRANSCRIPT.csv", has_ellie=True)
            require_complete_official_transcript_coverage(contract, manifest_dir)

            interviews = load_interviews(manifest_dir, split="train")

        self.assertEqual(len(interviews), 107)
        self.assertEqual(interviews[0].phq_score, 0)

    def test_audit_marks_incomplete_inputs_blocked(self) -> None:
        session_ids = sorted(CANONICAL_DAIC_WOZ_IDS)

        with tempfile.TemporaryDirectory() as directory:
            manifest_dir = Path(directory)
            contract = self._write_contract(manifest_dir, session_ids)
            for subject_id in session_ids:
                if subject_id != 458:
                    self._write_transcript(
                        manifest_dir / f"{subject_id}_TRANSCRIPT.csv",
                        has_ellie=subject_id != 451,
                    )
            audit = build_reproduction_audit(contract, manifest_dir)

        self.assertEqual(audit["status"], "blocked")
        self.assertEqual(audit["missing_transcript_ids"], [458])
        self.assertEqual(audit["missing_ellie_ids"], [451])
    def _write_contract(
        self, manifest_dir: Path, session_ids: list[int]
    ):
        splits = {
            "train": session_ids[:107],
            "dev": session_ids[107:142],
            "test": session_ids[142:],
        }
        self._write_manifest(
            manifest_dir / "train_split_Depression_AVEC2017.csv",
            "Participant_ID,PHQ8_Binary,PHQ8_Score",
            splits["train"],
        )
        self._write_manifest(
            manifest_dir / "dev_split_Depression_AVEC2017.csv",
            "Participant_ID,PHQ8_Binary,PHQ8_Score",
            splits["dev"],
        )
        self._write_manifest(
            manifest_dir / "test_split_Depression_AVEC2017.csv",
            "participant_ID,Gender",
            splits["test"],
        )
        self._write_manifest(
            manifest_dir / "full_test_split.csv",
            "Participant_ID,PHQ_Binary,PHQ_Score",
            splits["test"],
        )
        return load_official_avec2017_contract(manifest_dir)

    def _write_manifest(self, path: Path, header: str, session_ids: list[int]) -> None:
        columns = header.split(",")
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for subject_id in session_ids:
                row = {columns[0]: subject_id}
                for column in columns[1:]:
                    row[column] = 0
                writer.writerow(row)

    def _write_transcript(self, path: Path, has_ellie: bool) -> None:
        rows = ["start_time\tstop_time\tspeaker\tvalue"]
        if has_ellie:
            rows.append("0\t1\tEllie\tHow are you?")
        rows.append("1\t2\tParticipant\tFine")
        path.write_text("\n".join(rows))
