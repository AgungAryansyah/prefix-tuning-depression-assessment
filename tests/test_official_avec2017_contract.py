import csv
import tempfile
import unittest
from pathlib import Path

from prefix_tuning_depression.splits import (
    CANONICAL_DAIC_WOZ_IDS,
    load_official_avec2017_contract,
)


class OfficialAvec2017ContractTests(unittest.TestCase):
    def test_loads_the_complete_canonical_contract(self) -> None:
        session_ids = sorted(CANONICAL_DAIC_WOZ_IDS)
        splits = {
            "train": session_ids[:107],
            "dev": session_ids[107:142],
            "test": session_ids[142:],
        }

        with tempfile.TemporaryDirectory() as directory:
            manifest_dir = Path(directory)
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

            contract = load_official_avec2017_contract(manifest_dir)

        self.assertEqual(len(contract.split_map), 189)
        self.assertEqual(len(contract.labels), 189)
        self.assertEqual(contract.split_map[session_ids[0]], "train")
        self.assertEqual(contract.split_map[session_ids[-1]], "test")

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
