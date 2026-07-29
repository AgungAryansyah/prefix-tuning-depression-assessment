import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from prefix_tuning_depression.artifacts import (
    build_run_metadata,
    run_name,
    sha256_file,
)
from prefix_tuning_depression.config import ModelConfig, TrainingConfig
from prefix_tuning_depression.data import Interview
from prefix_tuning_depression.splits import OfficialDaicWozContract


class ArtifactTests(unittest.TestCase):
    def test_metadata_contains_config_splits_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = (
                "train_split_Depression_AVEC2017.csv",
                "dev_split_Depression_AVEC2017.csv",
                "test_split_Depression_AVEC2017.csv",
                "full_test_split.csv",
            )
            for name in names:
                (root / name).write_text(name)
            for subject_id in (300, 301, 302):
                (root / f"{subject_id}_TRANSCRIPT.csv").write_text(
                    "start_time\tstop_time\tspeaker\tvalue\n"
                    "0\t1\tEllie\tquestion\n"
                    "1\t2\tParticipant\tanswer\n"
                )
            transcript_hash = sha256_file(root / "300_TRANSCRIPT.csv")

            contract = OfficialDaicWozContract(
                split_map={300: "train", 301: "dev", 302: "test"},
                labels={300: (0.0, 0), 301: (1.0, 0), 302: (2.0, 0)},
            )
            interview = Interview(300, ["q r"], 0.0, 0, "train")
            metadata = build_run_metadata(
                root,
                contract,
                "prefix-only",
                ModelConfig(),
                TrainingConfig(),
                7,
                [interview],
                [],
                "cpu",
            )

        self.assertEqual(metadata["seed"], 7)
        self.assertEqual(metadata["model_config"], asdict(ModelConfig()))
        self.assertEqual(metadata["split_subject_ids"]["train"], [300])
        self.assertEqual(metadata["transcript_sha256"]["300"], transcript_hash)

    def test_dual_run_name_includes_fusion(self) -> None:
        self.assertEqual(run_name("dual-encoder", ModelConfig()), "dual-encoder_average")
