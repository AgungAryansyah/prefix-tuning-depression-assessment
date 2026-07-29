import json
import tempfile
import unittest
from pathlib import Path

from prefix_tuning_depression.config import ModelConfig
from prefix_tuning_depression.evaluation import select_best_checkpoint


class EvaluationTests(unittest.TestCase):
    def test_selects_best_seed_and_epoch_by_dev_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, losses in ((0, [3.0, 2.0]), (1, [1.5, 1.0])):
                stem = root / f"prefix-only_seed{seed}"
                (stem.with_name(stem.name + "_history.json")).write_text(
                    json.dumps(
                        {
                            "dev_loss": losses,
                            "dev_rmse": [4.0, 3.0],
                            "dev_mae": [3.0, 2.0],
                        }
                    )
                )
                (stem.with_suffix(".pt")).write_bytes(b"checkpoint")
                (stem.with_name(stem.name + "_metadata.json")).write_text("{}")

            selected = select_best_checkpoint(root, "prefix-only", ModelConfig(), [0, 1])

        self.assertEqual(selected.seed, 1)
        self.assertEqual(selected.epoch, 1)
        self.assertEqual(selected.dev_loss, 1.0)
