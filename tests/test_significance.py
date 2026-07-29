import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from prefix_tuning_depression.metrics import pairwise_error_anova
from prefix_tuning_depression.training import collect_predictions


class _BatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> dict[str, torch.Tensor]:
        return self.batch


class SignificanceTests(unittest.TestCase):
    def test_reports_absolute_and_squared_error_anova(self) -> None:
        result = pairwise_error_anova(
            np.array([0.0, 1.0, 3.0, 6.0]),
            np.array([0.0, 0.0, 2.0, 4.0]),
            np.array([1.0, 3.0, 4.0, 7.0]),
        )

        self.assertEqual(set(result), {"absolute", "squared"})
        self.assertGreaterEqual(result["absolute"]["p_value"], 0.0)
        self.assertLessEqual(result["squared"]["p_value"], 1.0)

    def test_rejects_misaligned_prediction_arrays(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_error_anova(
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0, 1.0]),
            )

    def test_collects_labels_and_predictions(self) -> None:
        batch = {
            "labels": torch.tensor([1.0, 2.0]),
            "interview_lengths": torch.tensor([1, 1]),
        }
        with patch(
            "prefix_tuning_depression.training._forward_batch",
            return_value=torch.tensor([[1.5], [2.5]]),
        ):
            labels, predictions = collect_predictions(
                nn.Linear(1, 1),
                DataLoader(_BatchDataset(batch), batch_size=None),
                torch.device("cpu"),
                "prefix-only",
            )

        np.testing.assert_array_equal(labels, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(predictions, np.array([1.5, 2.5]))
