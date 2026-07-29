import unittest

import torch

from prefix_tuning_depression.models.depression_model import (
    fuse_qr_embeddings,
    fusion_output_size,
    normalize_fusion_method,
)


class FusionTests(unittest.TestCase):
    def test_fusion_methods(self) -> None:
        first = torch.tensor([[1.0, 2.0]])
        second = torch.tensor([[3.0, 4.0]])

        self.assertTrue(
            torch.equal(
                fuse_qr_embeddings("addition", first, second),
                torch.tensor([[4.0, 6.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                fuse_qr_embeddings("average", first, second),
                torch.tensor([[2.0, 3.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                fuse_qr_embeddings("concatenation", first, second),
                torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            )
        )

    def test_normalizes_aliases_and_sizes_concatenation(self) -> None:
        self.assertEqual(normalize_fusion_method("avg"), "average")
        self.assertEqual(fusion_output_size("concatenation", 128), 256)
