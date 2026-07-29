import unittest
from unittest.mock import patch

import torch.nn as nn

from prefix_tuning_depression.models.encoders import (
    BaselineTransformerEncoder,
    SentenceTransformerEncoder,
    encoder_trainable_params,
)


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])


class FrozenEncoderTests(unittest.TestCase):
    @patch("prefix_tuning_depression.models.encoders.AutoModel.from_pretrained")
    def test_frozen_encoders_stay_in_evaluation_mode(self, from_pretrained) -> None:
        from_pretrained.side_effect = lambda _: _Transformer()
        sentence_encoder = SentenceTransformerEncoder("test")
        frozen_baseline = BaselineTransformerEncoder("test")
        tuned_baseline = BaselineTransformerEncoder("test", unfreeze_last_n=1)

        sentence_encoder.train()
        frozen_baseline.train()
        tuned_baseline.train()

        self.assertFalse(sentence_encoder.encoder.training)
        self.assertFalse(frozen_baseline.encoder.training)
        self.assertTrue(tuned_baseline.encoder.training)
        self.assertEqual(encoder_trainable_params(frozen_baseline), 0)
        self.assertEqual(encoder_trainable_params(tuned_baseline), 6)
