import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from prefix_tuning_depression.models.encoders import (
    BaselineTransformerEncoder,
    E5_ID,
    SentenceTransformerEncoder,
    build_encoder,
    encoder_trainable_params,
)


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        hidden = torch.ones(
            input_ids.shape[0], input_ids.shape[1], 2, device=input_ids.device
        )
        return SimpleNamespace(last_hidden_state=hidden)


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

    @patch("prefix_tuning_depression.models.encoders.AutoModel.from_pretrained")
    def test_bert_pt_uses_normalized_frozen_e5(self, from_pretrained) -> None:
        from_pretrained.return_value = _Transformer()

        encoder = build_encoder("bert-pt")
        encoder.train()
        embeddings = encoder(
            torch.tensor([[1, 2, 0]]), torch.tensor([[1, 1, 0]])
        )

        from_pretrained.assert_called_once_with(E5_ID)
        self.assertFalse(encoder.encoder.training)
        self.assertTrue(torch.allclose(embeddings.norm(dim=1), torch.ones(1)))
