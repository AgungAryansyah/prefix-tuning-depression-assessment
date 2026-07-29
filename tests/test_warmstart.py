import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from prefix_tuning_depression.config import ModelConfig
from prefix_tuning_depression.models import depression_model


class _PrefixModel(nn.Module):
    def __init__(self, _config: ModelConfig) -> None:
        super().__init__()
        self.prefix_encoder = nn.Linear(2, 2)
        self.prefix_projection = nn.Linear(2, 2)
        self.interview_encoder = nn.Linear(2, 1)


class _DualModel(_PrefixModel):
    pass


class WarmStartTests(unittest.TestCase):
    def test_copies_and_keeps_the_prefix_branch_trainable(self) -> None:
        source = _PrefixModel(ModelConfig())
        for parameter in source.parameters():
            nn.init.constant_(parameter, 0.25)

        with tempfile.NamedTemporaryFile() as checkpoint:
            torch.save(source.state_dict(), checkpoint.name)
            with patch.object(depression_model, "PrefixModel", _PrefixModel), patch.object(
                depression_model, "DualEncoderModel", _DualModel
            ):
                dual = depression_model.build_warmstarted_dual_encoder(
                    ModelConfig(), checkpoint.name, "cpu"
                )

        self._assert_same_state(dual.prefix_encoder, source.prefix_encoder)
        self._assert_same_state(dual.prefix_projection, source.prefix_projection)
        self._assert_same_state(dual.interview_encoder, source.interview_encoder)
        self.assertTrue(
            all(parameter.requires_grad for parameter in dual.prefix_encoder.parameters())
        )

    def _assert_same_state(self, actual: nn.Module, expected: nn.Module) -> None:
        for name, expected_tensor in expected.state_dict().items():
            self.assertTrue(torch.equal(actual.state_dict()[name], expected_tensor))
