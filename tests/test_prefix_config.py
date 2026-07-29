import unittest
from unittest.mock import patch

from transformers import RobertaConfig, RobertaModel

from prefix_tuning_depression.config import ModelConfig
from prefix_tuning_depression.models.prefix import build_prefix_encoder


class PrefixConfigurationTests(unittest.TestCase):
    def test_reserves_prefix_positions_from_the_roberta_token_budget(self) -> None:
        config = ModelConfig()

        self.assertEqual(config.prefix_text_max_token_length, 118)

    def test_keeps_roberta_internal_dropout_at_its_pretrained_value(self) -> None:
        config = RobertaConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=37,
            hidden_dropout_prob=0.1,
        )
        pretrained = RobertaModel(config, add_pooling_layer=False)

        with patch(
            "prefix_tuning_depression.models.prefix.AutoConfig.from_pretrained",
            return_value=config,
        ), patch(
            "prefix_tuning_depression.models.prefix.RobertaModel.from_pretrained",
            return_value=pretrained,
        ):
            encoder = build_prefix_encoder(pre_seq_len=10)

        self.assertEqual(encoder.dropout.p, 0.1)
