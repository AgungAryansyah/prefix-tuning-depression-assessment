import unittest
from unittest.mock import patch

import torch
from transformers import DebertaV2Config, DebertaV2Model

from prefix_tuning_depression.config import ModelConfig
from prefix_tuning_depression.models.prefix import (
    DebertaDoraEncoder,
    build_prefix_encoder,
    count_trainable_params,
)


class DoraConfigurationTests(unittest.TestCase):
    def test_uses_the_full_token_budget(self) -> None:
        self.assertEqual(ModelConfig().prefix_text_max_token_length, 128)

    def test_builds_a_query_value_dora_adapter(self) -> None:
        encoder = DebertaDoraEncoder(self._backbone())
        config = encoder.encoder.peft_config["default"]

        self.assertTrue(config.use_dora)
        self.assertEqual(config.target_modules, {"query_proj", "value_proj"})
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertEqual(config.lora_dropout, 0.1)
        self.assertGreater(count_trainable_params(encoder), 0)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in encoder.named_parameters()
                if "lora_" not in name and "lora_magnitude_vector" not in name
            )
        )

    def test_returns_masked_mean_embeddings(self) -> None:
        encoder = DebertaDoraEncoder(self._backbone())
        encoder.eval()

        embeddings = encoder(
            torch.tensor([[1, 2, 0, 0]]),
            torch.tensor([[1, 1, 0, 0]]),
        )

        self.assertEqual(embeddings.shape, (1, 32))

    def test_gradient_checkpointing_keeps_dora_gradients(self) -> None:
        encoder = DebertaDoraEncoder(self._backbone())
        encoder.train()

        encoder(torch.tensor([[1, 2]]), torch.tensor([[1, 1]])).sum().backward()

        self.assertTrue(encoder.encoder.get_base_model().is_gradient_checkpointing)
        self.assertTrue(
            any(
                parameter.grad is not None
                for name, parameter in encoder.named_parameters()
                if "lora_" in name
            )
        )

    def test_keeps_the_existing_factory_signature(self) -> None:
        backbone = self._backbone()
        with patch(
            "prefix_tuning_depression.models.prefix.AutoModel.from_pretrained",
            return_value=backbone,
        ) as from_pretrained:
            encoder = build_prefix_encoder("test-backbone", pre_seq_len=10)

        from_pretrained.assert_called_once_with("test-backbone")
        self.assertIsInstance(encoder, DebertaDoraEncoder)

    def _backbone(self) -> DebertaV2Model:
        return DebertaV2Model(
            DebertaV2Config(
                vocab_size=100,
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=64,
            )
        )
