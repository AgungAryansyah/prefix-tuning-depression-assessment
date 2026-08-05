import unittest
from unittest.mock import patch

import torch

from prefix_tuning_depression.config import ModelConfig
from prefix_tuning_depression.dataset import build_collator
from prefix_tuning_depression.models.encoders import E5_ID


class _Tokenizer:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def __call__(
        self,
        texts: list[str],
        *,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        self.inputs.append(texts)
        shape = (len(texts), max_length)
        return {
            "input_ids": torch.ones(shape, dtype=torch.long),
            "attention_mask": torch.ones(shape, dtype=torch.long),
        }


class DatasetCollatorTests(unittest.TestCase):
    @patch("prefix_tuning_depression.dataset.AutoTokenizer.from_pretrained")
    def test_bert_pt_uses_e5_query_inputs(self, from_pretrained) -> None:
        tokenizer = _Tokenizer()
        from_pretrained.return_value = tokenizer
        collator = build_collator(ModelConfig(), "bert-pt")

        collator([self._sample()])

        from_pretrained.assert_called_once_with(E5_ID)
        self.assertEqual(tokenizer.inputs, [["query: question answer"]])

    @patch("prefix_tuning_depression.dataset.AutoTokenizer.from_pretrained")
    def test_bert_fine_tuning_inputs_are_unchanged(self, from_pretrained) -> None:
        tokenizer = _Tokenizer()
        from_pretrained.return_value = tokenizer
        collator = build_collator(ModelConfig(), "bert-ft1")

        collator([self._sample()])

        from_pretrained.assert_called_once_with("bert-base-uncased")
        self.assertEqual(tokenizer.inputs, [["question answer"]])

    @patch("prefix_tuning_depression.dataset.AutoTokenizer.from_pretrained")
    def test_prefix_only_uses_the_slow_deberta_tokenizer(self, from_pretrained) -> None:
        from_pretrained.return_value = _Tokenizer()

        build_collator(ModelConfig(), "prefix-only")

        from_pretrained.assert_called_once_with(
            "microsoft/deberta-v3-base", use_fast=False
        )

    def _sample(self) -> dict[str, object]:
        return {
            "text_input": ["question answer"],
            "score_label": 0.0,
            "binary_label": 0,
        }
