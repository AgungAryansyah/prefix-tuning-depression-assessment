"""PyTorch Dataset and collator for interview-level training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from prefix_tuning_depression.config import ModelConfig
from prefix_tuning_depression.data import Interview


def _qr_pairs_from_sample(sample: dict[str, object]) -> list[str]:
    return sample["text_input"]  # type: ignore[return-value]


class InterviewDataset(Dataset):
    """Dataset of interviews represented as lists of QR pair strings."""

    def __init__(self, interviews: list[Interview]):
        self.interviews = interviews

    def __len__(self) -> int:
        return len(self.interviews)

    def __getitem__(self, idx: int) -> dict[str, object]:
        interview = self.interviews[idx]
        return {
            "subject_id": interview.subject_id,
            "text_input": interview.qr_pairs,
            "score_label": interview.phq_score,
            "binary_label": interview.phq_binary,
        }


class InterviewCollator:
    """Tokenize and pad QR pairs for both ST and prefix encoders."""

    def __init__(self, config: ModelConfig):
        self.st_tokenizer = AutoTokenizer.from_pretrained(config.transformer_pretrained_id)
        self.prefix_tokenizer = AutoTokenizer.from_pretrained(
            config.prefix_backbone, use_fast=True
        )
        self.st_max_len = config.st_max_token_length
        self.prefix_max_len = config.prefix_max_token_length
        self.pre_seq_len = config.pre_seq_len

    def _encode(
        self,
        qr_pairs: list[str],
        tokenizer: AutoTokenizer,
        max_len: int,
    ) -> torch.Tensor:
        """Return tensor of shape (num_qr, 2, max_len): [input_ids, attention_mask]."""
        encoded = tokenizer(
            qr_pairs,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return torch.stack([input_ids, attention_mask], dim=1)

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        interview_lengths = torch.tensor(
            [len(_qr_pairs_from_sample(sample)) for sample in batch], dtype=torch.long
        )
        max_len = int(interview_lengths.max())
        batch_size = len(batch)

        st_inputs = torch.zeros(
            batch_size, max_len, 2, self.st_max_len, dtype=torch.long
        )
        prefix_inputs = torch.zeros(
            batch_size, max_len, 2, self.prefix_max_len, dtype=torch.long
        )

        for sample_idx, sample in enumerate(batch):
            qr_pairs = _qr_pairs_from_sample(sample)
            st_inputs[sample_idx, : len(qr_pairs)] = self._encode(
                qr_pairs, self.st_tokenizer, self.st_max_len
            )
            prefix_inputs[sample_idx, : len(qr_pairs)] = self._encode(
                qr_pairs, self.prefix_tokenizer, self.prefix_max_len
            )

        labels = torch.tensor(
            [sample["score_label"] for sample in batch], dtype=torch.float32
        )

        return {
            "st_inputs": st_inputs,
            "prefix_inputs": prefix_inputs,
            "interview_lengths": interview_lengths,
            "labels": labels,
        }


class BaselineCollator:
    """Collator for single-encoder baseline models."""

    def __init__(self, config: ModelConfig, tokenizer_name: str, max_len: int):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def _encode(
        self, qr_pairs: list[str], tokenizer: AutoTokenizer, max_len: int
    ) -> torch.Tensor:
        encoded = tokenizer(
            qr_pairs,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return torch.stack([encoded["input_ids"], encoded["attention_mask"]], dim=1)

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        interview_lengths = torch.tensor(
            [len(_qr_pairs_from_sample(sample)) for sample in batch], dtype=torch.long
        )
        max_len = int(interview_lengths.max())
        batch_size = len(batch)

        inputs = torch.zeros(batch_size, max_len, 2, self.max_len, dtype=torch.long)
        for sample_idx, sample in enumerate(batch):
            qr_pairs = _qr_pairs_from_sample(sample)
            inputs[sample_idx, : len(qr_pairs)] = self._encode(
                qr_pairs, self.tokenizer, self.max_len
            )

        labels = torch.tensor(
            [sample["score_label"] for sample in batch], dtype=torch.float32
        )

        return {
            "inputs": inputs,
            "interview_lengths": interview_lengths,
            "labels": labels,
        }


def build_collator(config: ModelConfig, model_type: str):
    """Return the appropriate collator for the model type."""
    match model_type:
        case "prefix-only":
            return BaselineCollator(
                config, config.prefix_backbone, config.prefix_max_token_length
            )
        case "st-only" | "dual-encoder":
            return InterviewCollator(config)
        case "bert-pt" | "bert-ft1" | "bert-ft2":
            return BaselineCollator(
                config, "bert-base-uncased", config.prefix_max_token_length
            )
        case "roberta-pt" | "roberta-ft1" | "roberta-ft2":
            return BaselineCollator(
                config, "roberta-base", config.prefix_max_token_length
            )
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
