"""Full depression severity models: QR encoder + interview-level encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from prefix_tuning_depression.config import ModelConfig
from prefix_tuning_depression.models.encoders import (
    ST_ID,
    SentenceTransformerEncoder,
    build_encoder as build_baseline_encoder,
)
from prefix_tuning_depression.models.initialization import init_linear_layer
from prefix_tuning_depression.models.interview import InterviewEncoder
from prefix_tuning_depression.models.prefix import build_prefix_encoder


class BaseDepressionModel(nn.Module):
    """Shared interview-level processing pipeline."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout_prob)
        self.interview_encoder = InterviewEncoder(
            input_size=config.encoding_projection_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout_prob,
            num_labels=config.num_labels,
        )

    def _encode_qr_sequence(
        self,
        embeddings: torch.Tensor,
        interview_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Pass a sequence of QR embeddings through the interview encoder."""
        return self.interview_encoder(embeddings, interview_lengths)


class PrefixModel(BaseDepressionModel):
    """Prefix-tuned RoBERTa QR encoder + interview-level encoder."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.prefix_encoder = build_prefix_encoder(
            prefix_backbone=config.prefix_backbone,
            pre_seq_len=config.pre_seq_len,
            dropout_prob=config.dropout_prob,
        )
        self.prefix_projection = nn.Linear(768, config.encoding_projection_size)
        self.prefix_projection.apply(init_linear_layer)

    def forward(
        self,
        prefix_inputs: torch.Tensor,
        interview_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            prefix_inputs: Tensor of shape
                (batch, max_len, 2, prefix_max_token_len). The last dimension
                stores input_ids at index 0 and attention_mask at index 1.
            interview_lengths: Tensor of shape (batch,) with the number of QR
                pairs per interview.

        Returns:
            Logits of shape (batch, num_labels).
        """
        batch_size = prefix_inputs.shape[0]
        max_len = int(interview_lengths.max())
        device = prefix_inputs.device

        encoder_outputs = torch.zeros(
            batch_size,
            max_len,
            self.config.encoding_projection_size,
            device=device,
        )

        for sample_idx in range(batch_size):
            length = int(interview_lengths[sample_idx])
            input_ids = prefix_inputs[sample_idx, :length, 0, :]
            attention_mask = prefix_inputs[sample_idx, :length, 1, :]

            encodings = self.prefix_encoder(input_ids, attention_mask)
            encodings = self.dropout(encodings)
            encodings = self.prefix_projection(encodings)
            encoder_outputs[sample_idx, :length, :] = encodings

        return self._encode_qr_sequence(encoder_outputs, interview_lengths)


class STModel(BaseDepressionModel):
    """Sentence Transformer QR encoder + interview-level encoder."""

    def __init__(self, config: ModelConfig, model_id: str = ST_ID):
        super().__init__(config)
        self.st_encoder = SentenceTransformerEncoder(model_id=model_id)
        self.st_projection = nn.Linear(768, config.encoding_projection_size)
        self.st_projection.apply(init_linear_layer)

    def forward(
        self,
        st_inputs: torch.Tensor,
        interview_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            st_inputs: Tensor of shape (batch, max_len, 2, st_max_token_len).
            interview_lengths: Tensor of shape (batch,).

        Returns:
            Logits of shape (batch, num_labels).
        """
        batch_size = st_inputs.shape[0]
        max_len = int(interview_lengths.max())
        device = st_inputs.device

        encoder_outputs = torch.zeros(
            batch_size,
            max_len,
            self.config.encoding_projection_size,
            device=device,
        )

        for sample_idx in range(batch_size):
            length = int(interview_lengths[sample_idx])
            input_ids = st_inputs[sample_idx, :length, 0, :]
            attention_mask = st_inputs[sample_idx, :length, 1, :]

            encodings = self.st_encoder(input_ids, attention_mask)
            encodings = self.dropout(encodings)
            encodings = self.st_projection(encodings)
            encoder_outputs[sample_idx, :length, :] = encodings

        return self._encode_qr_sequence(encoder_outputs, interview_lengths)


class BaselineModel(BaseDepressionModel):
    """Frozen or fine-tuned BERT/RoBERTa baseline + interview-level encoder."""

    def __init__(self, config: ModelConfig, encoder_type: str):
        super().__init__(config)
        self.qr_encoder = build_baseline_encoder(encoder_type)
        self.qr_projection = nn.Linear(768, config.encoding_projection_size)
        self.qr_projection.apply(init_linear_layer)

    def forward(
        self,
        inputs: torch.Tensor,
        interview_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            inputs: Tensor of shape (batch, max_len, 2, max_token_len).
            interview_lengths: Tensor of shape (batch,).

        Returns:
            Logits of shape (batch, num_labels).
        """
        batch_size = inputs.shape[0]
        max_len = int(interview_lengths.max())
        device = inputs.device

        encoder_outputs = torch.zeros(
            batch_size,
            max_len,
            self.config.encoding_projection_size,
            device=device,
        )

        for sample_idx in range(batch_size):
            length = int(interview_lengths[sample_idx])
            input_ids = inputs[sample_idx, :length, 0, :]
            attention_mask = inputs[sample_idx, :length, 1, :]

            encodings = self.qr_encoder(input_ids, attention_mask)
            encodings = self.dropout(encodings)
            encodings = self.qr_projection(encodings)
            encoder_outputs[sample_idx, :length, :] = encodings

        return self._encode_qr_sequence(encoder_outputs, interview_lengths)


class DualEncoderModel(BaseDepressionModel):
    """Average fusion of prefix-tuned and Sentence Transformer embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.prefix_encoder = build_prefix_encoder(
            prefix_backbone=config.prefix_backbone,
            pre_seq_len=config.pre_seq_len,
            dropout_prob=config.dropout_prob,
        )
        self.st_encoder = SentenceTransformerEncoder(model_id=ST_ID)
        self.prefix_projection = nn.Linear(768, config.encoding_projection_size)
        self.st_projection = nn.Linear(768, config.encoding_projection_size)
        self.prefix_projection.apply(init_linear_layer)
        self.st_projection.apply(init_linear_layer)

        # In warm-start mode the prefix branch is frozen.
        self.freeze_prefix = False

    def average_fusion(
        self, encoding_0: torch.Tensor, encoding_1: torch.Tensor
    ) -> torch.Tensor:
        return (encoding_0 + encoding_1) / 2

    def forward(
        self,
        st_inputs: torch.Tensor,
        prefix_inputs: torch.Tensor,
        interview_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with average fusion.

        Args:
            st_inputs: Tensor of shape (batch, max_len, 2, st_max_token_len).
            prefix_inputs: Tensor of shape
                (batch, max_len, 2, prefix_max_token_len).
            interview_lengths: Tensor of shape (batch,).

        Returns:
            Logits of shape (batch, num_labels).
        """
        batch_size = st_inputs.shape[0]
        max_len = int(interview_lengths.max())
        device = st_inputs.device

        st_outputs = torch.zeros(
            batch_size,
            max_len,
            self.config.encoding_projection_size,
            device=device,
        )
        prefix_outputs = torch.zeros(
            batch_size,
            max_len,
            self.config.encoding_projection_size,
            device=device,
        )

        for sample_idx in range(batch_size):
            length = int(interview_lengths[sample_idx])

            # Sentence Transformer branch.
            input_ids = st_inputs[sample_idx, :length, 0, :]
            attention_mask = st_inputs[sample_idx, :length, 1, :]
            st_encodings = self.st_encoder(input_ids, attention_mask)
            st_encodings = self.dropout(st_encodings)
            st_encodings = self.st_projection(st_encodings)
            st_outputs[sample_idx, :length, :] = st_encodings

            # Prefix branch.
            input_ids = prefix_inputs[sample_idx, :length, 0, :]
            attention_mask = prefix_inputs[sample_idx, :length, 1, :]
            prefix_encodings = self.prefix_encoder(input_ids, attention_mask)
            prefix_encodings = self.dropout(prefix_encodings)
            prefix_encodings = self.prefix_projection(prefix_encodings)
            prefix_outputs[sample_idx, :length, :] = prefix_encodings

        encoder_outputs = self.average_fusion(st_outputs, prefix_outputs)
        return self._encode_qr_sequence(encoder_outputs, interview_lengths)


def build_depression_model(config: ModelConfig, model_type: str) -> BaseDepressionModel:
    """Factory for full depression severity models."""
    match model_type:
        case "prefix-only":
            return PrefixModel(config)
        case "st-only":
            return STModel(config)
        case "dual-encoder":
            return DualEncoderModel(config)
        case "bert-pt" | "bert-ft1" | "bert-ft2" | "roberta-pt" | "roberta-ft1" | "roberta-ft2":
            return BaselineModel(config, model_type)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
