"""Full depression severity models: QR encoder + interview-level encoder."""

from __future__ import annotations

from collections.abc import Callable

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

    def _encode_qr_embeddings(
        self,
        inputs: torch.Tensor,
        interview_lengths: torch.Tensor,
        encode_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Encode QR pairs in chunks to avoid OOM with long interviews.

        Args:
            inputs: Tensor of shape (batch, max_len, 2, max_token_len).
            interview_lengths: Tensor of shape (batch,).
            encode_fn: Function that takes (input_ids, attention_mask) and
                returns embeddings of shape (num_qr, hidden_size).

        Returns:
            Tensor of shape (batch, max_len, hidden_size).
        """
        batch_size = inputs.shape[0]
        max_len = int(interview_lengths.max())
        device = inputs.device

        # Determine the encoder's output dimension from a tiny probe.
        embedding_size = self.config.encoding_projection_size
        if batch_size > 0 and int(interview_lengths[0]) > 0:
            probe = encode_fn(
                inputs[0, :1, 0, :],
                inputs[0, :1, 1, :],
            )
            embedding_size = probe.shape[-1]

        outputs = torch.zeros(
            batch_size,
            max_len,
            embedding_size,
            device=device,
        )

        chunk_size = self.config.chunk_size
        for sample_idx in range(batch_size):
            length = int(interview_lengths[sample_idx])
            input_ids = inputs[sample_idx, :length, 0, :]
            attention_mask = inputs[sample_idx, :length, 1, :]

            chunks: list[torch.Tensor] = []
            for start in range(0, length, chunk_size):
                end = min(start + chunk_size, length)
                chunks.append(
                    encode_fn(input_ids[start:end], attention_mask[start:end])
                )

            if chunks:
                outputs[sample_idx, :length, :] = torch.cat(chunks, dim=0)

        return outputs

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
        encodings = self._encode_qr_embeddings(
            prefix_inputs, interview_lengths, self.prefix_encoder
        )
        encodings = self.dropout(encodings)
        encodings = self.prefix_projection(encodings)
        return self._encode_qr_sequence(encodings, interview_lengths)


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
        encodings = self._encode_qr_embeddings(
            st_inputs, interview_lengths, self.st_encoder
        )
        encodings = self.dropout(encodings)
        encodings = self.st_projection(encodings)
        return self._encode_qr_sequence(encodings, interview_lengths)


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
        encodings = self._encode_qr_embeddings(
            inputs, interview_lengths, self.qr_encoder
        )
        encodings = self.dropout(encodings)
        encodings = self.qr_projection(encodings)
        return self._encode_qr_sequence(encodings, interview_lengths)


class DualEncoderModel(BaseDepressionModel):
    """Average fusion of prefix-tuned and Sentence Transformer embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.prefix_encoder = build_prefix_encoder(
            prefix_backbone=config.prefix_backbone,
            pre_seq_len=config.pre_seq_len,
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
        st_encodings = self._encode_qr_embeddings(
            st_inputs, interview_lengths, self.st_encoder
        )
        st_encodings = self.dropout(st_encodings)
        st_encodings = self.st_projection(st_encodings)

        prefix_encodings = self._encode_qr_embeddings(
            prefix_inputs, interview_lengths, self.prefix_encoder
        )
        prefix_encodings = self.dropout(prefix_encodings)
        prefix_encodings = self.prefix_projection(prefix_encodings)

        encoder_outputs = self.average_fusion(st_encodings, prefix_encodings)
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
            return BaselineModel(config, encoder_type=model_type)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")


def build_warmstarted_dual_encoder(
    config: ModelConfig,
    prefix_checkpoint_path: str,
    device: torch.device | str,
) -> DualEncoderModel:
    """Create a dual encoder initialized from a trained prefix-only model.

    The prefix encoder, prefix projection, and interview-level layers are
    copied from the prefix checkpoint and the prefix encoder is frozen.
    """
    prefix_model = PrefixModel(config).to(device)
    prefix_model.load_state_dict(
        torch.load(prefix_checkpoint_path, map_location=device, weights_only=True)
    )

    dual_model = DualEncoderModel(config).to(device)
    dual_model.prefix_encoder.load_state_dict(prefix_model.prefix_encoder.state_dict())
    dual_model.prefix_projection.load_state_dict(
        prefix_model.prefix_projection.state_dict()
    )
    dual_model.interview_encoder.load_state_dict(
        prefix_model.interview_encoder.state_dict()
    )

    for param in dual_model.prefix_encoder.parameters():
        param.requires_grad = False
    dual_model.freeze_prefix = True

    return dual_model
