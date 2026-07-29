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

FUSION_METHODS = ("addition", "average", "concatenation")
_FUSION_ALIASES = {"add": "addition", "avg": "average", "concat": "concatenation"}


def normalize_fusion_method(method: str) -> str:
    """Return a supported QR-embedding fusion method."""
    method = _FUSION_ALIASES.get(method, method)
    if method not in FUSION_METHODS:
        raise ValueError(f"Unknown fusion method: {method}")
    return method


def fusion_output_size(method: str, embedding_size: int) -> int:
    """Return the QR embedding size after fusion."""
    return embedding_size * 2 if normalize_fusion_method(method) == "concatenation" else embedding_size


def fuse_qr_embeddings(
    method: str,
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Fuse projected prefix and sentence-transformer QR embeddings."""
    match normalize_fusion_method(method):
        case "addition":
            return first + second
        case "average":
            return (first + second) / 2
        case "concatenation":
            return torch.cat((first, second), dim=-1)
    raise AssertionError("unreachable")


class BaseDepressionModel(nn.Module):
    """Shared interview-level processing pipeline."""

    def __init__(self, config: ModelConfig, interview_input_size: int | None = None):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout_prob)
        self.interview_encoder = InterviewEncoder(
            input_size=interview_input_size or config.encoding_projection_size,
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
    """Fusion of prefix-tuned and Sentence Transformer embeddings."""

    def __init__(self, config: ModelConfig):
        self.fusion_method = normalize_fusion_method(config.fusion_method)
        super().__init__(
            config,
            interview_input_size=fusion_output_size(
                self.fusion_method, config.encoding_projection_size
            ),
        )
        self.prefix_encoder = build_prefix_encoder(
            prefix_backbone=config.prefix_backbone,
            pre_seq_len=config.pre_seq_len,
        )
        self.st_encoder = SentenceTransformerEncoder(model_id=ST_ID)
        self.prefix_projection = nn.Linear(768, config.encoding_projection_size)
        self.st_projection = nn.Linear(768, config.encoding_projection_size)
        self.prefix_projection.apply(init_linear_layer)
        self.st_projection.apply(init_linear_layer)

    def forward(
        self,
        st_inputs: torch.Tensor,
        prefix_inputs: torch.Tensor,
        interview_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with configured QR-embedding fusion.

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

        encoder_outputs = fuse_qr_embeddings(
            self.fusion_method, st_encodings, prefix_encodings
        )
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
    copied from the prefix checkpoint and remain trainable.
    """
    if normalize_fusion_method(config.fusion_method) != "average":
        raise ValueError("Warm-start dual encoder requires average fusion")

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

    return dual_model
