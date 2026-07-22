"""Interview-level encoder: BiLSTM + attention + prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn

from prefix_tuning_depression.models.attention import AttentionLayer
from prefix_tuning_depression.models.initialization import init_linear_layer, init_rnn


class InterviewEncoder(nn.Module):
    """BiLSTM with attention for summarizing a sequence of QR embeddings."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_labels: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        init_rnn(self.lstm)

        self.attention = AttentionLayer(hidden_size=hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.prediction_head = nn.Linear(hidden_size * 2, num_labels)
        self.prediction_head.apply(init_linear_layer)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | list[int],
    ) -> torch.Tensor:
        """Return PHQ-8 prediction logits for each interview.

        Args:
            x: Tensor of shape (batch, max_len, input_size).
            lengths: Sequence lengths per sample.

        Returns:
            Logits of shape (batch, num_labels).
        """
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, dtype=torch.long, device=x.device)

        # Pack and run BiLSTM.
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        self.lstm.flatten_parameters()
        lstm_output, _ = self.lstm(packed)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)

        # Attend over the QR sequence.
        attended, _ = self.attention(lstm_output, lengths)
        attended = self.dropout(attended)

        return self.prediction_head(attended)


def build_interview_encoder(
    input_size: int = 128,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.5,
    num_labels: int = 1,
) -> InterviewEncoder:
    """Create an interview-level encoder with paper hyperparameters."""
    return InterviewEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_labels=num_labels,
    )
