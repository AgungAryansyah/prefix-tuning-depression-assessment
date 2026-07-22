"""Attention mechanism over BiLSTM outputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Self-attention over a sequence using a learnable context vector.

    Based on Yang et al. (2016) hierarchical attention and the original
    dual-encoder implementation.
    """

    def __init__(self, hidden_size: int, bidirectional: bool = True):
        super().__init__()
        self.hidden_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.context_vector = nn.Parameter(
            torch.FloatTensor(self.hidden_size).normal_(0, 0.01)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the attended vector and attention weights.

        Args:
            x: Tensor of shape (batch, seq_len, hidden_size).
            lengths: Sequence lengths per sample, used to mask padding.

        Returns:
            (attended_output, attention_weights) where attended_output has
            shape (batch, hidden_size) and attention_weights has shape
            (batch, seq_len).
        """
        if isinstance(lengths, list):
            lengths = torch.tensor(lengths, dtype=torch.long, device=x.device)

        # (batch, seq_len, hidden_size)
        h = self.tanh(self.linear(x))
        # (batch, seq_len)
        alpha = torch.mul(h, self.context_vector).sum(dim=2)

        max_len = alpha.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None]
        alpha = alpha.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(alpha, dim=1)

        # (batch, hidden_size, 1)
        attended = torch.bmm(x.transpose(1, 2), alpha.unsqueeze(2)).squeeze(2)
        return attended, alpha
