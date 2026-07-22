"""Layer initialization helpers from the original implementation."""

from __future__ import annotations

import torch.nn as nn


def init_linear_layer(layer: nn.Module) -> None:
    """Xavier uniform init for weights; small positive bias init."""
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.01)


def init_rnn(rnn: nn.Module) -> None:
    """Xavier uniform init for RNN weights; zero bias init."""
    for name, param in rnn.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0.0)
        elif "weight" in name:
            nn.init.xavier_uniform_(param)
