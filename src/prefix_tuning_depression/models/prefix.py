"""Prefix-tuned RoBERTa QR encoder.

Implementation follows Lau et al. (2023), adapted from P-Tuning v2:
https://github.com/THUDM/P-tuning-v2

The pretrained RoBERTa weights are frozen; only the prefix vectors are trained.
QR embeddings are extracted by averaging the last hidden-state token vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, RobertaModel
from transformers.cache_utils import DynamicCache


class PrefixEncoder(nn.Module):
    """Trainable prefix embedding table.

    Outputs prefix vectors shaped for HuggingFace's ``past_key_values``.
    """

    def __init__(self, pre_seq_len: int, num_hidden_layers: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(prefix_tokens)


class RobertaPrefixEncoder(nn.Module):
    """RoBERTa encoder with trainable prefix vectors prepended to every layer."""

    def __init__(self, config: AutoConfig):
        super().__init__()
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Freeze the pretrained backbone.
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.register_buffer(
            "prefix_tokens", torch.arange(self.pre_seq_len).long(), persistent=False
        )
        self.prefix_encoder = PrefixEncoder(
            self.pre_seq_len, self.n_layer, config.hidden_size
        )

    def get_prompt(self, batch_size: int, device: torch.device) -> DynamicCache:
        """Build a ``DynamicCache`` from the trainable prefix embeddings."""
        prefix_tokens = (
            self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        )
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd,
        )
        past_key_values = self.dropout(past_key_values)
        # Convert to the tuple format expected by DynamicCache.
        per_layer = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        cache = DynamicCache()
        for layer_idx, (key, value) in enumerate(per_layer):
            cache.update(key, value, layer_idx)
        return cache

    def train(self, mode: bool = True) -> "RobertaPrefixEncoder":
        super().train(mode)
        self.roberta.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the mean-pooled last-layer representation for each input."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        past_key_values = self.get_prompt(batch_size, device)

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )

        # Mean pool over the full last hidden state (matches original code).
        last_hidden_state = outputs.hidden_states[-1]
        return torch.mean(last_hidden_state, dim=1)


def build_prefix_encoder(
    prefix_backbone: str = "roberta-base",
    pre_seq_len: int = 10,
) -> RobertaPrefixEncoder:
    """Create a prefix-tuned RoBERTa encoder with the given hyperparameters."""
    config = AutoConfig.from_pretrained(prefix_backbone)
    config.pre_seq_len = pre_seq_len
    encoder = RobertaPrefixEncoder(config)
    # Load pretrained RoBERTa weights into the frozen backbone.
    pretrained = RobertaModel.from_pretrained(prefix_backbone, add_pooling_layer=False)
    encoder.roberta.load_state_dict(pretrained.state_dict(), strict=True)
    return encoder


def count_trainable_params(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
