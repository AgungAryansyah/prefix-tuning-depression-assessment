"""Sentence Transformer and baseline transformer encoders.

All encoders output a fixed-size embedding per QR pair. The Sentence Transformer
is frozen and L2-normalized; baselines are either frozen (PT) or have the last
one/two layers unfrozen (FT1/FT2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


BERT_ID = "bert-base-uncased"
E5_ID = "intfloat/e5-base-v2"
ROBERTA_ID = "roberta-base"
ST_ID = "sentence-transformers/all-mpnet-base-v2"


def _mean_pooling(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool token embeddings, ignoring padding."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class SentenceTransformerEncoder(nn.Module):
    """Frozen sentence-transformers/all-mpnet-base-v2 encoder with L2 normalization."""

    def __init__(self, model_id: str = ST_ID):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "SentenceTransformerEncoder":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = _mean_pooling(outputs.last_hidden_state, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)


class BaselineTransformerEncoder(nn.Module):
    """Frozen or partially-fine-tuned transformer encoder.

    Args:
        model_id: HuggingFace model identifier.
        unfreeze_last_n: Number of transformer layers from the end to unfreeze.
            0 means fully frozen (PT); 1 means FT1; 2 means FT2.
    """

    def __init__(
        self,
        model_id: str,
        unfreeze_last_n: int = 0,
        normalize_output: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        self.unfreeze_last_n = unfreeze_last_n
        self.normalize_output = normalize_output

        # Freeze everything first.
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze the last N encoder layers.
        if unfreeze_last_n > 0:
            for layer in self.encoder.encoder.layer[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def train(self, mode: bool = True) -> "BaselineTransformerEncoder":
        super().train(mode)
        if self.unfreeze_last_n == 0:
            self.encoder.eval()
        return self

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = _mean_pooling(outputs.last_hidden_state, attention_mask)
        if self.normalize_output:
            return F.normalize(embeddings, p=2, dim=1)
        return embeddings


def build_encoder(encoder_type: str) -> nn.Module:
    """Factory for QR-level encoders.

    Supported types:
        - "st": SentenceTransformerEncoder (frozen all-mpnet-base-v2)
        - "bert-pt": frozen E5
        - "bert-ft1": BERT last layer trainable
        - "bert-ft2": BERT last two layers trainable
        - "roberta-pt": RoBERTa frozen
        - "roberta-ft1": RoBERTa last layer trainable
        - "roberta-ft2": RoBERTa last two layers trainable
    """
    match encoder_type:
        case "st":
            return SentenceTransformerEncoder()
        case "bert-pt":
            return BaselineTransformerEncoder(
                E5_ID, unfreeze_last_n=0, normalize_output=True
            )
        case "bert-ft1":
            return BaselineTransformerEncoder(BERT_ID, unfreeze_last_n=1)
        case "bert-ft2":
            return BaselineTransformerEncoder(BERT_ID, unfreeze_last_n=2)
        case "roberta-pt":
            return BaselineTransformerEncoder(ROBERTA_ID, unfreeze_last_n=0)
        case "roberta-ft1":
            return BaselineTransformerEncoder(ROBERTA_ID, unfreeze_last_n=1)
        case "roberta-ft2":
            return BaselineTransformerEncoder(ROBERTA_ID, unfreeze_last_n=2)
        case _:
            raise ValueError(f"Unknown encoder type: {encoder_type}")


def encoder_trainable_params(encoder: nn.Module) -> int:
    """Return the number of trainable parameters in an encoder."""
    return sum(p.numel() for p in encoder.parameters() if p.requires_grad)
