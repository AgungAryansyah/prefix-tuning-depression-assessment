"""DoRA-adapted DeBERTa QR encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel


class DebertaDoraEncoder(nn.Module):
    """Frozen DeBERTa encoder with trainable DoRA attention adapters."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        for param in backbone.parameters():
            param.requires_grad = False
        self.encoder = get_peft_model(
            backbone,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                use_dora=True,
                target_modules=["query_proj", "value_proj"],
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a masked mean-pooled last-layer representation."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        return (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def build_prefix_encoder(
    prefix_backbone: str = "microsoft/deberta-v3-base",
    pre_seq_len: int = 10,
) -> DebertaDoraEncoder:
    """Create a DoRA-adapted DeBERTa encoder."""
    del pre_seq_len
    return DebertaDoraEncoder(AutoModel.from_pretrained(prefix_backbone))


def count_trainable_params(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
