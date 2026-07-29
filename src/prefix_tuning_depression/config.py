"""Hyperparameters from Lau et al. (2023)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture hyperparameters."""

    transformer_pretrained_id: str = "sentence-transformers/all-mpnet-base-v2"
    prefix_backbone: str = "roberta-base"
    pre_seq_len: int = 10
    st_max_token_length: int = 256
    prefix_max_token_length: int = 128
    encoding_projection_size: int = 128
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    dropout_prob: float = 0.5
    num_labels: int = 1
    fusion_method: str = "average"
    chunk_size: int = 16

    @property
    def prefix_text_max_token_length(self) -> int:
        """Return the RoBERTa token budget after reserving prefix positions."""
        return self.prefix_max_token_length - self.pre_seq_len


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""

    seed: int = 0
    batch_size: int = 2
    learning_rate: float = 3e-4
    num_epochs: int = 200
    es_patience: int = 20
    optimizer: str = "AdamW"
    problem_type: str = "regression"
    num_workers: int = 2

    def replace(self, **kwargs) -> "TrainingConfig":
        """Return a copy with updated fields."""
        return type(self)(**(self.__dict__ | kwargs))


@dataclass(frozen=True)
class DataConfig:
    """Data paths and preprocessing."""

    data_root: str = "data"
    sessions_min: int = 300
    sessions_max: int = 492
    excluded_sessions: frozenset[int] = frozenset({342, 394, 398, 460})
