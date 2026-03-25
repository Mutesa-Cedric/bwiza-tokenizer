from __future__ import annotations

from .config import TrainerConfig
from .types import ModelV1, NormalizationConfig, VocabEntry

__all__ = [
    "ModelV1",
    "NormalizationConfig",
    "TrainerConfig",
    "VocabEntry",
    "decode_ids",
    "encode_text",
    "normalize_text",
    "train_from_iterator",
]


def normalize_text(text: str) -> str:
    raise NotImplementedError("normalize_text will be implemented in Phase 2.")


def train_from_iterator(docs, config: TrainerConfig):
    raise NotImplementedError("train_from_iterator will be implemented in Phase 2.")


def encode_text(text: str, model: ModelV1) -> list[int]:
    raise NotImplementedError("encode_text will be implemented in Phase 2.")


def decode_ids(ids: list[int], model: ModelV1) -> str:
    raise NotImplementedError("decode_ids will be implemented in Phase 2.")
