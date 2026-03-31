from __future__ import annotations

from .config import TrainerConfig
from .normalize.pipeline import normalize_text as _normalize_text
from .reference_runtime.decode import decode_ids as _decode_ids
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


def normalize_text(
    text: str,
    config: NormalizationConfig | None = None,
) -> str:
    return _normalize_text(text, config=config)


def train_from_iterator(docs, config: TrainerConfig):
    raise NotImplementedError("train_from_iterator will be implemented in Phase 2.")


def encode_text(text: str, model: ModelV1) -> list[int]:
    raise NotImplementedError("encode_text will be implemented in Phase 2.")


def decode_ids(ids: list[int], model: ModelV1) -> str:
    return _decode_ids(ids, model)
