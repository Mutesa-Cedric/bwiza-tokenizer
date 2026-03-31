from __future__ import annotations

from ..normalize.pipeline import normalize_text
from ..train.viterbi import segment_normalized
from ..types import ModelV1


def encode_to_pieces(text: str, model: ModelV1) -> list[str]:
    normalized = normalize_text(text, config=model.normalization)
    return list(segment_normalized(normalized, model).pieces)


def encode_to_ids(text: str, model: ModelV1) -> list[int]:
    normalized = normalize_text(text, config=model.normalization)
    return list(segment_normalized(normalized, model).ids)
