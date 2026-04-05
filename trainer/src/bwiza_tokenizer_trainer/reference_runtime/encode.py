from __future__ import annotations

from ..normalize.pipeline import normalize_text
from ..train.viterbi import build_model_index, iter_segment_ids, segment_normalized
from ..types import ModelV1


def encode_to_pieces(text: str, model: ModelV1) -> list[str]:
    normalized = normalize_text(text, config=model.normalization)
    model_index = build_model_index(model)
    return list(segment_normalized(normalized, model, model_index).pieces)


def encode_to_ids(text: str, model: ModelV1) -> list[int]:
    normalized = normalize_text(text, config=model.normalization)
    model_index = build_model_index(model)
    return list(iter_segment_ids(normalized, model, model_index))
