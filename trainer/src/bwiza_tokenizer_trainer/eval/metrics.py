from __future__ import annotations

from collections.abc import Iterable

from ..normalize.pipeline import normalize_text
from ..types import ModelV1
from ..train.viterbi import segment_normalized


def compute_metrics(
    docs: Iterable[str],
    model: ModelV1,
) -> dict[str, float]:
    normalized_docs = [
        normalized
        for normalized in (normalize_text(doc, config=model.normalization) for doc in docs)
        if normalized
    ]

    if not normalized_docs:
        return {
            "average_chars_per_token": 0.0,
            "average_tokens_per_document": 0.0,
            "unknown_rate": 0.0,
            "vocab_utilization": 0.0,
        }

    segmentations = [segment_normalized(doc, model) for doc in normalized_docs]
    total_chars = sum(len(doc) for doc in normalized_docs)
    total_tokens = sum(len(segmentation.ids) for segmentation in segmentations)
    unknown_id = model.special_token_ids["unk"]
    unknown_tokens = sum(
        1
        for segmentation in segmentations
        for token_id in segmentation.ids
        if token_id == unknown_id
    )
    used_ids = {
        token_id
        for segmentation in segmentations
        for token_id in segmentation.ids
    }

    if total_tokens == 0:
        average_chars_per_token = 0.0
        unknown_rate = 0.0
    else:
        average_chars_per_token = total_chars / total_tokens
        unknown_rate = unknown_tokens / total_tokens

    return {
        "average_chars_per_token": average_chars_per_token,
        "average_tokens_per_document": total_tokens / len(normalized_docs),
        "unknown_rate": unknown_rate,
        "vocab_utilization": len(used_ids) / len(model.vocab),
    }
