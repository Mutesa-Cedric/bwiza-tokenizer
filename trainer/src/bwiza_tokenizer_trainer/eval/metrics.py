from __future__ import annotations

from collections.abc import Iterable

from ..normalize.pipeline import normalize_text
from ..types import ModelV1
from ..train.counts import count_piece_usage
from ..train.native_runtime import count_piece_usage_native
from ..train.viterbi import build_model_index, iter_segment_ids


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

    unknown_id = model.special_token_ids["unk"]
    total_chars = sum(len(normalized) for normalized in normalized_docs)
    total_docs = len(normalized_docs)
    usage_counts = count_piece_usage_native(normalized_docs, model)

    if usage_counts is None:
        model_index = build_model_index(model)
        usage_counts = count_piece_usage(
            iter_segment_ids(normalized, model, model_index)
            for normalized in normalized_docs
        )

    total_tokens = sum(usage_counts.values())
    unknown_tokens = usage_counts.get(unknown_id, 0)
    used_ids = set(usage_counts)

    if total_tokens == 0:
        average_chars_per_token = 0.0
        unknown_rate = 0.0
    else:
        average_chars_per_token = total_chars / total_tokens
        unknown_rate = unknown_tokens / total_tokens

    return {
        "average_chars_per_token": average_chars_per_token,
        "average_tokens_per_document": total_tokens / total_docs,
        "unknown_rate": unknown_rate,
        "vocab_utilization": len(used_ids) / len(model.vocab),
    }
