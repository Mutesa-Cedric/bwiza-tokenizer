from __future__ import annotations

from collections.abc import Iterable

from ..normalize.pipeline import normalize_text
from ..types import ModelV1
from ..train.viterbi import build_model_index, iter_segment_ids


def compute_metrics(
    docs: Iterable[str],
    model: ModelV1,
) -> dict[str, float]:
    model_index = build_model_index(model)
    unknown_id = model.special_token_ids["unk"]
    total_chars = 0
    total_tokens = 0
    total_docs = 0
    unknown_tokens = 0
    used_ids: set[int] = set()

    for doc in docs:
        normalized = normalize_text(doc, config=model.normalization)
        if not normalized:
            continue

        total_docs += 1
        total_chars += len(normalized)

        for token_id in iter_segment_ids(normalized, model, model_index):
            total_tokens += 1
            unknown_tokens += token_id == unknown_id
            used_ids.add(token_id)

    if total_docs == 0:
        return {
            "average_chars_per_token": 0.0,
            "average_tokens_per_document": 0.0,
            "unknown_rate": 0.0,
            "vocab_utilization": 0.0,
        }

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
