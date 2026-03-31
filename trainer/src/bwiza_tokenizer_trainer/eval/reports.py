from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..types import ModelV1
from .metrics import compute_metrics
from .samples import sample_segmentations


def build_eval_report(
    docs: Iterable[str],
    model: ModelV1,
    sample_limit: int = 8,
) -> dict[str, Any]:
    docs_list = list(docs)
    metrics = compute_metrics(docs_list, model)
    samples = sample_segmentations(docs_list, model, limit=sample_limit)

    return {
        "model_name": model.name,
        "vocab_size": model.vocab_size,
        **metrics,
        "sample_segmentations": samples,
    }
