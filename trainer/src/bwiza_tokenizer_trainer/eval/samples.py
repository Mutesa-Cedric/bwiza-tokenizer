from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..normalize.pipeline import normalize_text
from ..reference_runtime.decode import decode_pieces
from ..types import ModelV1
from ..train.viterbi import segment_normalized


def sample_segmentations(
    docs: Iterable[str],
    model: ModelV1,
    limit: int = 8,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for doc in docs:
        normalized = normalize_text(doc, config=model.normalization)
        if not normalized:
            continue

        segmentation = segment_normalized(normalized, model)
        samples.append(
            {
                "input": doc,
                "normalized": normalized,
                "pieces": list(segmentation.pieces),
                "ids": list(segmentation.ids),
                "decoded": decode_pieces(segmentation.pieces, model=model),
            }
        )

        if len(samples) >= limit:
            break

    return samples
