from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .viterbi import SegmentationResult


def count_piece_usage(
    segmented_ids: Iterable[SegmentationResult | Iterable[int]],
) -> dict[int, int]:
    counts: Counter[int] = Counter()

    for segmented in segmented_ids:
        token_ids = segmented.ids if isinstance(segmented, SegmentationResult) else segmented
        for token_id in token_ids:
            counts[token_id] += 1

    return dict(counts)
