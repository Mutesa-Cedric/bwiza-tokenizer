from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .viterbi import SegmentationResult


def count_piece_usage(segmentations: Iterable[SegmentationResult]) -> dict[int, int]:
    counts: Counter[int] = Counter()

    for segmentation in segmentations:
        for token_id in segmentation.ids:
            counts[token_id] += 1

    return dict(counts)
