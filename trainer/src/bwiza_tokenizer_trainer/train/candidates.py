from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from ..config import TrainerConfig


@dataclass(frozen=True, slots=True)
class SeedCandidate:
    piece: str
    count: int
    protected: bool


def enumerate_seed_candidates(
    normalized_docs: Iterable[str],
    config: TrainerConfig,
) -> list[SeedCandidate]:
    counts: Counter[str] = Counter()
    protected_pieces: set[str] = set()

    for normalized in normalized_docs:
        if not isinstance(normalized, str):
            raise TypeError("normalized_docs must yield strings")

        if not normalized:
            continue

        protected_pieces.update(normalized)

        for start in range(len(normalized)):
            max_end = min(len(normalized), start + config.max_piece_chars)
            for end in range(start + 1, max_end + 1):
                counts[normalized[start:end]] += 1

    protected_candidates = [
        SeedCandidate(piece=piece, count=counts[piece], protected=True)
        for piece in sorted(protected_pieces)
    ]

    remaining_slots = max(config.seed_candidate_limit - len(protected_candidates), 0)

    ranked_candidates = sorted(
        (
            SeedCandidate(piece=piece, count=count, protected=False)
            for piece, count in counts.items()
            if piece not in protected_pieces and count >= config.min_candidate_freq
        ),
        key=lambda candidate: (-candidate.count, candidate.piece),
    )

    if remaining_slots == 0:
        return protected_candidates

    return protected_candidates + ranked_candidates[:remaining_slots]
