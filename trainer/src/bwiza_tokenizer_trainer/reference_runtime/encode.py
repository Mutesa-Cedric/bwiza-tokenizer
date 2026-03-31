from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..normalize.pipeline import normalize_text
from ..types import ModelV1, VocabEntry


@dataclass(frozen=True, slots=True)
class _Segmentation:
    score: float
    entries: tuple[VocabEntry, ...]


def _unknown_entry(model: ModelV1) -> VocabEntry:
    unknown_id = model.special_token_ids["unk"]

    for entry in model.vocab:
        if entry.id == unknown_id:
            return entry

    raise ValueError("model is missing the required <unk> entry")


def _match_candidates(
    normalized: str,
    offset: int,
    model: ModelV1,
) -> list[VocabEntry]:
    matches: list[VocabEntry] = []

    for entry in model.vocab:
        if entry.special is not None:
            continue

        if normalized.startswith(entry.piece, offset):
            matches.append(entry)

    return matches


def _is_better(left: _Segmentation, right: _Segmentation) -> bool:
    if left.score != right.score:
        return left.score > right.score

    if len(left.entries) != len(right.entries):
        return len(left.entries) < len(right.entries)

    for left_entry, right_entry in zip(left.entries, right.entries):
        if left_entry.id == right_entry.id:
            continue

        if len(left_entry.piece) != len(right_entry.piece):
            return len(left_entry.piece) > len(right_entry.piece)

        return left_entry.id < right_entry.id

    return False


def _segment_normalized(normalized: str, model: ModelV1) -> tuple[VocabEntry, ...]:
    unknown = _unknown_entry(model)

    @lru_cache(maxsize=None)
    def best_from(offset: int) -> _Segmentation:
        if offset >= len(normalized):
            return _Segmentation(score=0.0, entries=())

        best: _Segmentation | None = None
        candidates = _match_candidates(normalized, offset, model)

        if not candidates:
            suffix = best_from(offset + 1)
            return _Segmentation(
                score=unknown.score + suffix.score,
                entries=(unknown, *suffix.entries),
            )

        for entry in candidates:
            suffix = best_from(offset + len(entry.piece))
            candidate = _Segmentation(
                score=entry.score + suffix.score,
                entries=(entry, *suffix.entries),
            )
            if best is None or _is_better(candidate, best):
                best = candidate

        assert best is not None
        return best

    return best_from(0).entries


def encode_to_pieces(text: str, model: ModelV1) -> list[str]:
    normalized = normalize_text(text, config=model.normalization)
    return [entry.piece for entry in _segment_normalized(normalized, model)]


def encode_to_ids(text: str, model: ModelV1) -> list[int]:
    normalized = normalize_text(text, config=model.normalization)
    return [entry.id for entry in _segment_normalized(normalized, model)]
