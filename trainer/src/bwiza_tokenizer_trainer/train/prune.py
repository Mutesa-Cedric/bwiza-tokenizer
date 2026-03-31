from __future__ import annotations

from collections.abc import Sequence

from ..types import VocabEntry


def is_protected_piece(
    entry: VocabEntry,
    boundary_marker: str = "▁",
) -> bool:
    if entry.special is not None:
        return True

    if len(entry.piece) == 1:
        return True

    return entry.piece == boundary_marker


def split_protected_entries(
    vocab: Sequence[VocabEntry],
    boundary_marker: str = "▁",
) -> tuple[list[VocabEntry], list[VocabEntry]]:
    protected: list[VocabEntry] = []
    removable: list[VocabEntry] = []

    for entry in vocab:
        if is_protected_piece(entry, boundary_marker=boundary_marker):
            protected.append(entry)
        else:
            removable.append(entry)

    return protected, removable


def prune_vocabulary(
    vocab: Sequence[VocabEntry],
    prune_fraction: float,
    boundary_marker: str = "▁",
    minimum_vocab_size: int | None = None,
) -> list[VocabEntry]:
    if not 0.0 <= prune_fraction <= 1.0:
        raise ValueError("prune_fraction must be between 0.0 and 1.0")

    protected, removable = split_protected_entries(
        vocab,
        boundary_marker=boundary_marker,
    )

    removable_sorted = sorted(
        removable,
        key=lambda entry: (entry.score, len(entry.piece), entry.piece, entry.id),
    )

    prune_count = int(len(removable_sorted) * prune_fraction)

    if prune_fraction > 0.0 and prune_count == 0 and removable_sorted:
        prune_count = 1

    if minimum_vocab_size is not None:
        max_prunable = max(len(vocab) - minimum_vocab_size, 0)
        prune_count = min(prune_count, max_prunable)

    pruned_ids = {entry.id for entry in removable_sorted[:prune_count]}
    retained = [entry for entry in vocab if entry.id not in pruned_ids]

    return retained
