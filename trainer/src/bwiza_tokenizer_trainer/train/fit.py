from __future__ import annotations

from dataclasses import asdict
from math import log
from typing import Iterable

from ..config import TrainerConfig
from ..model.validate import validate_model
from ..normalize.pipeline import normalize_text
from ..types import ModelV1, NormalizationConfig, VocabEntry
from .candidates import SeedCandidate, enumerate_seed_candidates
from .counts import count_piece_usage
from .prune import is_protected_piece, prune_vocabulary
from .viterbi import segment_normalized

SPECIAL_TOKEN_ROWS: tuple[tuple[str, str, float], ...] = (
    ("unk", "<unk>", 0.0),
    ("bos", "<s>", 0.0),
    ("eos", "</s>", 0.0),
    ("pad", "<pad>", 0.0),
)
UNUSED_PIECE_SCORE = -1_000_000_000.0


def _special_entries() -> list[VocabEntry]:
    return [
        VocabEntry(id=index, piece=piece, score=score, special=special)
        for index, (special, piece, score) in enumerate(SPECIAL_TOKEN_ROWS)
    ]


def _score_seed_candidates(candidates: list[SeedCandidate]) -> list[VocabEntry]:
    total_count = sum(candidate.count for candidate in candidates)

    if total_count == 0:
        return []

    entries: list[VocabEntry] = []
    next_id = len(SPECIAL_TOKEN_ROWS)

    for candidate in candidates:
        entries.append(
            VocabEntry(
                id=next_id,
                piece=candidate.piece,
                score=log(candidate.count / total_count),
                special=None,
            )
        )
        next_id += 1

    return entries


def _recompute_scores(
    vocab: list[VocabEntry],
    usage_counts: dict[int, int],
) -> list[VocabEntry]:
    vocab_by_id = {entry.id: entry for entry in vocab}
    total_usage = sum(
        count
        for token_id, count in usage_counts.items()
        if vocab_by_id[token_id].special is None
    )

    updated: list[VocabEntry] = []

    for entry in vocab:
        if entry.special is not None:
            updated.append(entry)
            continue

        usage = usage_counts.get(entry.id, 0)
        score = UNUSED_PIECE_SCORE
        if usage > 0 and total_usage > 0:
            score = log(usage / total_usage)

        updated.append(
            VocabEntry(
                id=entry.id,
                piece=entry.piece,
                score=score,
                special=None,
            )
        )

    return updated


def _max_score_delta(previous: list[VocabEntry], current: list[VocabEntry]) -> float:
    previous_by_id = {entry.id: entry.score for entry in previous if entry.special is None}
    current_by_id = {entry.id: entry.score for entry in current if entry.special is None}

    shared_ids = previous_by_id.keys() & current_by_id.keys()
    if not shared_ids:
        return 0.0

    return max(abs(previous_by_id[token_id] - current_by_id[token_id]) for token_id in shared_ids)


def _final_trim(
    vocab: list[VocabEntry],
    target_vocab_size: int,
    boundary_marker: str,
) -> list[VocabEntry]:
    if len(vocab) <= target_vocab_size:
        return vocab

    protected = [entry for entry in vocab if is_protected_piece(entry, boundary_marker=boundary_marker)]
    removable = [entry for entry in vocab if not is_protected_piece(entry, boundary_marker=boundary_marker)]

    removable_sorted = sorted(
        removable,
        key=lambda entry: (entry.score, len(entry.piece), entry.piece, entry.id),
    )

    keep_count = max(target_vocab_size - len(protected), 0)
    kept_removable = sorted(
        removable_sorted[-keep_count:],
        key=lambda entry: entry.id,
    )

    retained_ids = {entry.id for entry in protected + kept_removable}
    return [entry for entry in vocab if entry.id in retained_ids]


def _reindex_vocab(vocab: list[VocabEntry]) -> tuple[list[VocabEntry], dict[str, int]]:
    special_order = {"unk": 0, "bos": 1, "eos": 2, "pad": 3}
    special_entries = sorted(
        (entry for entry in vocab if entry.special is not None),
        key=lambda entry: special_order[entry.special],
    )
    regular_entries = sorted(
        (entry for entry in vocab if entry.special is None),
        key=lambda entry: entry.id,
    )

    ordered = special_entries + regular_entries
    reindexed: list[VocabEntry] = []
    special_token_ids: dict[str, int] = {}

    for new_id, entry in enumerate(ordered):
        reindexed_entry = VocabEntry(
            id=new_id,
            piece=entry.piece,
            score=entry.score,
            special=entry.special,
        )
        reindexed.append(reindexed_entry)
        if entry.special is not None:
            special_token_ids[entry.special] = new_id

    return reindexed, special_token_ids


def train_from_iterator(
    docs: Iterable[str],
    config: TrainerConfig,
) -> ModelV1:
    normalization = NormalizationConfig()
    normalized_docs = [
        normalized
        for normalized in (normalize_text(doc, config=normalization) for doc in docs)
        if normalized
    ]

    seed_candidates = enumerate_seed_candidates(normalized_docs, config)
    vocab = _special_entries() + _score_seed_candidates(seed_candidates)

    if len(vocab) == len(SPECIAL_TOKEN_ROWS):
        model = ModelV1(
            name="bwiza-unigram-v1",
            vocab_size=len(vocab),
            normalization=normalization,
            special_token_ids={"unk": 0, "bos": 1, "eos": 2, "pad": 3},
            vocab=vocab,
            trainer=asdict(config),
        )
        validate_model(model)
        return model

    current_vocab = vocab

    for _ in range(config.max_iterations):
        working_model = ModelV1(
            name="bwiza-unigram-v1",
            vocab_size=len(current_vocab),
            normalization=normalization,
            special_token_ids={"unk": 0, "bos": 1, "eos": 2, "pad": 3},
            vocab=current_vocab,
            trainer=asdict(config),
        )

        segmentations = [segment_normalized(doc, working_model) for doc in normalized_docs]
        usage_counts = count_piece_usage(segmentations)
        rescored_vocab = _recompute_scores(current_vocab, usage_counts)
        pruned_vocab = prune_vocabulary(
            rescored_vocab,
            prune_fraction=config.prune_fraction,
            boundary_marker=normalization.boundary_marker,
            minimum_vocab_size=max(config.vocab_size, len(SPECIAL_TOKEN_ROWS)),
        )

        score_delta = _max_score_delta(current_vocab, rescored_vocab)
        current_vocab = pruned_vocab

        if len(current_vocab) <= config.vocab_size and score_delta <= config.min_score_delta:
            break

    trimmed_vocab = _final_trim(
        current_vocab,
        target_vocab_size=config.vocab_size,
        boundary_marker=normalization.boundary_marker,
    )
    reindexed_vocab, special_token_ids = _reindex_vocab(trimmed_vocab)

    model = ModelV1(
        name="bwiza-unigram-v1",
        vocab_size=len(reindexed_vocab),
        normalization=normalization,
        special_token_ids=special_token_ids,  # type: ignore[arg-type]
        vocab=reindexed_vocab,
        trainer=asdict(config),
    )
    validate_model(model)
    return model
