from __future__ import annotations

import importlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..config import TrainerConfig
from ..export import model_to_dict
from ..types import ModelV1, VocabEntry
from .candidates import SeedCandidate

_NATIVE_MODULE = "bwiza_tokenizer_runtime"
_NATIVE_BATCH_SIZE = 8192


@dataclass(slots=True)
class NativeTokenizerHandle:
    tokenizer: Any
    native_to_model_ids: dict[int, int]


def _reindex_model_for_native(model: ModelV1) -> tuple[ModelV1, dict[int, int]]:
    special_order = {"unk": 0, "bos": 1, "eos": 2, "pad": 3}
    special_entries = sorted(
        (entry for entry in model.vocab if entry.special is not None),
        key=lambda entry: special_order[entry.special],
    )
    regular_entries = sorted(
        (entry for entry in model.vocab if entry.special is None),
        key=lambda entry: entry.id,
    )

    ordered = special_entries + regular_entries
    reindexed_vocab: list[VocabEntry] = []
    special_token_ids: dict[str, int] = {}
    native_to_model_ids: dict[int, int] = {}

    for new_id, entry in enumerate(ordered):
        reindexed_vocab.append(
            VocabEntry(
                id=new_id,
                piece=entry.piece,
                score=entry.score,
                special=entry.special,
            )
        )
        native_to_model_ids[new_id] = entry.id
        if entry.special is not None:
            special_token_ids[entry.special] = new_id

    native_model = ModelV1(
        version=model.version,
        name=model.name,
        model_type=model.model_type,
        vocab_size=len(reindexed_vocab),
        normalization=model.normalization,
        special_token_ids=special_token_ids,  # type: ignore[arg-type]
        vocab=reindexed_vocab,
        trainer=dict(model.trainer),
    )

    return native_model, native_to_model_ids


def load_native_tokenizer(model: ModelV1) -> NativeTokenizerHandle | None:
    module = _load_native_module()
    if module is None:
        return None

    tokenizer_type = getattr(module, "Tokenizer", None)
    if tokenizer_type is None or not hasattr(tokenizer_type, "from_json"):
        return None

    native_model, native_to_model_ids = _reindex_model_for_native(model)
    payload = json.dumps(model_to_dict(native_model), ensure_ascii=False)
    return NativeTokenizerHandle(
        tokenizer=tokenizer_type.from_json(payload),
        native_to_model_ids=native_to_model_ids,
    )


def _load_native_module() -> Any | None:
    try:
        return importlib.import_module(_NATIVE_MODULE)
    except ModuleNotFoundError:
        return None


def count_piece_usage_native(
    normalized_docs: Iterable[str],
    model: ModelV1,
    *,
    chunk_size: int = _NATIVE_BATCH_SIZE,
) -> dict[int, int] | None:
    handle = load_native_tokenizer(model)
    if handle is None:
        return None

    counts: Counter[int] = Counter()
    chunk: list[str] = []

    for normalized in normalized_docs:
        chunk.append(normalized)
        if len(chunk) >= chunk_size:
            _update_counts_from_chunk(counts, handle, chunk)
            chunk.clear()

    if chunk:
        _update_counts_from_chunk(counts, handle, chunk)

    return dict(counts)


def enumerate_seed_candidates_native(
    normalized_docs: Iterable[str],
    config: TrainerConfig,
) -> list[SeedCandidate] | None:
    module = _load_native_module()
    if module is None or not hasattr(module, "enumerate_seed_candidates_normalized"):
        return None

    rows = module.enumerate_seed_candidates_normalized(
        _materialize_texts(normalized_docs),
        config.max_piece_chars,
        config.min_candidate_freq,
        config.seed_candidate_limit,
    )
    return [
        SeedCandidate(piece=str(piece), count=int(count), protected=bool(protected))
        for piece, count, protected in rows
    ]


def _update_counts_from_chunk(
    counts: Counter[int],
    handle: NativeTokenizerHandle,
    chunk: list[str],
) -> None:
    dense_method = getattr(handle.tokenizer, "count_piece_usage_normalized_dense", None)
    if dense_method is not None:
        dense_counts = dense_method(chunk)
        for native_token_id, count in enumerate(dense_counts):
            if count:
                counts[handle.native_to_model_ids[native_token_id]] += int(count)
        return

    sparse_counts = handle.tokenizer.count_piece_usage_normalized(chunk)
    counts.update(
        {
            handle.native_to_model_ids[int(token_id)]: int(count)
            for token_id, count in sparse_counts.items()
        }
    )


def _materialize_texts(texts: Iterable[str]) -> list[str]:
    if isinstance(texts, list):
        return texts

    return list(texts)
