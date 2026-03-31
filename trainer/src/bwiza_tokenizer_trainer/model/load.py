from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ..types import ModelV1, NormalizationConfig, VocabEntry
from .schema import (
    REQUIRED_MODEL_FIELDS,
    REQUIRED_NORMALIZATION_FIELDS,
    REQUIRED_SPECIAL_TOKEN_IDS,
)
from .validate import ModelValidationError, validate_model


def load_model(path: str | Path) -> ModelV1:
    """Load a tokenizer model from JSON on disk."""

    model_path = Path(path)
    with model_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    return load_model_dict(raw)


def load_model_dict(raw: object) -> ModelV1:
    """Load a tokenizer model from a decoded JSON object."""

    root = _require_mapping(raw, context="model")
    _require_fields(root, REQUIRED_MODEL_FIELDS, context="model")

    normalization_raw = _require_mapping(root["normalization"], context="normalization")
    _require_fields(
        normalization_raw,
        REQUIRED_NORMALIZATION_FIELDS,
        context="normalization",
    )

    special_ids_raw = _require_mapping(
        root["special_token_ids"],
        context="special_token_ids",
    )
    _require_fields(
        special_ids_raw,
        tuple(REQUIRED_SPECIAL_TOKEN_IDS.keys()),
        context="special_token_ids",
    )

    vocab_raw = root["vocab"]
    if not isinstance(vocab_raw, list):
        raise ModelValidationError("vocab must be a list")

    trainer_raw = _require_mapping(root["trainer"], context="trainer")

    normalization = NormalizationConfig(
        unicode_form=_require_str(
            normalization_raw["unicode_form"],
            context="normalization.unicode_form",
        ),
        whitespace_policy=_require_str(
            normalization_raw["whitespace_policy"],
            context="normalization.whitespace_policy",
        ),
        trim=_require_bool(normalization_raw["trim"], context="normalization.trim"),
        boundary_marker=_require_str(
            normalization_raw["boundary_marker"],
            context="normalization.boundary_marker",
        ),
        prepend_leading_boundary=_require_bool(
            normalization_raw["prepend_leading_boundary"],
            context="normalization.prepend_leading_boundary",
        ),
    )

    special_token_ids = {
        name: _require_int(special_ids_raw[name], context=f"special_token_ids.{name}")
        for name in REQUIRED_SPECIAL_TOKEN_IDS
    }

    vocab = [
        _load_vocab_entry(entry, index=index)
        for index, entry in enumerate(vocab_raw)
    ]

    model = ModelV1(
        version=_require_str(root["version"], context="version"),
        name=_require_str(root["name"], context="name"),
        model_type=_require_str(root["model_type"], context="model_type"),
        vocab_size=_require_int(root["vocab_size"], context="vocab_size"),
        normalization=normalization,
        special_token_ids=special_token_ids,
        vocab=vocab,
        trainer=dict(trainer_raw),
    )

    validate_model(model)
    return model


def _load_vocab_entry(entry: object, *, index: int) -> VocabEntry:
    raw = _require_mapping(entry, context=f"vocab[{index}]")
    _require_fields(raw, ("id", "piece", "score", "special"), context=f"vocab[{index}]")

    special = raw["special"]
    if special is not None and not isinstance(special, str):
        raise ModelValidationError(f"vocab[{index}].special must be string or null")

    return VocabEntry(
        id=_require_int(raw["id"], context=f"vocab[{index}].id"),
        piece=_require_str(raw["piece"], context=f"vocab[{index}].piece"),
        score=_require_number(raw["score"], context=f"vocab[{index}].score"),
        special=special,
    )


def _require_mapping(raw: object, *, context: str) -> Mapping[str, object]:
    if isinstance(raw, Mapping):
        return raw

    raise ModelValidationError(f"{context} must be a JSON object")


def _require_fields(
    raw: Mapping[str, object],
    fields: tuple[str, ...],
    *,
    context: str,
) -> None:
    for field in fields:
        if field not in raw:
            raise ModelValidationError(f"{context} is missing field {field!r}")


def _require_str(raw: object, *, context: str) -> str:
    if isinstance(raw, str):
        return raw

    raise ModelValidationError(f"{context} must be a string")


def _require_bool(raw: object, *, context: str) -> bool:
    if isinstance(raw, bool):
        return raw

    raise ModelValidationError(f"{context} must be a boolean")


def _require_int(raw: object, *, context: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ModelValidationError(f"{context} must be an integer")

    return raw


def _require_number(raw: object, *, context: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ModelValidationError(f"{context} must be a number")

    return float(raw)
