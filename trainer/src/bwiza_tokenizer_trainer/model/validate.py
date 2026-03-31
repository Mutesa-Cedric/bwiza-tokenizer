from __future__ import annotations

from ..types import ModelV1, NormalizationConfig
from .schema import ALLOWED_SPECIAL_NAMES, MODEL_TYPE, MODEL_VERSION, REQUIRED_SPECIAL_TOKEN_IDS


class ModelValidationError(ValueError):
    """Raised when a tokenizer model artifact violates the v1 contract."""


def validate_model(model: ModelV1) -> None:
    """Validate semantic rules that go beyond basic JSON shape."""

    if model.version != MODEL_VERSION:
        raise ModelValidationError(
            f"model version must be {MODEL_VERSION!r}, got {model.version!r}"
        )

    if model.model_type != MODEL_TYPE:
        raise ModelValidationError(
            f"model_type must be {MODEL_TYPE!r}, got {model.model_type!r}"
        )

    if model.normalization != NormalizationConfig():
        raise ModelValidationError("normalization config does not match normalization-v1")

    if model.special_token_ids != REQUIRED_SPECIAL_TOKEN_IDS:
        raise ModelValidationError("special_token_ids must match the fixed v1 mapping")

    if model.vocab_size != len(model.vocab):
        raise ModelValidationError(
            f"vocab_size must equal len(vocab), got {model.vocab_size} and {len(model.vocab)}"
        )

    ids = [entry.id for entry in model.vocab]
    expected_ids = list(range(len(model.vocab)))
    if sorted(ids) != expected_ids:
        raise ModelValidationError("vocab ids must be unique and contiguous from 0")

    pieces = [entry.piece for entry in model.vocab]
    if len(set(pieces)) != len(pieces):
        raise ModelValidationError("vocab pieces must be unique")

    for entry in model.vocab:
        if entry.special is not None and entry.special not in ALLOWED_SPECIAL_NAMES:
            raise ModelValidationError(f"unsupported special token name {entry.special!r}")

    for special_name, expected_id in REQUIRED_SPECIAL_TOKEN_IDS.items():
        matches = [entry for entry in model.vocab if entry.special == special_name]
        if len(matches) != 1:
            raise ModelValidationError(
                f"expected exactly one vocab entry with special={special_name!r}"
            )

        if matches[0].id != expected_id:
            raise ModelValidationError(
                f"special token {special_name!r} must use id {expected_id}"
            )
