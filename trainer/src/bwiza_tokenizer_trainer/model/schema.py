from __future__ import annotations

from typing import Final

from ..types import default_special_token_ids

MODEL_VERSION: Final[str] = "model-v1"
MODEL_TYPE: Final[str] = "unigram"
REQUIRED_MODEL_FIELDS: Final[tuple[str, ...]] = (
    "version",
    "name",
    "model_type",
    "vocab_size",
    "normalization",
    "special_token_ids",
    "vocab",
    "trainer",
)
REQUIRED_NORMALIZATION_FIELDS: Final[tuple[str, ...]] = (
    "unicode_form",
    "whitespace_policy",
    "trim",
    "boundary_marker",
    "prepend_leading_boundary",
)
REQUIRED_SPECIAL_TOKEN_IDS: Final[dict[str, int]] = default_special_token_ids()
ALLOWED_SPECIAL_NAMES: Final[frozenset[str]] = frozenset(
    REQUIRED_SPECIAL_TOKEN_IDS.keys(),
)
