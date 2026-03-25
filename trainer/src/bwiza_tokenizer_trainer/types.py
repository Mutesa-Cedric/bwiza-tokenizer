from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Phase-0 placeholder normalization configuration."""

    unicode_form: str = "NFC"
    whitespace_policy: str = "all_whitespace_to_space_then_collapse"
    trim: bool = True
    boundary_marker: str = "▁"
    prepend_leading_boundary: bool = True


@dataclass(frozen=True, slots=True)
class VocabEntry:
    """Phase-0 placeholder vocab entry."""

    id: int
    piece: str
    score: float = 0.0
    special: str | None = None


@dataclass(slots=True)
class ModelV1:
    """Phase-0 placeholder model artifact representation."""

    version: str = "model-v1"
    name: str = "bwiza-unigram-v1"
    model_type: str = "unigram"
    vocab_size: int = 0
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    special_token_ids: dict[str, int] = field(
        default_factory=lambda: {"unk": 0, "bos": 1, "eos": 2, "pad": 3}
    )
    vocab: list[VocabEntry] = field(default_factory=list)
    trainer: dict[str, Any] = field(default_factory=dict)
