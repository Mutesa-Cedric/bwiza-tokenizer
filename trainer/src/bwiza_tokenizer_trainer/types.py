from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

SpecialTokenName: TypeAlias = Literal["unk", "bos", "eos", "pad"]


def default_special_token_ids() -> dict[SpecialTokenName, int]:
    return {"unk": 0, "bos": 1, "eos": 2, "pad": 3}


def default_trainer_metadata() -> dict[str, Any]:
    return {
        "target_vocab_size": 16000,
        "seed_candidate_limit": 128000,
        "max_piece_chars": 24,
        "min_candidate_freq": 2,
        "prune_fraction": 0.15,
        "max_iterations": 20,
        "min_score_delta": 1e-4,
        "seed_candidate_multiplier": 8,
        "reserved_special_tokens": 4,
    }


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Normalization contract shared by the trainer and runtime."""

    unicode_form: str = "NFC"
    whitespace_policy: str = "all_whitespace_to_space_then_collapse"
    trim: bool = True
    boundary_marker: str = "▁"
    prepend_leading_boundary: bool = True


@dataclass(frozen=True, slots=True)
class VocabEntry:
    """One tokenizer vocabulary row, including special-token entries."""

    id: int
    piece: str
    score: float = 0.0
    special: SpecialTokenName | None = None


@dataclass(slots=True)
class ModelV1:
    """Python representation of the v1 tokenizer artifact."""

    version: Literal["model-v1"] = "model-v1"
    name: str = "bwiza-unigram-v1"
    model_type: Literal["unigram"] = "unigram"
    vocab_size: int = 16000
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    special_token_ids: dict[SpecialTokenName, int] = field(default_factory=default_special_token_ids)
    vocab: list[VocabEntry] = field(default_factory=list)
    trainer: dict[str, Any] = field(default_factory=default_trainer_metadata)
