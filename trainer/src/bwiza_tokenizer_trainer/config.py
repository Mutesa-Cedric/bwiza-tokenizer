from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class TrainerConfig:
    """Deterministic unigram trainer settings for tokenizer v1."""

    model_type: Literal["unigram"] = "unigram"
    vocab_size: int = 16000
    seed_candidate_multiplier: int = 8
    seed_candidate_limit: int = 128000
    max_piece_chars: int = 24
    min_candidate_freq: int = 2
    prune_fraction: float = 0.15
    max_iterations: int = 20
    min_score_delta: float = 1e-4
    byte_fallback: bool = True
    reserved_special_tokens: Literal[4] = 4
