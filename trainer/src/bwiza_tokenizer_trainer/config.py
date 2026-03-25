from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrainerConfig:
    """Phase-0 placeholder trainer configuration."""

    vocab_size: int = 0
