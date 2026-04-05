from __future__ import annotations

from .types import VocabEntry

BYTE_FALLBACK_PREFIX = "\uE000"
BYTE_FALLBACK_SCORE = -100.0
BYTE_FALLBACK_COUNT = 256


def byte_fallback_piece(byte_value: int) -> str:
    if not 0 <= byte_value < BYTE_FALLBACK_COUNT:
        raise ValueError("byte fallback piece requires a byte value between 0 and 255")

    return f"{BYTE_FALLBACK_PREFIX}{byte_value:02X}"


def parse_byte_fallback_piece(piece: str) -> int | None:
    if len(piece) != 3 or not piece.startswith(BYTE_FALLBACK_PREFIX):
        return None

    try:
        value = int(piece[1:], 16)
    except ValueError:
        return None

    if not 0 <= value < BYTE_FALLBACK_COUNT:
        return None

    return value


def is_byte_fallback_piece(piece: str) -> bool:
    return parse_byte_fallback_piece(piece) is not None


def byte_fallback_entries(start_id: int) -> list[VocabEntry]:
    return [
        VocabEntry(
            id=start_id + byte_value,
            piece=byte_fallback_piece(byte_value),
            score=BYTE_FALLBACK_SCORE,
            special=None,
        )
        for byte_value in range(BYTE_FALLBACK_COUNT)
    ]
