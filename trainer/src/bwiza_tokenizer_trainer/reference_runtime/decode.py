from __future__ import annotations

from collections.abc import Sequence

from ..byte_fallback import parse_byte_fallback_piece
from ..types import ModelV1, NormalizationConfig


def decode_pieces(
    pieces: Sequence[str],
    model: ModelV1 | None = None,
) -> str:
    """Decode piece surfaces into normalized readable text."""

    boundary_marker = (
        model.normalization.boundary_marker
        if model is not None
        else NormalizationConfig().boundary_marker
    )

    decoded_parts: list[str] = []
    pending_bytes = bytearray()

    def flush_pending_bytes() -> None:
        if not pending_bytes:
            return

        decoded_parts.append(pending_bytes.decode("utf-8", errors="replace"))
        pending_bytes.clear()

    for piece in pieces:
        byte_value = parse_byte_fallback_piece(piece)
        if byte_value is not None:
            pending_bytes.append(byte_value)
            continue

        flush_pending_bytes()
        decoded_parts.append(piece)

    flush_pending_bytes()

    decoded = "".join(decoded_parts).replace(boundary_marker, " ")
    if decoded.startswith(" "):
        return decoded[1:]

    return decoded


def decode_ids(
    ids: Sequence[int],
    model: ModelV1,
) -> str:
    """Decode token ids through the loaded model artifact."""

    id_to_piece = {entry.id: entry.piece for entry in model.vocab}

    try:
        pieces = [id_to_piece[token_id] for token_id in ids]
    except KeyError as exc:
        raise ValueError(f"unknown token id {exc.args[0]}") from exc

    return decode_pieces(pieces, model=model)
