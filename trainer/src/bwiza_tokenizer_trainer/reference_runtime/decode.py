from __future__ import annotations

from collections.abc import Sequence

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

    decoded = "".join(pieces).replace(boundary_marker, " ")
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
