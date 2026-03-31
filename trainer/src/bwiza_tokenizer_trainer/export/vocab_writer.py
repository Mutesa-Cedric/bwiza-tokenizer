from __future__ import annotations

from pathlib import Path

from ..types import ModelV1


def _score_to_text(score: float) -> str:
    return format(score, ".17g")


def write_vocab_tsv(
    model: ModelV1,
    path: str | Path,
) -> Path:
    destination = Path(path)
    lines = ["id\tpiece\tscore\tspecial"]

    for entry in model.vocab:
        special = "" if entry.special is None else entry.special
        lines.append(
            f"{entry.id}\t{entry.piece}\t{_score_to_text(entry.score)}\t{special}"
        )

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
