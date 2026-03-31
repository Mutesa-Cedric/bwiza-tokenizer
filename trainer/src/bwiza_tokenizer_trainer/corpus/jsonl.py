from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .base import PathLike, ensure_text_document


def iter_jsonl_field(path: PathLike, field: str, *, encoding: str = "utf-8") -> Iterator[str]:
    """Yield one string field from each JSON object line."""

    file_path = Path(path)
    with file_path.open("r", encoding=encoding, newline=None) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {file_path}:{line_number}") from exc

            if not isinstance(record, dict):
                raise TypeError(f"{file_path}:{line_number} must contain a JSON object")

            if field not in record:
                raise KeyError(f"{file_path}:{line_number} is missing field {field!r}")

            yield ensure_text_document(
                record[field],
                context=f"{file_path}:{line_number}:{field}",
            )


def iter_jsonl_fields(
    paths: Iterable[PathLike],
    field: str,
    *,
    encoding: str = "utf-8",
) -> Iterator[str]:
    """Yield one string field from many JSONL files in caller-supplied order."""

    for path in paths:
        yield from iter_jsonl_field(path, field=field, encoding=encoding)
