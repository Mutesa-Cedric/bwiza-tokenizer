from __future__ import annotations

from collections.abc import Iterator
from os import PathLike as OsPathLike
from typing import Protocol, TypeAlias

PathLike: TypeAlias = str | OsPathLike[str]


class DocumentSource(Protocol):
    """Protocol for any source that yields raw text documents."""

    def __iter__(self) -> Iterator[str]:
        ...


def ensure_text_document(value: object, *, context: str) -> str:
    """Reject non-string documents at the corpus boundary."""

    if isinstance(value, str):
        return value

    raise TypeError(f"{context} must be str, got {type(value).__name__}")
