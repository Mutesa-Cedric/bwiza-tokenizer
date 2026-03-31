from __future__ import annotations

from collections.abc import Iterable, Iterator

from .base import ensure_text_document


def iter_from_iterable(docs: Iterable[str]) -> Iterator[str]:
    """Yield documents from an iterable while enforcing string boundaries."""

    for index, value in enumerate(docs):
        yield ensure_text_document(value, context=f"document {index}")
