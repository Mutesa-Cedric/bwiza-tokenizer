from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from .base import PathLike


def iter_text_file(path: PathLike, *, encoding: str = "utf-8") -> Iterator[str]:
    """Yield one document per line from a UTF-8 text file."""

    file_path = Path(path)
    with file_path.open("r", encoding=encoding, newline=None) as handle:
        for line in handle:
            yield line.rstrip("\r\n")


def iter_text_files(paths: Iterable[PathLike], *, encoding: str = "utf-8") -> Iterator[str]:
    """Yield documents from many text files in the order given by the caller."""

    for path in paths:
        yield from iter_text_file(path, encoding=encoding)
