from __future__ import annotations

import re
import unicodedata

_ASCII_SPACE_RUN_RE = re.compile(r" +")


def normalize_line_endings(text: str) -> str:
    """Convert CRLF and CR line endings to LF."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_unicode_text(text: str, *, form: str = "NFC") -> str:
    """Apply the configured Unicode normalization form."""

    return unicodedata.normalize(form, text)


def map_whitespace_to_ascii_space(text: str) -> str:
    """Replace every Unicode whitespace code point with ASCII space."""

    return "".join(" " if char.isspace() else char for char in text)


def collapse_ascii_spaces(text: str) -> str:
    """Collapse runs of ASCII spaces into one ASCII space."""

    return _ASCII_SPACE_RUN_RE.sub(" ", text)


def replace_ascii_spaces_with_boundary_marker(
    text: str,
    *,
    boundary_marker: str,
) -> str:
    """Replace remaining ASCII spaces with the configured boundary marker."""

    return text.replace(" ", boundary_marker)


def prepend_leading_boundary_marker(
    text: str,
    *,
    boundary_marker: str,
) -> str:
    """Ensure non-empty normalized text starts with the boundary marker."""

    if not text or text.startswith(boundary_marker):
        return text

    return f"{boundary_marker}{text}"
