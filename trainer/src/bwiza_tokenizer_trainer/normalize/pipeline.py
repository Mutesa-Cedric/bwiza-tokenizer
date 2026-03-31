from __future__ import annotations

from ..types import NormalizationConfig

from .rules import (
    collapse_ascii_spaces,
    map_whitespace_to_ascii_space,
    normalize_line_endings,
    normalize_unicode_text,
    prepend_leading_boundary_marker,
    replace_ascii_spaces_with_boundary_marker,
)


def normalize_text(
    text: str,
    config: NormalizationConfig | None = None,
) -> str:
    """Apply the full normalization-v1 pipeline."""

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    active_config = config or NormalizationConfig()

    normalized = normalize_line_endings(text)
    normalized = normalize_unicode_text(
        normalized,
        form=active_config.unicode_form,
    )
    normalized = map_whitespace_to_ascii_space(normalized)
    normalized = collapse_ascii_spaces(normalized)

    if active_config.trim:
        normalized = normalized.strip(" ")

    if not normalized:
        return ""

    normalized = replace_ascii_spaces_with_boundary_marker(
        normalized,
        boundary_marker=active_config.boundary_marker,
    )

    if active_config.prepend_leading_boundary:
        normalized = prepend_leading_boundary_marker(
            normalized,
            boundary_marker=active_config.boundary_marker,
        )

    return normalized
