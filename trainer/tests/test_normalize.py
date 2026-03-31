import unicodedata

import pytest

from bwiza_tokenizer_trainer import normalize_text
from bwiza_tokenizer_trainer.normalize.rules import (
    collapse_ascii_spaces,
    map_whitespace_to_ascii_space,
    normalize_line_endings,
    normalize_unicode_text,
)


def test_normalize_text_basic_word_boundaries() -> None:
    assert normalize_text("Muraho neza") == "▁Muraho▁neza"


def test_normalize_text_collapses_tabs_newlines_and_spaces() -> None:
    assert normalize_text("  Muraho\t\tneza  ") == "▁Muraho▁neza"
    assert normalize_text("Muraho\nneza") == "▁Muraho▁neza"
    assert normalize_text("Muraho\r\nneza") == "▁Muraho▁neza"


def test_normalize_text_empty_input_stays_empty() -> None:
    assert normalize_text("") == ""
    assert normalize_text(" \t \n ") == ""


def test_normalize_text_matches_nfc_for_decomposed_text() -> None:
    composed = "café"
    decomposed = unicodedata.normalize("NFD", composed)

    assert normalize_text(composed) == normalize_text(decomposed) == "▁café"


def test_normalize_text_preserves_punctuation_adjacency() -> None:
    assert normalize_text("Muraho, neza!") == "▁Muraho,▁neza!"


def test_normalize_text_is_deterministic() -> None:
    text = "  Muraho\tneza \n"

    assert normalize_text(text) == normalize_text(text)


def test_normalize_text_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="text must be str, got int"):
        normalize_text(3)  # type: ignore[arg-type]


def test_normalize_rules_are_composable() -> None:
    text = "Muraho\r\n\tneza"

    assert normalize_line_endings(text) == "Muraho\n\tneza"
    assert normalize_unicode_text("cafe\u0301") == "café"
    assert map_whitespace_to_ascii_space("A\tB\nC") == "A B C"
    assert collapse_ascii_spaces("A   B") == "A B"
