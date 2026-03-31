from pathlib import Path

import pytest

from bwiza_tokenizer_trainer.corpus.jsonl import iter_jsonl_field, iter_jsonl_fields


def test_iter_jsonl_field_reads_one_text_field_per_line(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text(
        '{"text":"Muraho"}\n{"text":"amakuru"}\n',
        encoding="utf-8",
    )

    assert list(iter_jsonl_field(path, "text")) == ["Muraho", "amakuru"]


def test_iter_jsonl_field_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text('{"body":"Muraho"}\n', encoding="utf-8")

    with pytest.raises(KeyError, match="missing field 'text'"):
        list(iter_jsonl_field(path, "text"))


def test_iter_jsonl_field_rejects_non_string_values(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text('{"text":7}\n', encoding="utf-8")

    with pytest.raises(TypeError, match="text must be str, got int"):
        list(iter_jsonl_field(path, "text"))


def test_iter_jsonl_fields_respects_input_order(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text('{"text":"mbere"}\n', encoding="utf-8")
    second.write_text('{"text":"nyuma"}\n', encoding="utf-8")

    assert list(iter_jsonl_fields([second, first], "text")) == ["nyuma", "mbere"]
