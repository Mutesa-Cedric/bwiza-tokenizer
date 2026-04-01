from __future__ import annotations

from typing import Literal

from bwiza_tokenizer_trainer.corpus.jsonl import iter_jsonl_fields
from bwiza_tokenizer_trainer.corpus.text_files import iter_text_files

InputFormat = Literal["text", "jsonl"]


def load_documents(
    *,
    paths: list[str],
    input_format: InputFormat,
    jsonl_field: str | None,
) -> list[str]:
    if input_format == "text":
        return list(iter_text_files(paths))

    if jsonl_field is None:
        raise ValueError("--field is required when --input-format=jsonl")

    return list(iter_jsonl_fields(paths, field=jsonl_field))
