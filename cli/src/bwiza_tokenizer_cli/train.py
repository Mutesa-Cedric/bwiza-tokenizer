from __future__ import annotations

from pathlib import Path
from typing import Literal

from bwiza_tokenizer_trainer import TrainerConfig, train_from_iterator
from bwiza_tokenizer_trainer.corpus.jsonl import iter_jsonl_fields
from bwiza_tokenizer_trainer.corpus.text_files import iter_text_files
from bwiza_tokenizer_trainer.eval.reports import build_eval_report
from bwiza_tokenizer_trainer.export import write_eval_json, write_model_json, write_vocab_tsv

InputFormat = Literal["text", "jsonl"]


def train_tokenizer(
    *,
    paths: list[str],
    input_format: InputFormat,
    output_dir: str,
    vocab_size: int,
    model_name: str,
    jsonl_field: str | None = None,
    sample_limit: int = 8,
) -> dict[str, object]:
    docs = _load_documents(paths=paths, input_format=input_format, jsonl_field=jsonl_field)
    config = TrainerConfig(vocab_size=vocab_size)
    model = train_from_iterator(docs, config)
    model.name = model_name

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = write_model_json(model, output_path / "model.v1.json")
    vocab_path = write_vocab_tsv(model, output_path / "vocab.tsv")
    eval_report = build_eval_report(docs, model, sample_limit=sample_limit)
    eval_path = write_eval_json(eval_report, output_path / "eval.json")

    return {
        "name": model.name,
        "documents": len(docs),
        "vocab_size": model.vocab_size,
        "model": str(model_path),
        "vocab": str(vocab_path),
        "eval": str(eval_path),
    }


def _load_documents(
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
