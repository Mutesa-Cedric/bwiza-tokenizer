from __future__ import annotations

from pathlib import Path
from typing import Literal

from bwiza_tokenizer_trainer import TrainerConfig, train_from_iterator
from bwiza_tokenizer_trainer.eval.reports import build_eval_report
from bwiza_tokenizer_trainer.export import write_eval_json, write_model_json, write_vocab_tsv

from .corpus import InputFormat, load_documents
from .sentencepiece_backend import train_sentencepiece_model

TrainBackend = Literal["custom", "sentencepiece"]


def train_tokenizer(
    *,
    paths: list[str],
    input_format: InputFormat,
    output_dir: str,
    vocab_size: int,
    model_name: str,
    backend: TrainBackend = "custom",
    jsonl_field: str | None = None,
    sample_limit: int = 8,
) -> dict[str, object]:
    docs = load_documents(paths=paths, input_format=input_format, jsonl_field=jsonl_field)
    if backend == "sentencepiece":
        return train_sentencepiece_model(
            docs=docs,
            output_dir=output_dir,
            vocab_size=vocab_size,
            model_name=model_name,
            sample_limit=sample_limit,
        )

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
