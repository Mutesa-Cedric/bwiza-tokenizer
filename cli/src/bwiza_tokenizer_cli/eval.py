from __future__ import annotations

from pathlib import Path
from typing import Literal

from bwiza_tokenizer_trainer.eval.reports import build_eval_report
from bwiza_tokenizer_trainer.export import write_eval_json
from bwiza_tokenizer_trainer.model.load import load_model

from .corpus import InputFormat, load_documents
from .sentencepiece_backend import detect_model_backend, evaluate_sentencepiece_model

EvalBackend = Literal["auto", "custom", "sentencepiece"]


def evaluate_tokenizer(
    *,
    model_path: str,
    paths: list[str],
    input_format: InputFormat,
    backend: EvalBackend = "auto",
    jsonl_field: str | None = None,
    sample_limit: int = 8,
    output_path: str | None = None,
) -> dict[str, object]:
    docs = load_documents(paths=paths, input_format=input_format, jsonl_field=jsonl_field)
    resolved_backend = detect_model_backend(model_path) if backend == "auto" else backend

    if resolved_backend == "sentencepiece":
        return evaluate_sentencepiece_model(
            model_path=model_path,
            docs=docs,
            sample_limit=sample_limit,
            output_path=output_path,
        )

    model = load_model(model_path)
    report = build_eval_report(docs, model, sample_limit=sample_limit)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_eval_json(report, destination)

    return report
