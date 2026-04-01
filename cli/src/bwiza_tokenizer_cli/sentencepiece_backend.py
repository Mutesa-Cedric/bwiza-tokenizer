from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, cast

from bwiza_tokenizer_trainer.export import write_eval_json
from bwiza_tokenizer_trainer.normalize.pipeline import normalize_text

def detect_model_backend(model_path: str) -> Literal["custom", "sentencepiece"]:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".json":
        return "custom"
    if suffix == ".model":
        return "sentencepiece"
    raise ValueError("could not infer backend from model path; use --backend with 'custom' or 'sentencepiece'")


def train_sentencepiece_model(
    *,
    docs: list[str],
    output_dir: str,
    vocab_size: int,
    model_name: str,
    sample_limit: int = 8,
) -> dict[str, object]:
    spm = _load_sentencepiece_module()
    normalized_docs = _normalize_docs(docs)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prefix_name = _slugify_name(model_name)
    model_prefix = output_path / prefix_name
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(normalized_docs),
        model_prefix=str(model_prefix),
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=1.0,
        normalization_rule_name="identity",
        shuffle_input_sentence=False,
        input_sentence_size=0,
        hard_vocab_limit=False,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
    )

    model_path = model_prefix.with_suffix(".model")
    vocab_path = model_prefix.with_suffix(".vocab")
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    eval_report = build_sentencepiece_eval_report(
        docs,
        processor,
        model_name=prefix_name,
        sample_limit=sample_limit,
    )
    eval_path = write_eval_json(eval_report, output_path / "eval.json")

    return {
        "name": prefix_name,
        "documents": len(docs),
        "vocab_size": processor.get_piece_size(),
        "model": str(model_path),
        "vocab": str(vocab_path),
        "eval": str(eval_path),
    }


def evaluate_sentencepiece_model(
    *,
    model_path: str,
    docs: list[str],
    sample_limit: int = 8,
    output_path: str | None = None,
) -> dict[str, object]:
    spm = _load_sentencepiece_module()
    processor = spm.SentencePieceProcessor(model_file=model_path)
    report = build_sentencepiece_eval_report(
        docs,
        processor,
        model_name=Path(model_path).stem,
        sample_limit=sample_limit,
    )

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_eval_json(report, destination)

    return report


def build_sentencepiece_eval_report(
    docs: list[str],
    processor: Any,
    *,
    model_name: str,
    sample_limit: int = 8,
) -> dict[str, object]:
    total_chars = 0
    total_tokens = 0
    unknown_tokens = 0
    document_count = 0
    used_ids: set[int] = set()
    samples: list[dict[str, Any]] = []
    unknown_id = cast(int, processor.unk_id())
    vocab_size = cast(int, processor.get_piece_size())

    for doc in docs:
        normalized = normalize_text(doc)
        if not normalized:
            continue

        ids = cast(list[int], list(processor.encode(normalized, out_type=int)))
        pieces = cast(list[str], list(processor.encode(normalized, out_type=str)))
        decoded = cast(str, processor.decode(ids))

        document_count += 1
        total_chars += len(normalized)
        total_tokens += len(ids)
        unknown_tokens += sum(1 for token_id in ids if token_id == unknown_id)
        used_ids.update(ids)

        if len(samples) < sample_limit:
            samples.append(
                {
                    "input": doc,
                    "normalized": normalized,
                    "pieces": pieces,
                    "ids": ids,
                    "decoded": decoded,
                }
            )

    if total_tokens == 0 or vocab_size == 0:
        average_chars_per_token = 0.0
        average_tokens_per_document = 0.0
        unknown_rate = 0.0
        vocab_utilization = 0.0
    else:
        average_chars_per_token = total_chars / total_tokens
        average_tokens_per_document = total_tokens / document_count
        unknown_rate = unknown_tokens / total_tokens
        vocab_utilization = len(used_ids) / vocab_size

    return {
        "model_name": model_name,
        "vocab_size": vocab_size,
        "average_chars_per_token": average_chars_per_token,
        "average_tokens_per_document": average_tokens_per_document,
        "unknown_rate": unknown_rate,
        "vocab_utilization": vocab_utilization,
        "sample_segmentations": samples,
    }


def _normalize_docs(docs: list[str]) -> list[str]:
    normalized_docs: list[str] = []
    for doc in docs:
        normalized = normalize_text(doc)
        if normalized:
            normalized_docs.append(normalized)
    return normalized_docs


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    slug = slug.strip("-")
    return slug or "sentencepiece-unigram"


def _load_sentencepiece_module() -> Any:
    try:
        import sentencepiece as spm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sentencepiece backend requires the 'sentencepiece' package. Install it with: python3 -m pip install sentencepiece"
        ) from exc

    return spm
