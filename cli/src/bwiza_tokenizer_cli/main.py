from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bwiza_tokenizer_cli.runtime import load_backend
from bwiza_tokenizer_cli.train import train_tokenizer
from bwiza_tokenizer_trainer.normalize.pipeline import normalize_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bwiza-tokenizer",
        description="Command-line interface for bwiza-tokenizer.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="bwiza-tokenizer 0.0.0",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize raw text according to the tokenizer contract.",
    )
    normalize_parser.add_argument("text", help="Input text to normalize.")
    normalize_parser.set_defaults(handler=_run_normalize)

    train_parser = subparsers.add_parser(
        "train",
        help="Train a tokenizer model from text or JSONL corpora.",
    )
    train_parser.add_argument("--output-dir", required=True, help="Directory to write model artifacts into.")
    train_parser.add_argument(
        "--input-format",
        choices=("text", "jsonl"),
        default="text",
        help="Corpus file format.",
    )
    train_parser.add_argument("--field", help="JSONL field to read when --input-format=jsonl.")
    train_parser.add_argument("--name", default="bwiza-unigram-v1", help="Model name to write into the artifact.")
    train_parser.add_argument("--vocab-size", type=int, default=16000, help="Target vocabulary size.")
    train_parser.add_argument("--sample-limit", type=int, default=8, help="How many sample segmentations to include in eval output.")
    train_parser.add_argument("paths", nargs="+", help="One or more corpus files.")
    train_parser.set_defaults(handler=_run_train)

    encode_parser = subparsers.add_parser(
        "encode",
        help="Encode raw text into tokenizer ids or pieces.",
    )
    _add_model_arguments(encode_parser)
    encode_parser.add_argument(
        "--pieces",
        action="store_true",
        help="Emit piece surfaces instead of ids.",
    )
    encode_parser.add_argument("text", help="Input text to encode.")
    encode_parser.set_defaults(handler=_run_encode)

    decode_parser = subparsers.add_parser(
        "decode",
        help="Decode tokenizer ids back into readable text.",
    )
    _add_model_arguments(decode_parser)
    decode_parser.add_argument("ids", nargs="+", type=int, help="Token ids to decode.")
    decode_parser.set_defaults(handler=_run_decode)

    decode_pieces_parser = subparsers.add_parser(
        "decode-pieces",
        help="Decode tokenizer piece surfaces back into readable text.",
    )
    _add_model_arguments(decode_pieces_parser)
    decode_pieces_parser.add_argument("pieces", nargs="+", help="Piece surfaces to decode.")
    decode_pieces_parser.set_defaults(handler=_run_decode_pieces)

    parity_parser = subparsers.add_parser(
        "parity",
        help="Check model behavior against committed parity fixtures.",
    )
    _add_model_arguments(parity_parser)
    parity_parser.add_argument(
        "--cases",
        required=True,
        help="Path to cases.v1.jsonl parity fixtures.",
    )
    parity_parser.set_defaults(handler=_run_parity)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError, ModuleNotFoundError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Path to model.v1.json")
    parser.add_argument(
        "--runtime",
        choices=("auto", "python", "native"),
        default="auto",
        help="Tokenizer runtime to use for model-backed commands.",
    )


def _run_normalize(args: argparse.Namespace) -> int:
    print(normalize_text(args.text))
    return 0


def _run_train(args: argparse.Namespace) -> int:
    summary = train_tokenizer(
        paths=list(args.paths),
        input_format=args.input_format,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        model_name=args.name,
        jsonl_field=args.field,
        sample_limit=args.sample_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _run_encode(args: argparse.Namespace) -> int:
    backend = load_backend(args.model, runtime=args.runtime)
    payload: list[int] | list[str]
    if args.pieces:
        payload = backend.encode_pieces(args.text)
    else:
        payload = backend.encode_ids(args.text)

    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _run_decode(args: argparse.Namespace) -> int:
    backend = load_backend(args.model, runtime=args.runtime)
    print(backend.decode_ids(args.ids))
    return 0


def _run_decode_pieces(args: argparse.Namespace) -> int:
    backend = load_backend(args.model, runtime=args.runtime)
    print(backend.decode_pieces(args.pieces))
    return 0


def _run_parity(args: argparse.Namespace) -> int:
    backend = load_backend(args.model, runtime=args.runtime)
    cases_path = Path(args.cases)

    total = 0
    mismatches: list[dict[str, Any]] = []
    with cases_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            total += 1
            case = json.loads(line)
            input_text = case["input"]
            normalized = backend.normalize(input_text)
            pieces = backend.encode_pieces(input_text)
            ids = backend.encode_ids(input_text)
            decoded_text = backend.decode_ids(ids)

            failed_fields: list[str] = []
            if normalized != case["normalized"]:
                failed_fields.append("normalized")
            if pieces != case["pieces"]:
                failed_fields.append("pieces")
            if ids != case["ids"]:
                failed_fields.append("ids")
            if decoded_text != case["decoded"]:
                failed_fields.append("decoded")

            if failed_fields:
                mismatches.append(
                    {
                        "case_id": case["case_id"],
                        "line": line_number,
                        "failed_fields": failed_fields,
                    }
                )

    summary = {
        "model": str(Path(args.model)),
        "cases": str(cases_path),
        "runtime": backend.runtime_name,
        "total": total,
        "failed": len(mismatches),
        "passed": total - len(mismatches),
        "ok": not mismatches,
        "mismatches": mismatches,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
