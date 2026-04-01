from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bwiza_tokenizer_trainer.model.load import load_model
from bwiza_tokenizer_trainer.normalize.pipeline import normalize_text
from bwiza_tokenizer_trainer.reference_runtime.decode import decode_ids, decode_pieces
from bwiza_tokenizer_trainer.reference_runtime.encode import encode_to_ids, encode_to_pieces


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

    encode_parser = subparsers.add_parser(
        "encode",
        help="Encode raw text into tokenizer ids or pieces.",
    )
    encode_parser.add_argument("--model", required=True, help="Path to model.v1.json")
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
    decode_parser.add_argument("--model", required=True, help="Path to model.v1.json")
    decode_parser.add_argument("ids", nargs="+", type=int, help="Token ids to decode.")
    decode_parser.set_defaults(handler=_run_decode)

    decode_pieces_parser = subparsers.add_parser(
        "decode-pieces",
        help="Decode tokenizer piece surfaces back into readable text.",
    )
    decode_pieces_parser.add_argument("--model", required=True, help="Path to model.v1.json")
    decode_pieces_parser.add_argument("pieces", nargs="+", help="Piece surfaces to decode.")
    decode_pieces_parser.set_defaults(handler=_run_decode_pieces)

    parity_parser = subparsers.add_parser(
        "parity",
        help="Check model behavior against committed parity fixtures.",
    )
    parity_parser.add_argument("--model", required=True, help="Path to model.v1.json")
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
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _run_normalize(args: argparse.Namespace) -> int:
    print(normalize_text(args.text))
    return 0


def _run_encode(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    payload: list[int] | list[str]
    if args.pieces:
        payload = encode_to_pieces(args.text, model)
    else:
        payload = encode_to_ids(args.text, model)

    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _run_decode(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    print(decode_ids(args.ids, model))
    return 0


def _run_decode_pieces(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    print(decode_pieces(args.pieces, model))
    return 0


def _run_parity(args: argparse.Namespace) -> int:
    model = load_model(args.model)
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
            normalized = normalize_text(input_text, config=model.normalization)
            pieces = encode_to_pieces(input_text, model)
            ids = encode_to_ids(input_text, model)
            decoded_text = decode_ids(ids, model)

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
