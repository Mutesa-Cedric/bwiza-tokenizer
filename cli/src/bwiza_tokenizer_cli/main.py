from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bwiza-tokenizer",
        description="CLI scaffold for bwiza-tokenizer.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="bwiza-tokenizer 0.0.0",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
