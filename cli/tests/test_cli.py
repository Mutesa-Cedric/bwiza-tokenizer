from __future__ import annotations

import json
from pathlib import Path

from bwiza_tokenizer_cli.main import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "tests" / "golden" / "model.v1.json"
CASES_PATH = REPO_ROOT / "tests" / "golden" / "cases.v1.jsonl"


def test_cli_parser_has_expected_program_name() -> None:
    assert build_parser().prog == "bwiza-tokenizer"


def test_normalize_command_prints_normalized_text(capsys) -> None:
    exit_code = main(["normalize", "  Muraho\t\tneza  "])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "▁Muraho▁neza\n"
    assert captured.err == ""


def test_encode_command_prints_token_ids_as_json(capsys) -> None:
    exit_code = main(["encode", "--model", str(MODEL_PATH), "Muraho neza"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == [7, 8, 9]
    assert captured.err == ""


def test_encode_command_prints_piece_surfaces_as_json(capsys) -> None:
    exit_code = main(["encode", "--model", str(MODEL_PATH), "--pieces", "Muraho neza"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == ["▁Mur", "aho", "▁neza"]
    assert captured.err == ""


def test_decode_command_prints_text(capsys) -> None:
    exit_code = main(["decode", "--model", str(MODEL_PATH), "7", "8", "9"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "Muraho neza\n"
    assert captured.err == ""


def test_decode_pieces_command_prints_text(capsys) -> None:
    exit_code = main(["decode-pieces", "--model", str(MODEL_PATH), "▁Mur", "aho", "▁neza"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "Muraho neza\n"
    assert captured.err == ""


def test_native_runtime_flag_reports_missing_extension(capsys) -> None:
    exit_code = main(["encode", "--model", str(MODEL_PATH), "--runtime", "native", "Muraho neza"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "bwiza_tokenizer_runtime" in captured.err

def test_parity_command_passes_against_committed_fixtures(capsys) -> None:
    exit_code = main(["parity", "--model", str(MODEL_PATH), "--cases", str(CASES_PATH)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload == {
        "model": str(MODEL_PATH),
        "cases": str(CASES_PATH),
        "runtime": "python",
        "total": 10,
        "failed": 0,
        "passed": 10,
        "ok": True,
        "mismatches": [],
    }
    assert captured.err == ""
