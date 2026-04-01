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


def test_train_command_writes_artifacts_for_text_input(tmp_path, capsys) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("Muraho neza\nMuraho\nNeza\n", encoding="utf-8")
    output_dir = tmp_path / "artifacts"

    exit_code = main([
        "train",
        "--output-dir",
        str(output_dir),
        "--name",
        "cli-text-demo",
        "--vocab-size",
        "8",
        str(corpus_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["name"] == "cli-text-demo"
    assert payload["documents"] == 3
    assert output_dir.joinpath("model.v1.json").exists()
    assert output_dir.joinpath("vocab.tsv").exists()
    assert output_dir.joinpath("eval.json").exists()

    model_payload = json.loads(output_dir.joinpath("model.v1.json").read_text(encoding="utf-8"))
    assert model_payload["name"] == "cli-text-demo"
    assert model_payload["model_type"] == "unigram"
    assert captured.err == ""


def test_train_command_writes_artifacts_for_jsonl_input(tmp_path, capsys) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"text":"Muraho neza"}\n{"text":"Muraho"}\n{"text":"Neza"}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"

    exit_code = main([
        "train",
        "--output-dir",
        str(output_dir),
        "--input-format",
        "jsonl",
        "--field",
        "text",
        "--name",
        "cli-jsonl-demo",
        "--vocab-size",
        "8",
        str(corpus_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["name"] == "cli-jsonl-demo"
    assert payload["documents"] == 3
    eval_payload = json.loads(output_dir.joinpath("eval.json").read_text(encoding="utf-8"))
    assert eval_payload["model_name"] == "cli-jsonl-demo"
    assert captured.err == ""


def test_eval_command_reports_metrics_for_text_input(tmp_path, capsys) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("Muraho neza\nMuraho\n", encoding="utf-8")

    exit_code = main([
        "eval",
        "--model",
        str(MODEL_PATH),
        str(corpus_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["model_name"] == "parity-demo"
    assert payload["vocab_size"] == 14
    assert payload["average_tokens_per_document"] > 0
    assert captured.err == ""


def test_eval_command_writes_report_for_jsonl_input(tmp_path, capsys) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"text":"Muraho neza"}\n{"text":"Muraho world"}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"

    exit_code = main([
        "eval",
        "--model",
        str(MODEL_PATH),
        "--input-format",
        "jsonl",
        "--field",
        "text",
        "--output",
        str(output_path),
        str(corpus_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == payload
    assert payload["model_name"] == "parity-demo"
    assert len(payload["sample_segmentations"]) > 0
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
