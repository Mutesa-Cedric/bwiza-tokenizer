from __future__ import annotations

import json

import pytest

from bwiza_tokenizer_cli.main import main

pytest.importorskip("sentencepiece")


def test_train_command_supports_sentencepiece_backend(tmp_path, capsys) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "Muraho neza\nMuraho bwiza\nNeza cyane\nIsi yose\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"

    exit_code = main([
        "train",
        "--backend",
        "sentencepiece",
        "--output-dir",
        str(output_dir),
        "--name",
        "Cli SentencePiece Demo",
        "--vocab-size",
        "32",
        str(corpus_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["name"] == "cli-sentencepiece-demo"
    assert payload["documents"] == 4
    assert payload["vocab_size"] > 0
    assert output_dir.joinpath("cli-sentencepiece-demo.model").exists()
    assert output_dir.joinpath("cli-sentencepiece-demo.vocab").exists()
    assert output_dir.joinpath("eval.json").exists()
    assert captured.err == ""


def test_eval_command_supports_sentencepiece_backend_auto_detection(tmp_path, capsys) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "Muraho neza\nMuraho bwiza\nNeza cyane\nIsi yose\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"
    report_path = tmp_path / "report.json"

    train_exit = main([
        "train",
        "--backend",
        "sentencepiece",
        "--output-dir",
        str(output_dir),
        "--name",
        "Cli SentencePiece Demo",
        "--vocab-size",
        "32",
        str(corpus_path),
    ])
    assert train_exit == 0
    capsys.readouterr()

    model_path = output_dir / "cli-sentencepiece-demo.model"
    exit_code = main([
        "eval",
        "--model",
        str(model_path),
        "--output",
        str(report_path),
        str(corpus_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written == payload
    assert payload["model_name"] == "cli-sentencepiece-demo"
    assert payload["vocab_size"] > 0
    assert payload["average_tokens_per_document"] > 0
    assert len(payload["sample_segmentations"]) > 0
    assert captured.err == ""
