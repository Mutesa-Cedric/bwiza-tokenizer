from pathlib import Path

from bwiza_tokenizer_trainer.corpus.text_files import iter_text_file, iter_text_files


def test_iter_text_file_yields_one_document_per_line(tmp_path: Path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("Muraho\namakuru\n\n", encoding="utf-8")

    assert list(iter_text_file(path)) == ["Muraho", "amakuru", ""]


def test_iter_text_files_respects_input_order(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("mbere\n", encoding="utf-8")
    second.write_text("nyuma\n", encoding="utf-8")

    assert list(iter_text_files([second, first])) == ["nyuma", "mbere"]
