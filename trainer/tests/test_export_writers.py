import json

from bwiza_tokenizer_trainer import TrainerConfig, train_from_iterator
from bwiza_tokenizer_trainer.export import write_eval_json, write_model_json, write_vocab_tsv
from bwiza_tokenizer_trainer.model.load import load_model


def trained_model():
    return train_from_iterator(
        ["Muraho neza", "Muraho", "Neza"],
        TrainerConfig(
            vocab_size=8,
            seed_candidate_limit=16,
            max_piece_chars=4,
            min_candidate_freq=1,
            prune_fraction=0.5,
            max_iterations=4,
            min_score_delta=1e-6,
        ),
    )


def test_write_model_json_round_trips(tmp_path) -> None:
    path = tmp_path / "tokenizer.model.json"
    model = trained_model()

    write_model_json(model, path)
    reloaded = load_model(path)

    assert reloaded.vocab_size == model.vocab_size
    assert [entry.piece for entry in reloaded.vocab] == [entry.piece for entry in model.vocab]
    assert reloaded.special_token_ids == model.special_token_ids


def test_write_model_json_is_stable(tmp_path) -> None:
    model = trained_model()
    first = tmp_path / "first.model.json"
    second = tmp_path / "second.model.json"

    write_model_json(model, first)
    write_model_json(model, second)

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_write_vocab_tsv_has_expected_header(tmp_path) -> None:
    path = tmp_path / "tokenizer.vocab.tsv"

    write_vocab_tsv(trained_model(), path)
    lines = path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "id\tpiece\tscore\tspecial"
    assert lines[1].startswith("0\t<unk>\t")


def test_write_eval_json_is_stable(tmp_path) -> None:
    report = {
        "unknown_rate": 0.0,
        "average_chars_per_token": 2.5,
        "sample_segmentations": [{"input": "Muraho neza", "pieces": ["▁Muraho", "▁neza"]}],
    }
    first = tmp_path / "first.eval.json"
    second = tmp_path / "second.eval.json"

    write_eval_json(report, first)
    write_eval_json(report, second)

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    assert json.loads(first.read_text(encoding="utf-8"))["unknown_rate"] == 0.0
