from bwiza_tokenizer_trainer import TrainerConfig, encode_text, train_from_iterator
from bwiza_tokenizer_trainer.model.validate import validate_model


def training_config() -> TrainerConfig:
    return TrainerConfig(
        vocab_size=8,
        seed_candidate_limit=16,
        max_piece_chars=4,
        min_candidate_freq=1,
        prune_fraction=0.5,
        max_iterations=4,
        min_score_delta=1e-6,
    )


def model_signature(model) -> tuple[tuple[int, str, float, str | None], ...]:
    return tuple(
        (entry.id, entry.piece, entry.score, entry.special)
        for entry in model.vocab
    )


def test_train_from_iterator_produces_valid_model() -> None:
    model = train_from_iterator(
        ["Muraho neza", "Muraho", "Neza"],
        training_config(),
    )

    validate_model(model)
    assert model.vocab_size == len(model.vocab)
    assert model.special_token_ids == {"unk": 0, "bos": 1, "eos": 2, "pad": 3}
    assert model.vocab[model.special_token_ids["unk"]].score == -10.0


def test_train_from_iterator_is_deterministic() -> None:
    docs = ["Muraho neza", "Muraho", "Neza"]
    config = training_config()

    first = train_from_iterator(docs, config)
    second = train_from_iterator(docs, config)

    assert model_signature(first) == model_signature(second)


def test_train_from_iterator_returns_encodable_model() -> None:
    model = train_from_iterator(
        ["Muraho neza", "Muraho", "Neza"],
        training_config(),
    )

    encoded = encode_text("Muraho neza", model)
    assert encoded
    assert all(isinstance(token_id, int) for token_id in encoded)


def test_train_from_iterator_handles_empty_corpus() -> None:
    model = train_from_iterator(["   ", "\t"], training_config())

    assert model.vocab_size == 4
    assert [entry.piece for entry in model.vocab] == ["<unk>", "<s>", "</s>", "<pad>"]
    assert [entry.score for entry in model.vocab] == [-10.0, 0.0, 0.0, 0.0]
