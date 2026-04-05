from bwiza_tokenizer_trainer.byte_fallback import byte_fallback_piece
from bwiza_tokenizer_trainer.model.load import load_model_dict
from bwiza_tokenizer_trainer.train import segment_normalized
from bwiza_tokenizer_trainer.train.viterbi import iter_segment_ids


def tie_break_model():
    return load_model_dict(
        {
            "version": "model-v1",
            "name": "tie-break-demo",
            "model_type": "unigram",
            "vocab_size": 8,
            "normalization": {
                "unicode_form": "NFC",
                "whitespace_policy": "all_whitespace_to_space_then_collapse",
                "trim": True,
                "boundary_marker": "▁",
                "prepend_leading_boundary": True,
            },
            "special_token_ids": {
                "unk": 0,
                "bos": 1,
                "eos": 2,
                "pad": 3,
            },
            "vocab": [
                {"id": 0, "piece": "<unk>", "score": -10.0, "special": "unk"},
                {"id": 1, "piece": "<s>", "score": 0.0, "special": "bos"},
                {"id": 2, "piece": "</s>", "score": 0.0, "special": "eos"},
                {"id": 3, "piece": "<pad>", "score": 0.0, "special": "pad"},
                {"id": 4, "piece": "▁Mu", "score": -1.0, "special": None},
                {"id": 5, "piece": "raho", "score": -3.0, "special": None},
                {"id": 6, "piece": "▁Mur", "score": -2.0, "special": None},
                {"id": 7, "piece": "aho", "score": -2.0, "special": None},
            ],
            "trainer": {
                "target_vocab_size": 8,
                "seed_candidate_limit": 64,
                "max_piece_chars": 12,
                "min_candidate_freq": 1,
                "prune_fraction": 0.15,
                "max_iterations": 6,
            },
        }
    )


def unknown_model():
    return load_model_dict(
        {
            "version": "model-v1",
            "name": "unknown-demo",
            "model_type": "unigram",
            "vocab_size": 6,
            "normalization": {
                "unicode_form": "NFC",
                "whitespace_policy": "all_whitespace_to_space_then_collapse",
                "trim": True,
                "boundary_marker": "▁",
                "prepend_leading_boundary": True,
            },
            "special_token_ids": {
                "unk": 0,
                "bos": 1,
                "eos": 2,
                "pad": 3,
            },
            "vocab": [
                {"id": 0, "piece": "<unk>", "score": -10.0, "special": "unk"},
                {"id": 1, "piece": "<s>", "score": 0.0, "special": "bos"},
                {"id": 2, "piece": "</s>", "score": 0.0, "special": "eos"},
                {"id": 3, "piece": "<pad>", "score": 0.0, "special": "pad"},
                {"id": 4, "piece": "▁a", "score": -1.0, "special": None},
                {"id": 5, "piece": "b", "score": -1.0, "special": None},
            ],
            "trainer": {
                "target_vocab_size": 6,
                "seed_candidate_limit": 64,
                "max_piece_chars": 12,
                "min_candidate_freq": 1,
                "prune_fraction": 0.15,
                "max_iterations": 6,
            },
        }
    )


def byte_fallback_model():
    vocab = [
        {"id": 0, "piece": "<unk>", "score": -10.0, "special": "unk"},
        {"id": 1, "piece": "<s>", "score": 0.0, "special": "bos"},
        {"id": 2, "piece": "</s>", "score": 0.0, "special": "eos"},
        {"id": 3, "piece": "<pad>", "score": 0.0, "special": "pad"},
        {"id": 4, "piece": "▁a", "score": -1.0, "special": None},
        {"id": 5, "piece": "b", "score": -1.0, "special": None},
    ]
    vocab.extend(
        {
            "id": 6 + byte_value,
            "piece": byte_fallback_piece(byte_value),
            "score": -100.0,
            "special": None,
        }
        for byte_value in range(256)
    )

    return load_model_dict(
        {
            "version": "model-v1",
            "name": "byte-fallback-demo",
            "model_type": "unigram",
            "vocab_size": len(vocab),
            "normalization": {
                "unicode_form": "NFC",
                "whitespace_policy": "all_whitespace_to_space_then_collapse",
                "trim": True,
                "boundary_marker": "▁",
                "prepend_leading_boundary": True,
            },
            "special_token_ids": {
                "unk": 0,
                "bos": 1,
                "eos": 2,
                "pad": 3,
            },
            "vocab": vocab,
            "trainer": {
                "target_vocab_size": len(vocab),
                "seed_candidate_limit": 64,
                "max_piece_chars": 12,
                "min_candidate_freq": 1,
                "prune_fraction": 0.15,
                "max_iterations": 6,
                "byte_fallback": True,
            },
        }
    )


def test_segment_normalized_prefers_longer_earliest_piece_on_score_tie() -> None:
    result = segment_normalized("▁Muraho", tie_break_model())

    assert result.pieces == ("▁Mur", "aho")
    assert result.ids == (6, 7)
    assert result.score == -4.0


def test_segment_normalized_uses_unknown_fallback_for_one_scalar() -> None:
    result = segment_normalized("▁abx", unknown_model())

    assert result.pieces == ("▁a", "b", "<unk>")
    assert result.ids == (4, 5, 0)
    assert result.score == -12.0


def test_segment_normalized_is_deterministic() -> None:
    model = tie_break_model()

    first = segment_normalized("▁Muraho", model)
    second = segment_normalized("▁Muraho", model)

    assert first == second


def test_iter_segment_ids_matches_segment_normalized_ids() -> None:
    model = unknown_model()

    assert list(iter_segment_ids("▁abx", model)) == list(segment_normalized("▁abx", model).ids)


def test_segment_normalized_rejects_non_string_input() -> None:
    try:
        segment_normalized(123, tie_break_model())  # type: ignore[arg-type]
    except TypeError as exc:
        assert str(exc) == "segment_normalized expects a normalized string"
    else:
        raise AssertionError("segment_normalized should reject non-string input")


def test_segment_normalized_handles_long_unknown_runs() -> None:
    result = segment_normalized("x" * 5000, unknown_model())

    assert len(result.ids) == 5000
    assert result.ids[:4] == (0, 0, 0, 0)
    assert result.pieces[:4] == ("<unk>", "<unk>", "<unk>", "<unk>")


def test_segment_normalized_uses_byte_fallback_for_unseen_characters() -> None:
    result = segment_normalized("▁ab😅", byte_fallback_model())

    assert result.ids[:2] == (4, 5)
    assert result.pieces[:2] == ("▁a", "b")
    assert result.pieces[2:] == (
        byte_fallback_piece(0xF0),
        byte_fallback_piece(0x9F),
        byte_fallback_piece(0x98),
        byte_fallback_piece(0x85),
    )
