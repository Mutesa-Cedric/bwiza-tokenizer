from bwiza_tokenizer_trainer import encode_text
from bwiza_tokenizer_trainer.model.load import load_model_dict
from bwiza_tokenizer_trainer.reference_runtime.encode import encode_to_ids, encode_to_pieces


def base_model():
    return load_model_dict(
        {
            "version": "model-v1",
            "name": "demo-unigram",
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
                {"id": 0, "piece": "<unk>", "score": 0.0, "special": "unk"},
                {"id": 1, "piece": "<s>", "score": 0.0, "special": "bos"},
                {"id": 2, "piece": "</s>", "score": 0.0, "special": "eos"},
                {"id": 3, "piece": "<pad>", "score": 0.0, "special": "pad"},
                {"id": 4, "piece": "▁Mu", "score": -1.9, "special": None},
                {"id": 5, "piece": "raho", "score": -2.3, "special": None},
                {"id": 6, "piece": "▁neza", "score": -2.1, "special": None},
                {"id": 7, "piece": ".", "score": -3.0, "special": None},
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


def test_encode_to_pieces_uses_normalized_input() -> None:
    model = base_model()

    assert encode_to_pieces("  Muraho\tneza  ", model) == ["▁Mu", "raho", "▁neza"]


def test_encode_text_returns_ids() -> None:
    model = base_model()

    assert encode_text("Muraho neza", model) == [4, 5, 6]
    assert encode_to_ids("Muraho neza", model) == [4, 5, 6]


def test_encode_prefers_longer_earliest_piece_when_scores_tie() -> None:
    model = tie_break_model()

    assert encode_to_pieces("Muraho", model) == ["▁Mur", "aho"]
    assert encode_to_ids("Muraho", model) == [6, 7]


def test_encode_uses_unknown_fallback_one_scalar_at_a_time() -> None:
    model = unknown_model()

    assert encode_to_pieces("abx", model) == ["▁a", "b", "<unk>"]
    assert encode_to_ids("abx", model) == [4, 5, 0]


def test_encode_empty_string_returns_empty_sequence() -> None:
    model = base_model()

    assert encode_to_pieces("", model) == []
    assert encode_to_ids("", model) == []
