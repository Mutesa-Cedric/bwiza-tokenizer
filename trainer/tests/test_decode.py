from bwiza_tokenizer_trainer.byte_fallback import byte_fallback_piece
from bwiza_tokenizer_trainer import decode_ids
from bwiza_tokenizer_trainer.model.load import load_model_dict
from bwiza_tokenizer_trainer.reference_runtime.decode import decode_pieces


def tiny_model():
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


def byte_fallback_model():
    vocab = [
        {"id": 0, "piece": "<unk>", "score": 0.0, "special": "unk"},
        {"id": 1, "piece": "<s>", "score": 0.0, "special": "bos"},
        {"id": 2, "piece": "</s>", "score": 0.0, "special": "eos"},
        {"id": 3, "piece": "<pad>", "score": 0.0, "special": "pad"},
        {"id": 4, "piece": "▁Hi", "score": -1.0, "special": None},
    ]
    vocab.extend(
        {
            "id": 5 + byte_value,
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


def test_decode_pieces_trims_one_leading_boundary_space() -> None:
    model = tiny_model()

    assert decode_pieces(["▁Mu", "raho", "▁neza"], model=model) == "Muraho neza"


def test_decode_ids_uses_model_piece_lookup() -> None:
    model = tiny_model()

    assert decode_ids([4, 5, 6], model) == "Muraho neza"


def test_decode_ids_rejects_unknown_ids() -> None:
    model = tiny_model()

    try:
        decode_ids([999], model)
    except ValueError as exc:
        assert str(exc) == "unknown token id 999"
    else:
        raise AssertionError("decode_ids should reject unknown token ids")


def test_decode_pieces_round_trips_byte_fallback_sequences() -> None:
    model = byte_fallback_model()

    assert decode_pieces(
        [
            "▁Hi",
            byte_fallback_piece(0xF0),
            byte_fallback_piece(0x9F),
            byte_fallback_piece(0x98),
            byte_fallback_piece(0x85),
        ],
        model=model,
    ) == "Hi😅"
