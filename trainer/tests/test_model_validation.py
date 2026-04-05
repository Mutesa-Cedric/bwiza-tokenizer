import json
from pathlib import Path

import pytest

from bwiza_tokenizer_trainer.byte_fallback import byte_fallback_piece
from bwiza_tokenizer_trainer.model.load import load_model, load_model_dict
from bwiza_tokenizer_trainer.model.validate import ModelValidationError


def valid_model_dict() -> dict[str, object]:
    return {
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


def test_load_model_dict_accepts_valid_model() -> None:
    model = load_model_dict(valid_model_dict())

    assert model.version == "model-v1"
    assert model.model_type == "unigram"
    assert model.vocab_size == 8
    assert model.vocab[4].piece == "▁Mu"


def test_load_model_reads_json_file(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.model.json"
    path.write_text(json.dumps(valid_model_dict()), encoding="utf-8")

    model = load_model(path)

    assert model.name == "demo-unigram"
    assert model.special_token_ids["unk"] == 0


def test_load_model_rejects_wrong_version() -> None:
    model_data = valid_model_dict()
    model_data["version"] = "model-v2"

    with pytest.raises(ModelValidationError, match="model version must be 'model-v1'"):
        load_model_dict(model_data)


def test_load_model_rejects_wrong_model_type() -> None:
    model_data = valid_model_dict()
    model_data["model_type"] = "bpe"

    with pytest.raises(ModelValidationError, match="model_type must be 'unigram'"):
        load_model_dict(model_data)


def test_load_model_rejects_vocab_size_mismatch() -> None:
    model_data = valid_model_dict()
    model_data["vocab_size"] = 7

    with pytest.raises(ModelValidationError, match="vocab_size must equal len\\(vocab\\)"):
        load_model_dict(model_data)


def test_load_model_rejects_non_contiguous_ids() -> None:
    model_data = valid_model_dict()
    vocab = model_data["vocab"]
    assert isinstance(vocab, list)
    vocab[7]["id"] = 9

    with pytest.raises(ModelValidationError, match="vocab ids must be unique and contiguous"):
        load_model_dict(model_data)


def test_load_model_rejects_duplicate_pieces() -> None:
    model_data = valid_model_dict()
    vocab = model_data["vocab"]
    assert isinstance(vocab, list)
    vocab[7]["piece"] = "▁Mu"

    with pytest.raises(ModelValidationError, match="vocab pieces must be unique"):
        load_model_dict(model_data)


def test_load_model_rejects_wrong_special_token_mapping() -> None:
    model_data = valid_model_dict()
    special_ids = model_data["special_token_ids"]
    assert isinstance(special_ids, dict)
    special_ids["unk"] = 9

    with pytest.raises(ModelValidationError, match="special_token_ids must match the fixed v1 mapping"):
        load_model_dict(model_data)


def test_load_model_rejects_missing_special_token_id() -> None:
    model_data = valid_model_dict()
    special_ids = model_data["special_token_ids"]
    assert isinstance(special_ids, dict)
    del special_ids["pad"]

    with pytest.raises(ModelValidationError, match="special_token_ids is missing field 'pad'"):
        load_model_dict(model_data)


def test_load_model_rejects_wrong_normalization_contract() -> None:
    model_data = valid_model_dict()
    normalization = model_data["normalization"]
    assert isinstance(normalization, dict)
    normalization["boundary_marker"] = "_"

    with pytest.raises(ModelValidationError, match="normalization config does not match normalization-v1"):
        load_model_dict(model_data)


def test_load_model_rejects_incomplete_byte_fallback_set() -> None:
    model_data = valid_model_dict()
    vocab = model_data["vocab"]
    assert isinstance(vocab, list)
    vocab.append({"id": 8, "piece": byte_fallback_piece(0), "score": -100.0, "special": None})
    model_data["vocab_size"] = 9

    with pytest.raises(ModelValidationError, match="byte fallback pieces must include the full 0x00-0xFF set"):
        load_model_dict(model_data)
