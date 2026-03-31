import json
from pathlib import Path

from bwiza_tokenizer_trainer import decode_ids, normalize_text
from bwiza_tokenizer_trainer.model.load import load_model_dict
from bwiza_tokenizer_trainer.reference_runtime.encode import encode_to_ids, encode_to_pieces


def parity_model():
    return load_model_dict(
        {
            "version": "model-v1",
            "name": "parity-demo",
            "model_type": "unigram",
            "vocab_size": 14,
            "normalization": {
                "unicode_form": "NFC",
                "whitespace_policy": "all_whitespace_to_space_then_collapse",
                "trim": True,
                "boundary_marker": "▁",
                "prepend_leading_boundary": True,
            },
            "special_token_ids": {"unk": 0, "bos": 1, "eos": 2, "pad": 3},
            "vocab": [
                {"id": 0, "piece": "<unk>", "score": -10.0, "special": "unk"},
                {"id": 1, "piece": "<s>", "score": 0.0, "special": "bos"},
                {"id": 2, "piece": "</s>", "score": 0.0, "special": "eos"},
                {"id": 3, "piece": "<pad>", "score": 0.0, "special": "pad"},
                {"id": 4, "piece": "▁", "score": -10.0, "special": None},
                {"id": 5, "piece": "▁Mu", "score": -1.0, "special": None},
                {"id": 6, "piece": "raho", "score": -3.0, "special": None},
                {"id": 7, "piece": "▁Mur", "score": -2.0, "special": None},
                {"id": 8, "piece": "aho", "score": -2.0, "special": None},
                {"id": 9, "piece": "▁neza", "score": -1.5, "special": None},
                {"id": 10, "piece": ".", "score": -1.0, "special": None},
                {"id": 11, "piece": "▁world", "score": -1.2, "special": None},
                {"id": 12, "piece": "▁2026-03-31", "score": -1.0, "special": None},
                {"id": 13, "piece": "▁Café", "score": -1.0, "special": None},
            ],
            "trainer": {
                "target_vocab_size": 14,
                "seed_candidate_limit": 64,
                "max_piece_chars": 16,
                "min_candidate_freq": 1,
                "prune_fraction": 0.15,
                "max_iterations": 6,
            },
        }
    )


def load_cases():
    root = Path(__file__).resolve().parents[2]
    path = root / "tests" / "golden" / "cases.v1.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_parity_fixture_file_covers_required_categories() -> None:
    case_ids = {case["case_id"] for case in load_cases()}

    assert case_ids == {
        "empty_001",
        "boundary_001",
        "punct_001",
        "whitespace_001",
        "unicode_001",
        "digits_001",
        "mixed_001",
        "long_001",
        "unknown_001",
        "tiebreak_001",
    }


def test_parity_fixtures_match_python_reference_behavior() -> None:
    model = parity_model()

    for case in load_cases():
        normalized = normalize_text(case["input"], config=model.normalization)
        pieces = encode_to_pieces(case["input"], model)
        ids = encode_to_ids(case["input"], model)
        decoded = decode_ids(ids, model)

        assert normalized == case["normalized"]
        assert pieces == case["pieces"]
        assert ids == case["ids"]
        assert decoded == case["decoded"]
