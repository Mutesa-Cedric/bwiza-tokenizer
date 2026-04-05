import json
from types import SimpleNamespace

import pytest

from bwiza_tokenizer_trainer.model.load import load_model_dict
from bwiza_tokenizer_trainer.types import ModelV1, VocabEntry
from bwiza_tokenizer_trainer.train.native_runtime import (
    count_piece_usage_native,
    enumerate_seed_candidates_native,
    load_native_tokenizer,
)


def native_model():
    return load_model_dict(
        {
            "version": "model-v1",
            "name": "native-demo",
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
                {"id": 4, "piece": "▁Mu", "score": -1.0, "special": None},
                {"id": 5, "piece": "raho", "score": -1.0, "special": None},
            ],
            "trainer": {
                "target_vocab_size": 6,
                "seed_candidate_limit": 16,
                "max_piece_chars": 12,
                "min_candidate_freq": 1,
                "prune_fraction": 0.15,
                "max_iterations": 6,
            },
        }
    )


class FakeTokenizer:
    def __init__(self, payload: str) -> None:
        self.payload = json.loads(payload)

    def count_piece_usage_normalized(self, texts: list[str]) -> dict[int, int]:
        return {4: len(texts), 5: len(texts)}

    def count_piece_usage_normalized_dense(self, texts: list[str]) -> list[int]:
        counts = [0] * len(self.payload["vocab"])
        counts[4] = len(texts)
        counts[5] = len(texts)
        return counts


class FakeTokenizerType:
    @staticmethod
    def from_json(payload: str) -> FakeTokenizer:
        return FakeTokenizer(payload)


def fake_module() -> SimpleNamespace:
    return SimpleNamespace(
        Tokenizer=FakeTokenizerType,
        enumerate_seed_candidates_normalized=lambda texts, max_piece_chars, min_candidate_freq, seed_candidate_limit: [
            ("a", 4, True),
            ("b", 2, True),
            ("ab", 2, False),
        ],
    )


def test_load_native_tokenizer_uses_json_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bwiza_tokenizer_trainer.train.native_runtime.importlib.import_module",
        lambda name: fake_module(),
    )

    tokenizer = load_native_tokenizer(native_model())

    assert tokenizer is not None
    assert isinstance(tokenizer.tokenizer, FakeTokenizer)
    assert tokenizer.tokenizer.payload["name"] == "native-demo"
    assert tokenizer.tokenizer.payload["special_token_ids"]["unk"] == 0
    assert tokenizer.native_to_model_ids == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def test_count_piece_usage_native_batches_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bwiza_tokenizer_trainer.train.native_runtime.importlib.import_module",
        lambda name: fake_module(),
    )

    counts = count_piece_usage_native(
        ["▁Muraho", "▁Muraho", "▁Muraho"],
        native_model(),
        chunk_size=2,
    )

    assert counts == {4: 3, 5: 3}


def test_count_piece_usage_native_returns_none_without_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "bwiza_tokenizer_trainer.train.native_runtime.importlib.import_module",
        raise_missing,
    )

    assert count_piece_usage_native(["▁Muraho"], native_model()) is None


def test_load_native_tokenizer_reindexes_sparse_working_vocab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bwiza_tokenizer_trainer.train.native_runtime.importlib.import_module",
        lambda name: fake_module(),
    )

    model = ModelV1(
        name="sparse-demo",
        vocab_size=5,
        special_token_ids={"unk": 0, "bos": 1, "eos": 2, "pad": 3},
        vocab=[
            VocabEntry(id=0, piece="<unk>", score=-10.0, special="unk"),
            VocabEntry(id=1, piece="<s>", score=0.0, special="bos"),
            VocabEntry(id=2, piece="</s>", score=0.0, special="eos"),
            VocabEntry(id=3, piece="<pad>", score=0.0, special="pad"),
            VocabEntry(id=9, piece="▁Muraho", score=-1.0, special=None),
        ],
    )

    tokenizer = load_native_tokenizer(model)

    assert tokenizer is not None
    assert tokenizer.native_to_model_ids[4] == 9
    assert tokenizer.tokenizer.payload["vocab"][-1]["id"] == 4


def test_enumerate_seed_candidates_native_uses_native_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "bwiza_tokenizer_trainer.train.native_runtime.importlib.import_module",
        lambda name: fake_module(),
    )

    candidates = enumerate_seed_candidates_native(
        ["aba", "aba"],
        SimpleNamespace(
            max_piece_chars=2,
            min_candidate_freq=1,
            seed_candidate_limit=4,
        ),
    )

    assert candidates is not None
    assert [(candidate.piece, candidate.count, candidate.protected) for candidate in candidates] == [
        ("a", 4, True),
        ("b", 2, True),
        ("ab", 2, False),
    ]
