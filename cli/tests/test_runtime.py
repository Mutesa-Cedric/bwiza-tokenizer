from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bwiza_tokenizer_cli.runtime import NativeTokenizerBackend, PythonTokenizerBackend, load_backend

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = str(REPO_ROOT / "tests" / "golden" / "model.v1.json")


class FakeNativeTokenizer:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def normalize(self, text: str) -> str:
        return f"native:{text}"

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def encode_pieces(self, text: str) -> list[str]:
        return ["▁native"]

    def decode(self, ids: list[int]) -> str:
        return "decoded-native"

    def decode_pieces(self, pieces: list[str]) -> str:
        return "decoded-native-pieces"


def test_auto_runtime_falls_back_to_python_when_native_module_is_missing(monkeypatch) -> None:
    def raise_missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("bwiza_tokenizer_cli.runtime.importlib.import_module", raise_missing)

    backend = load_backend(MODEL_PATH, runtime="auto")

    assert isinstance(backend, PythonTokenizerBackend)
    assert backend.runtime_name == "python"


def test_native_runtime_requires_installed_module(monkeypatch) -> None:
    def raise_missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("bwiza_tokenizer_cli.runtime.importlib.import_module", raise_missing)

    with pytest.raises(ModuleNotFoundError):
        load_backend(MODEL_PATH, runtime="native")


def test_native_runtime_uses_extension_when_available(monkeypatch) -> None:
    fake_module = SimpleNamespace(Tokenizer=FakeNativeTokenizer)
    monkeypatch.setattr(
        "bwiza_tokenizer_cli.runtime.importlib.import_module",
        lambda name: fake_module,
    )

    backend = load_backend(MODEL_PATH, runtime="native")

    assert isinstance(backend, NativeTokenizerBackend)
    assert backend.runtime_name == "native"
    assert backend.normalize("Muraho") == "native:Muraho"
    assert backend.encode_ids("Muraho") == [1, 2, 3]
    assert backend.encode_pieces("Muraho") == ["▁native"]
    assert backend.decode_ids([1, 2, 3]) == "decoded-native"
    assert backend.decode_pieces(["▁native"]) == "decoded-native-pieces"
