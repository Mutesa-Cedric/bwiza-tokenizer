from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Literal, Protocol

from bwiza_tokenizer_trainer.model.load import load_model
from bwiza_tokenizer_trainer.normalize.pipeline import normalize_text
from bwiza_tokenizer_trainer.reference_runtime.decode import decode_ids, decode_pieces
from bwiza_tokenizer_trainer.reference_runtime.encode import encode_to_ids, encode_to_pieces
from bwiza_tokenizer_trainer.types import ModelV1

RuntimeMode = Literal["auto", "python", "native"]
RuntimeName = Literal["python", "native"]

_NATIVE_MODULE = "bwiza_tokenizer_runtime"


class TokenizerBackend(Protocol):
    runtime_name: RuntimeName

    def normalize(self, text: str) -> str: ...

    def encode_ids(self, text: str) -> list[int]: ...

    def encode_pieces(self, text: str) -> list[str]: ...

    def decode_ids(self, ids: list[int]) -> str: ...

    def decode_pieces(self, pieces: list[str]) -> str: ...


@dataclass(slots=True)
class PythonTokenizerBackend:
    model: ModelV1
    runtime_name: RuntimeName = "python"

    def normalize(self, text: str) -> str:
        return normalize_text(text, config=self.model.normalization)

    def encode_ids(self, text: str) -> list[int]:
        return encode_to_ids(text, self.model)

    def encode_pieces(self, text: str) -> list[str]:
        return encode_to_pieces(text, self.model)

    def decode_ids(self, ids: list[int]) -> str:
        return decode_ids(ids, self.model)

    def decode_pieces(self, pieces: list[str]) -> str:
        return decode_pieces(pieces, self.model)


@dataclass(slots=True)
class NativeTokenizerBackend:
    tokenizer: object
    runtime_name: RuntimeName = "native"

    def normalize(self, text: str) -> str:
        return self.tokenizer.normalize(text)

    def encode_ids(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def encode_pieces(self, text: str) -> list[str]:
        return list(self.tokenizer.encode_pieces(text))

    def decode_ids(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def decode_pieces(self, pieces: list[str]) -> str:
        return self.tokenizer.decode_pieces(pieces)


def load_backend(model_path: str, *, runtime: RuntimeMode) -> TokenizerBackend:
    if runtime == "python":
        return PythonTokenizerBackend(model=load_model(model_path))

    if runtime == "native":
        return _load_native_backend(model_path)

    try:
        return _load_native_backend(model_path)
    except ModuleNotFoundError:
        return PythonTokenizerBackend(model=load_model(model_path))


def _load_native_backend(model_path: str) -> NativeTokenizerBackend:
    module = importlib.import_module(_NATIVE_MODULE)
    tokenizer = module.Tokenizer(model_path)
    return NativeTokenizerBackend(tokenizer=tokenizer)
