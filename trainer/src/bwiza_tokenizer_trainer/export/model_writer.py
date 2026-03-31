from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..model.validate import validate_model
from ..types import ModelV1


def model_to_dict(model: ModelV1) -> dict[str, Any]:
    return {
        "version": model.version,
        "name": model.name,
        "model_type": model.model_type,
        "vocab_size": model.vocab_size,
        "normalization": {
            "unicode_form": model.normalization.unicode_form,
            "whitespace_policy": model.normalization.whitespace_policy,
            "trim": model.normalization.trim,
            "boundary_marker": model.normalization.boundary_marker,
            "prepend_leading_boundary": model.normalization.prepend_leading_boundary,
        },
        "special_token_ids": {
            "unk": model.special_token_ids["unk"],
            "bos": model.special_token_ids["bos"],
            "eos": model.special_token_ids["eos"],
            "pad": model.special_token_ids["pad"],
        },
        "vocab": [
            {
                "id": entry.id,
                "piece": entry.piece,
                "score": entry.score,
                "special": entry.special,
            }
            for entry in model.vocab
        ],
        "trainer": dict(sorted(model.trainer.items())),
    }


def write_model_json(
    model: ModelV1,
    path: str | Path,
) -> Path:
    validate_model(model)
    destination = Path(path)
    payload = model_to_dict(model)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
