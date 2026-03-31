from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_eval_json(
    report: dict[str, Any],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
