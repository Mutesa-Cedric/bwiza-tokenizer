from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
MODEL_PATH = REPO_ROOT / "tests" / "golden" / "model.v1.json"
CASES_PATH = REPO_ROOT / "tests" / "golden" / "cases.v1.jsonl"
PYTHONPATH = f"{REPO_ROOT / 'trainer' / 'src'}:{REPO_ROOT / 'cli' / 'src'}"


def run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="bwiza-tokenizer-native-smoke-") as temp_dir:
        venv_dir = Path(temp_dir) / "venv"
        run([sys.executable, "-m", "venv", str(venv_dir)])

        bin_dir = venv_dir / ("Scripts" if sys.platform == "win32" else "bin")
        python_bin = bin_dir / ("python.exe" if sys.platform == "win32" else "python")

        run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python_bin), "-m", "pip", "install", "-e", str(RUNTIME_DIR)])

        env = dict(subprocess.os.environ)
        env["PYTHONPATH"] = PYTHONPATH

        parity = run(
            [
                str(python_bin),
                "-m",
                "bwiza_tokenizer_cli.main",
                "parity",
                "--model",
                str(MODEL_PATH),
                "--cases",
                str(CASES_PATH),
                "--runtime",
                "native",
            ],
            env=env,
        )
        parity_payload = json.loads(parity.stdout)

        encode = run(
            [
                str(python_bin),
                "-m",
                "bwiza_tokenizer_cli.main",
                "encode",
                "--model",
                str(MODEL_PATH),
                "--runtime",
                "native",
                "Muraho neza",
            ],
            env=env,
        )
        encode_payload = json.loads(encode.stdout)

        summary = {
            "parity": parity_payload,
            "encode_ids": encode_payload,
            "runtime": parity_payload["runtime"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        if parity_payload["runtime"] != "native":
            raise SystemExit("native smoke used the wrong runtime")
        if parity_payload["failed"] != 0:
            raise SystemExit("native parity reported mismatches")
        if encode_payload != [7, 8, 9]:
            raise SystemExit("native encode smoke returned unexpected ids")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
