.PHONY: setup fmt lint test parity native-smoke bench

setup:
	@echo "Install local packages with: python3 -m pip install -e trainer && python3 -m pip install -e cli"
	@echo "Optional native runtime: python3 -m pip install -e runtime"
	@echo "Optional SentencePiece backend: python3 -m pip install -e 'trainer[sentencepiece]'"
	@echo "Check the Rust crate with: cargo check --manifest-path runtime/Cargo.toml"

fmt:
	@echo "Formatting hooks will be wired in later phases."

lint:
	@echo "Lint hooks will be wired in later phases."

test:
	cd trainer && python3 -m pytest
	cd cli && python3 -m pytest
	cargo test --manifest-path runtime/Cargo.toml
	cargo test --manifest-path runtime/Cargo.toml --features python

parity:
	PYTHONPATH=trainer/src:cli/src python3 -m bwiza_tokenizer_cli.main parity --model tests/golden/model.v1.json --cases tests/golden/cases.v1.jsonl

native-smoke:
	python3 tests/parity/native_runtime_smoke.py

bench:
	@echo "Benchmark command is not implemented yet."
