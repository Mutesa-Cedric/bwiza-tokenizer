.PHONY: setup fmt lint test parity bench

setup:
	@echo "Install local packages with: python3 -m pip install -e trainer && python3 -m pip install -e cli"
	@echo "Check the Rust crate with: cargo check --manifest-path runtime/Cargo.toml"

fmt:
	@echo "Formatting hooks will be wired in later phases."

lint:
	@echo "Lint hooks will be wired in later phases."

test:
	cd trainer && python3 -m pytest
	cd cli && python3 -m pytest
	cargo test --manifest-path runtime/Cargo.toml

parity:
	@echo "Parity command is not implemented yet."

bench:
	@echo "Benchmark command is not implemented yet."
