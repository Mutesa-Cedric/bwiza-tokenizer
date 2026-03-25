# bwiza-tokenizer

`bwiza-tokenizer` is a public tokenizer toolkit focused on Kinyarwanda text.

The project is targeting a Kinyarwanda-first unigram tokenizer with:
- an explicit, versioned model format
- a Python reference trainer and evaluator
- a Rust runtime for loading, encoding, and decoding
- exact parity checks across implementations

The goal is to ship a tokenizer that is correct, inspectable, reproducible, and
portable across different training and inference environments.

Scope:
- tokenizer specification
- tokenizer training
- tokenizer evaluation
- tokenizer runtime
- tokenizer parity testing

Out of scope:
- model training
- model serving
- corpus collection and storage infrastructure

Local development:
- Python 3.12 or newer
- Rust stable toolchain
