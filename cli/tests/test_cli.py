from bwiza_tokenizer_cli.main import build_parser


def test_cli_parser_has_expected_program_name() -> None:
    assert build_parser().prog == "bwiza-tokenizer"
