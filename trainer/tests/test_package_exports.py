from bwiza_tokenizer_trainer import ModelV1, NormalizationConfig, TrainerConfig, VocabEntry


def test_public_exports_exist() -> None:
    assert TrainerConfig.__name__ == "TrainerConfig"
    assert ModelV1.__name__ == "ModelV1"
    assert NormalizationConfig.__name__ == "NormalizationConfig"
    assert VocabEntry.__name__ == "VocabEntry"
