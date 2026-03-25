from bwiza_tokenizer_trainer import ModelV1, TrainerConfig


def test_public_exports_exist() -> None:
    assert TrainerConfig.__name__ == "TrainerConfig"
    assert ModelV1.__name__ == "ModelV1"
