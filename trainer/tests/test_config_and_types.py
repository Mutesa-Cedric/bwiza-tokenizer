from bwiza_tokenizer_trainer.config import TrainerConfig
from bwiza_tokenizer_trainer.types import ModelV1, NormalizationConfig, VocabEntry


def test_trainer_config_defaults_match_v1_contract() -> None:
    config = TrainerConfig()

    assert config.model_type == "unigram"
    assert config.vocab_size == 16000
    assert config.seed_candidate_multiplier == 8
    assert config.seed_candidate_limit == 128000
    assert config.max_piece_chars == 24
    assert config.min_candidate_freq == 2
    assert config.prune_fraction == 0.15
    assert config.max_iterations == 20
    assert config.min_score_delta == 1e-4
    assert config.reserved_special_tokens == 4


def test_model_defaults_match_v1_contract() -> None:
    model = ModelV1()

    assert model.version == "model-v1"
    assert model.model_type == "unigram"
    assert model.vocab_size == 16000
    assert model.special_token_ids == {"unk": 0, "bos": 1, "eos": 2, "pad": 3}
    assert model.trainer["target_vocab_size"] == 16000
    assert model.trainer["seed_candidate_limit"] == 128000
    assert model.trainer["max_piece_chars"] == 24
    assert model.trainer["min_candidate_freq"] == 2
    assert model.trainer["prune_fraction"] == 0.15
    assert model.trainer["max_iterations"] == 20


def test_normalization_and_vocab_entry_shapes_are_explicit() -> None:
    normalization = NormalizationConfig()
    entry = VocabEntry(id=0, piece="<unk>", special="unk")

    assert normalization.unicode_form == "NFC"
    assert normalization.whitespace_policy == "all_whitespace_to_space_then_collapse"
    assert normalization.trim is True
    assert normalization.boundary_marker == "▁"
    assert normalization.prepend_leading_boundary is True
    assert entry.id == 0
    assert entry.piece == "<unk>"
    assert entry.special == "unk"
