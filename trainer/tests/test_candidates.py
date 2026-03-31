from bwiza_tokenizer_trainer import TrainerConfig
from bwiza_tokenizer_trainer.train import enumerate_seed_candidates


def test_seed_candidates_keep_observed_single_scalars() -> None:
    config = TrainerConfig(
        seed_candidate_limit=16,
        max_piece_chars=2,
        min_candidate_freq=2,
    )

    candidates = enumerate_seed_candidates(["▁ab", "▁ac"], config)
    by_piece = {candidate.piece: candidate for candidate in candidates}

    assert by_piece["▁"].protected is True
    assert by_piece["a"].protected is True
    assert by_piece["b"].protected is True
    assert by_piece["c"].protected is True
    assert by_piece["b"].count == 1
    assert by_piece["c"].count == 1


def test_seed_candidates_are_deterministic_for_same_input() -> None:
    config = TrainerConfig(
        seed_candidate_limit=16,
        max_piece_chars=3,
        min_candidate_freq=1,
    )
    docs = ["▁aba", "▁aba", "▁aca"]

    first = enumerate_seed_candidates(docs, config)
    second = enumerate_seed_candidates(docs, config)

    assert first == second


def test_seed_candidates_apply_limit_after_protected_pieces() -> None:
    config = TrainerConfig(
        seed_candidate_limit=5,
        max_piece_chars=2,
        min_candidate_freq=1,
    )

    candidates = enumerate_seed_candidates(["▁ab", "▁ab", "▁ac"], config)
    pieces = [candidate.piece for candidate in candidates]

    assert pieces == ["a", "b", "c", "▁", "▁a"]


def test_seed_candidates_reject_non_string_documents() -> None:
    config = TrainerConfig()

    try:
        enumerate_seed_candidates(["▁ab", 3], config)  # type: ignore[list-item]
    except TypeError as exc:
        assert str(exc) == "normalized_docs must yield strings"
    else:
        raise AssertionError("enumerate_seed_candidates should reject non-string documents")
