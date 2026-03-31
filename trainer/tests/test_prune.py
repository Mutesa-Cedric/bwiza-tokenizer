from bwiza_tokenizer_trainer.train import is_protected_piece, prune_vocabulary, split_protected_entries
from bwiza_tokenizer_trainer.types import VocabEntry


def vocab_fixture() -> list[VocabEntry]:
    return [
        VocabEntry(id=0, piece="<unk>", score=0.0, special="unk"),
        VocabEntry(id=1, piece="<s>", score=0.0, special="bos"),
        VocabEntry(id=2, piece="</s>", score=0.0, special="eos"),
        VocabEntry(id=3, piece="<pad>", score=0.0, special="pad"),
        VocabEntry(id=4, piece="▁", score=-1.0),
        VocabEntry(id=5, piece="a", score=-1.0),
        VocabEntry(id=6, piece="ab", score=-5.0),
        VocabEntry(id=7, piece="ba", score=-5.0),
        VocabEntry(id=8, piece="abc", score=-2.0),
    ]


def test_is_protected_piece_covers_specials_and_single_scalars() -> None:
    vocab = vocab_fixture()

    assert is_protected_piece(vocab[0]) is True
    assert is_protected_piece(vocab[4]) is True
    assert is_protected_piece(vocab[5]) is True
    assert is_protected_piece(vocab[6]) is False


def test_split_protected_entries_keeps_boundary_support() -> None:
    protected, removable = split_protected_entries(vocab_fixture())

    assert [entry.id for entry in protected] == [0, 1, 2, 3, 4, 5]
    assert [entry.id for entry in removable] == [6, 7, 8]


def test_prune_vocabulary_never_removes_protected_entries() -> None:
    pruned = prune_vocabulary(vocab_fixture(), prune_fraction=1.0)

    assert [entry.id for entry in pruned] == [0, 1, 2, 3, 4, 5]


def test_prune_vocabulary_uses_deterministic_tie_breaks() -> None:
    pruned = prune_vocabulary(vocab_fixture(), prune_fraction=1 / 3)

    assert [entry.id for entry in pruned] == [0, 1, 2, 3, 4, 5, 7, 8]


def test_prune_vocabulary_honors_minimum_vocab_size() -> None:
    pruned = prune_vocabulary(
        vocab_fixture(),
        prune_fraction=1.0,
        minimum_vocab_size=8,
    )

    assert len(pruned) == 8
    assert [entry.id for entry in pruned] == [0, 1, 2, 3, 4, 5, 7, 8]
