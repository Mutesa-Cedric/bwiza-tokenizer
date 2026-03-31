from typing import cast

import pytest

from bwiza_tokenizer_trainer.corpus.iterator import iter_from_iterable


def test_iter_from_iterable_preserves_order() -> None:
    docs = ["Muraho", "amakuru"]

    assert list(iter_from_iterable(docs)) == docs


def test_iter_from_iterable_rejects_non_strings() -> None:
    docs = cast(list[str], ["Muraho", 3])

    with pytest.raises(TypeError, match="document 1 must be str, got int"):
        list(iter_from_iterable(docs))
