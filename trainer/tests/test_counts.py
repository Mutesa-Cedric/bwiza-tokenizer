from bwiza_tokenizer_trainer.train import SegmentationResult, count_piece_usage


def test_count_piece_usage_aggregates_segment_ids() -> None:
    counts = count_piece_usage(
        [
            SegmentationResult(score=-2.0, ids=(4, 5), pieces=("▁Mu", "raho")),
            SegmentationResult(score=-2.1, ids=(4, 6), pieces=("▁Mu", "▁neza")),
            SegmentationResult(score=-2.1, ids=(), pieces=()),
        ]
    )

    assert counts == {4: 2, 5: 1, 6: 1}
