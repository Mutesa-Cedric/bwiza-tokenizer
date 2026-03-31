from bwiza_tokenizer_trainer.eval import build_eval_report, compute_metrics, sample_segmentations
from bwiza_tokenizer_trainer.model.load import load_model_dict


def eval_model():
    return load_model_dict(
        {
            "version": "model-v1",
            "name": "eval-demo",
            "model_type": "unigram",
            "vocab_size": 7,
            "normalization": {
                "unicode_form": "NFC",
                "whitespace_policy": "all_whitespace_to_space_then_collapse",
                "trim": True,
                "boundary_marker": "▁",
                "prepend_leading_boundary": True,
            },
            "special_token_ids": {
                "unk": 0,
                "bos": 1,
                "eos": 2,
                "pad": 3,
            },
            "vocab": [
                {"id": 0, "piece": "<unk>", "score": -10.0, "special": "unk"},
                {"id": 1, "piece": "<s>", "score": 0.0, "special": "bos"},
                {"id": 2, "piece": "</s>", "score": 0.0, "special": "eos"},
                {"id": 3, "piece": "<pad>", "score": 0.0, "special": "pad"},
                {"id": 4, "piece": "▁Mu", "score": -1.0, "special": None},
                {"id": 5, "piece": "raho", "score": -1.0, "special": None},
                {"id": 6, "piece": "▁neza", "score": -1.0, "special": None},
            ],
            "trainer": {
                "target_vocab_size": 7,
                "seed_candidate_limit": 16,
                "max_piece_chars": 12,
                "min_candidate_freq": 1,
                "prune_fraction": 0.15,
                "max_iterations": 6,
            },
        }
    )


def test_compute_metrics_reports_unknown_rate() -> None:
    metrics = compute_metrics(["Muraho neza", "Muraho x"], eval_model())

    assert metrics["average_chars_per_token"] > 0.0
    assert metrics["average_tokens_per_document"] > 0.0
    assert metrics["unknown_rate"] > 0.0
    assert 0.0 < metrics["vocab_utilization"] <= 1.0


def test_sample_segmentations_are_human_readable() -> None:
    samples = sample_segmentations(["Muraho neza", "Muraho x"], eval_model(), limit=2)

    assert len(samples) == 2
    assert samples[0]["input"] == "Muraho neza"
    assert samples[0]["normalized"] == "▁Muraho▁neza"
    assert samples[0]["decoded"] == "Muraho neza"
    assert isinstance(samples[1]["pieces"], list)
    assert isinstance(samples[1]["ids"], list)


def test_build_eval_report_includes_metrics_and_samples() -> None:
    report = build_eval_report(["Muraho neza", "Muraho x"], eval_model(), sample_limit=2)

    assert report["model_name"] == "eval-demo"
    assert report["vocab_size"] == 7
    assert "average_chars_per_token" in report
    assert "unknown_rate" in report
    assert len(report["sample_segmentations"]) == 2
