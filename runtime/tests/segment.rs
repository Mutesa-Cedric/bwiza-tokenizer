use std::fs;
use std::path::PathBuf;

use bwiza_tokenizer_runtime::model::{load_model_str, ModelV1};
use bwiza_tokenizer_runtime::segment::segment_normalized;
use bwiza_tokenizer_runtime::trie::PieceTrie;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct FixtureCase {
    normalized: String,
    pieces: Vec<String>,
    ids: Vec<usize>,
}

fn parity_model() -> ModelV1 {
    load_model_str(
        r#"{
  "version": "model-v1",
  "name": "parity-demo",
  "model_type": "unigram",
  "vocab_size": 14,
  "normalization": {
    "unicode_form": "NFC",
    "whitespace_policy": "all_whitespace_to_space_then_collapse",
    "trim": true,
    "boundary_marker": "▁",
    "prepend_leading_boundary": true
  },
  "special_token_ids": {
    "unk": 0,
    "bos": 1,
    "eos": 2,
    "pad": 3
  },
  "vocab": [
    {"id": 0, "piece": "<unk>", "score": -10.0, "special": "unk"},
    {"id": 1, "piece": "<s>", "score": 0.0, "special": "bos"},
    {"id": 2, "piece": "</s>", "score": 0.0, "special": "eos"},
    {"id": 3, "piece": "<pad>", "score": 0.0, "special": "pad"},
    {"id": 4, "piece": "▁", "score": -10.0, "special": null},
    {"id": 5, "piece": "▁Mu", "score": -1.0, "special": null},
    {"id": 6, "piece": "raho", "score": -3.0, "special": null},
    {"id": 7, "piece": "▁Mur", "score": -2.0, "special": null},
    {"id": 8, "piece": "aho", "score": -2.0, "special": null},
    {"id": 9, "piece": "▁neza", "score": -1.5, "special": null},
    {"id": 10, "piece": ".", "score": -1.0, "special": null},
    {"id": 11, "piece": "▁world", "score": -1.2, "special": null},
    {"id": 12, "piece": "▁2026-03-31", "score": -1.0, "special": null},
    {"id": 13, "piece": "▁Café", "score": -1.0, "special": null}
  ],
  "trainer": {
    "target_vocab_size": 14,
    "seed_candidate_limit": 64,
    "max_piece_chars": 16,
    "min_candidate_freq": 1,
    "prune_fraction": 0.15,
    "max_iterations": 6
  }
}"#,
    )
    .expect("parity demo model should load")
}

#[test]
fn prefers_longer_earliest_piece_when_scores_tie() {
    let model = parity_model();
    let trie = PieceTrie::from_model(&model);
    let result = segment_normalized("▁Muraho", &model, &trie).expect("segmentation should work");

    assert_eq!(result.ids, vec![7, 8]);
    assert_eq!(result.pieces, vec!["▁Mur".to_string(), "aho".to_string()]);
    assert_eq!(result.score, -4.0);
}

#[test]
fn uses_unknown_fallback_one_scalar_at_a_time() {
    let model = parity_model();
    let trie = PieceTrie::from_model(&model);
    let result = segment_normalized("▁Muraho▁x", &model, &trie).expect("segmentation should work");

    assert_eq!(result.ids, vec![7, 8, 4, 0]);
    assert_eq!(
        result.pieces,
        vec![
            "▁Mur".to_string(),
            "aho".to_string(),
            "▁".to_string(),
            "<unk>".to_string(),
        ]
    );
}

#[test]
fn segmentation_is_deterministic() {
    let model = parity_model();
    let trie = PieceTrie::from_model(&model);

    let first = segment_normalized("▁Muraho", &model, &trie).expect("segmentation should work");
    let second = segment_normalized("▁Muraho", &model, &trie).expect("segmentation should work");

    assert_eq!(first, second);
}

#[test]
fn matches_committed_fixture_piece_and_id_outputs() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/golden/cases.v1.jsonl");
    let raw = fs::read_to_string(path).expect("fixture file should exist");
    let model = parity_model();
    let trie = PieceTrie::from_model(&model);

    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let case: FixtureCase = serde_json::from_str(line).expect("fixture line should parse");
        let result =
            segment_normalized(case.normalized.as_str(), &model, &trie).expect("fixture should segment");

        assert_eq!(result.pieces, case.pieces);
        assert_eq!(result.ids, case.ids);
    }
}
