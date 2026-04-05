use std::fs;
use std::path::PathBuf;

use bwiza_tokenizer_runtime::encode::{encode_ids, encode_pieces};
use bwiza_tokenizer_runtime::model::{ModelV1, load_model_str};
use bwiza_tokenizer_runtime::trie::PieceTrie;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct FixtureCase {
    input: String,
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
fn encodes_pieces_from_raw_text() {
    let model = parity_model();
    let trie = PieceTrie::from_model(&model);

    let pieces = encode_pieces("Muraho neza", &model, &trie).expect("encode should work");
    assert_eq!(
        pieces,
        vec!["▁Mur".to_string(), "aho".to_string(), "▁neza".to_string()]
    );
}

#[test]
fn encodes_ids_from_raw_text() {
    let model = parity_model();
    let trie = PieceTrie::from_model(&model);

    let ids = encode_ids("Muraho neza", &model, &trie).expect("encode should work");
    assert_eq!(ids, vec![7, 8, 9]);
}

#[test]
fn matches_committed_fixture_piece_and_id_outputs() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/golden/cases.v1.jsonl");
    let raw = fs::read_to_string(path).expect("fixture file should exist");
    let model = parity_model();
    let trie = PieceTrie::from_model(&model);

    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let case: FixtureCase = serde_json::from_str(line).expect("fixture line should parse");
        let pieces =
            encode_pieces(case.input.as_str(), &model, &trie).expect("fixture should encode");
        let ids = encode_ids(case.input.as_str(), &model, &trie).expect("fixture should encode");

        assert_eq!(pieces, case.pieces);
        assert_eq!(ids, case.ids);
    }
}
