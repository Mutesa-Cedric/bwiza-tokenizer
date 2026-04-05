use std::fs;
use std::path::PathBuf;

use bwiza_tokenizer_runtime::decode::{decode_ids, decode_pieces};
use bwiza_tokenizer_runtime::errors::RuntimeError;
use bwiza_tokenizer_runtime::model::{ModelV1, load_model_str};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct FixtureCase {
    ids: Vec<usize>,
    decoded: String,
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
fn decodes_piece_surfaces_to_readable_text() {
    let decoded = decode_pieces(&["▁Mur", "aho", "▁neza"], "▁");

    assert_eq!(decoded, "Muraho neza");
}

#[test]
fn decodes_token_ids_via_model_lookup() {
    let decoded = decode_ids(&[7, 8, 9], &parity_model()).expect("ids should decode");

    assert_eq!(decoded, "Muraho neza");
}

#[test]
fn rejects_unknown_token_ids() {
    let error = decode_ids(&[999], &parity_model()).expect_err("decode should fail");

    assert!(matches!(error, RuntimeError::UnknownTokenId(999)));
    assert_eq!(error.to_string(), "unknown token id 999");
}

#[test]
fn matches_committed_fixture_decoded_outputs() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/golden/cases.v1.jsonl");
    let raw = fs::read_to_string(path).expect("fixture file should exist");
    let model = parity_model();

    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let case: FixtureCase = serde_json::from_str(line).expect("fixture line should parse");
        let decoded = decode_ids(&case.ids, &model).expect("fixture ids should decode");
        assert_eq!(decoded, case.decoded);
    }
}
