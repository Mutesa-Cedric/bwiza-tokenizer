use bwiza_tokenizer_runtime::byte_fallback::byte_fallback_piece;
use bwiza_tokenizer_runtime::errors::RuntimeError;
use bwiza_tokenizer_runtime::model::{ModelV1, load_model_str};

fn valid_model_json() -> String {
    r#"{
  "version": "model-v1",
  "name": "demo-unigram",
  "model_type": "unigram",
  "vocab_size": 8,
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
    {"id": 0, "piece": "<unk>", "score": 0.0, "special": "unk"},
    {"id": 1, "piece": "<s>", "score": 0.0, "special": "bos"},
    {"id": 2, "piece": "</s>", "score": 0.0, "special": "eos"},
    {"id": 3, "piece": "<pad>", "score": 0.0, "special": "pad"},
    {"id": 4, "piece": "▁Mu", "score": -1.9, "special": null},
    {"id": 5, "piece": "raho", "score": -2.3, "special": null},
    {"id": 6, "piece": "▁neza", "score": -2.1, "special": null},
    {"id": 7, "piece": ".", "score": -3.0, "special": null}
  ],
  "trainer": {
    "max_iterations": 6,
    "max_piece_chars": 12,
    "min_candidate_freq": 1,
    "prune_fraction": 0.15,
    "seed_candidate_limit": 64,
    "target_vocab_size": 8
  }
}"#
    .to_string()
}

fn expect_validation_error(raw: &str) -> String {
    match load_model_str(raw) {
        Err(RuntimeError::Validation(message)) => message,
        other => panic!("expected validation error, got {other:?}"),
    }
}

#[test]
fn loads_valid_model_v1() {
    let model: ModelV1 = load_model_str(&valid_model_json()).expect("model should load");

    assert_eq!(model.vocab_size, 8);
    assert_eq!(model.special_token_ids.unk, 0);
    assert_eq!(model.vocab[4].piece, "▁Mu");
}

#[test]
fn rejects_wrong_version() {
    let message =
        expect_validation_error(&valid_model_json().replace("\"model-v1\"", "\"model-v2\""));

    assert_eq!(
        message,
        "model version must be \"model-v1\", got \"model-v2\""
    );
}

#[test]
fn rejects_wrong_model_type() {
    let message = expect_validation_error(&valid_model_json().replace("\"unigram\"", "\"bpe\""));

    assert_eq!(message, "model_type must be \"unigram\", got \"bpe\"");
}

#[test]
fn rejects_vocab_size_mismatch() {
    let message = expect_validation_error(
        &valid_model_json().replace("\"vocab_size\": 8", "\"vocab_size\": 7"),
    );

    assert_eq!(message, "vocab_size must equal len(vocab), got 7 and 8");
}

#[test]
fn rejects_non_contiguous_ids() {
    let message = expect_validation_error(&valid_model_json().replace("\"id\": 7", "\"id\": 8"));

    assert_eq!(message, "vocab ids must be unique and contiguous from 0");
}

#[test]
fn rejects_duplicate_pieces() {
    let message = expect_validation_error(
        &valid_model_json().replace("\"piece\": \".\"", "\"piece\": \"raho\""),
    );

    assert_eq!(message, "vocab pieces must be unique");
}

#[test]
fn rejects_wrong_special_token_mapping() {
    let message = expect_validation_error(&valid_model_json().replace("\"unk\": 0", "\"unk\": 9"));

    assert_eq!(message, "special_token_ids must match the fixed v1 mapping");
}

#[test]
fn rejects_incomplete_byte_fallback_set() {
    let message = expect_validation_error(
        &valid_model_json()
            .replace("\"vocab_size\": 8", "\"vocab_size\": 9")
            .replace(
                "  ],\n  \"trainer\": {",
                format!(
                    "    ,{{\"id\": 8, \"piece\": \"{}\", \"score\": -100.0, \"special\": null}}\n  ],\n  \"trainer\": {{",
                    byte_fallback_piece(0)
                )
                .as_str(),
            ),
    );

    assert_eq!(
        message,
        "byte fallback pieces must include the full 0x00-0xFF set when enabled"
    );
}
