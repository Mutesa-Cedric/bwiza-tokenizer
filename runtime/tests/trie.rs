use bwiza_tokenizer_runtime::model::{load_model_str, ModelV1};
use bwiza_tokenizer_runtime::trie::PieceTrie;

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
fn returns_expected_candidates_at_start_offset() {
    let trie = PieceTrie::from_model(&parity_model());

    assert_eq!(trie.candidate_ids_at("▁Muraho", 0), vec![4, 5, 7]);
}

#[test]
fn returns_expected_candidates_at_inner_offset() {
    let trie = PieceTrie::from_model(&parity_model());
    let text = "▁Muraho▁neza";
    let offset = text.char_indices().nth(4).map(|(index, _)| index).unwrap();

    assert_eq!(trie.candidate_ids_at(text, offset), vec![8]);
}

#[test]
fn special_tokens_never_appear_as_candidates() {
    let trie = PieceTrie::from_model(&parity_model());
    let candidates = trie.candidate_ids_at("▁Muraho", 0);

    assert!(!candidates.contains(&0));
    assert!(!candidates.contains(&1));
    assert!(!candidates.contains(&2));
    assert!(!candidates.contains(&3));
}

#[test]
fn non_boundary_offsets_return_no_candidates() {
    let trie = PieceTrie::from_model(&parity_model());
    let text = "▁Muraho";

    assert!(trie.candidate_ids_at(text, 1).is_empty());
}
