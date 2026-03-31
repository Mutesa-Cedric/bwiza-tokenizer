use std::fs;
use std::path::PathBuf;

use bwiza_tokenizer_runtime::normalize::normalize_text;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct FixtureCase {
    input: String,
    normalized: String,
}

#[test]
fn normalizes_basic_text() {
    assert_eq!(normalize_text("Muraho neza"), "▁Muraho▁neza");
}

#[test]
fn collapses_tabs_and_newlines() {
    assert_eq!(normalize_text("  Muraho\t\tneza\n"), "▁Muraho▁neza");
}

#[test]
fn normalizes_unicode_to_nfc() {
    assert_eq!(normalize_text("Cafe\u{301}"), "▁Café");
}

#[test]
fn preserves_punctuation_adjacency() {
    assert_eq!(normalize_text("Muraho."), "▁Muraho.");
}

#[test]
fn empty_or_whitespace_only_input_becomes_empty() {
    assert_eq!(normalize_text(""), "");
    assert_eq!(normalize_text(" \t\n "), "");
}

#[test]
fn matches_committed_fixture_normalized_outputs() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/golden/cases.v1.jsonl");
    let raw = fs::read_to_string(path).expect("fixture file should exist");

    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let case: FixtureCase = serde_json::from_str(line).expect("fixture line should parse");
        assert_eq!(normalize_text(&case.input), case.normalized);
    }
}
