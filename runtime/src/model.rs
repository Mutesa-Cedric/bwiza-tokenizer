use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::errors::RuntimeError;

const MODEL_VERSION: &str = "model-v1";
const MODEL_TYPE: &str = "unigram";

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct NormalizationConfig {
    pub unicode_form: String,
    pub whitespace_policy: String,
    pub trim: bool,
    pub boundary_marker: String,
    pub prepend_leading_boundary: bool,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            unicode_form: "NFC".to_string(),
            whitespace_policy: "all_whitespace_to_space_then_collapse".to_string(),
            trim: true,
            boundary_marker: "▁".to_string(),
            prepend_leading_boundary: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SpecialTokenIds {
    pub unk: usize,
    pub bos: usize,
    pub eos: usize,
    pub pad: usize,
}

impl Default for SpecialTokenIds {
    fn default() -> Self {
        Self {
            unk: 0,
            bos: 1,
            eos: 2,
            pad: 3,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct VocabEntry {
    pub id: usize,
    pub piece: String,
    pub score: f64,
    pub special: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ModelV1 {
    pub version: String,
    pub name: String,
    pub model_type: String,
    pub vocab_size: usize,
    pub normalization: NormalizationConfig,
    pub special_token_ids: SpecialTokenIds,
    pub vocab: Vec<VocabEntry>,
    pub trainer: BTreeMap<String, Value>,
}

pub fn load_model(path: impl AsRef<Path>) -> Result<ModelV1, RuntimeError> {
    let raw = fs::read_to_string(path)?;
    load_model_str(&raw)
}

pub fn load_model_str(raw: &str) -> Result<ModelV1, RuntimeError> {
    let model: ModelV1 = serde_json::from_str(raw)?;
    validate_model(&model)?;
    Ok(model)
}

pub fn validate_model(model: &ModelV1) -> Result<(), RuntimeError> {
    if model.version != MODEL_VERSION {
        return Err(RuntimeError::Validation(format!(
            "model version must be {MODEL_VERSION:?}, got {:?}",
            model.version
        )));
    }

    if model.model_type != MODEL_TYPE {
        return Err(RuntimeError::Validation(format!(
            "model_type must be {MODEL_TYPE:?}, got {:?}",
            model.model_type
        )));
    }

    if model.normalization != NormalizationConfig::default() {
        return Err(RuntimeError::Validation(
            "normalization config does not match normalization-v1".to_string(),
        ));
    }

    if model.special_token_ids != SpecialTokenIds::default() {
        return Err(RuntimeError::Validation(
            "special_token_ids must match the fixed v1 mapping".to_string(),
        ));
    }

    if model.vocab_size != model.vocab.len() {
        return Err(RuntimeError::Validation(format!(
            "vocab_size must equal len(vocab), got {} and {}",
            model.vocab_size,
            model.vocab.len()
        )));
    }

    let actual_ids: BTreeSet<usize> = model.vocab.iter().map(|entry| entry.id).collect();
    let expected_ids: BTreeSet<usize> = (0..model.vocab.len()).collect();
    if actual_ids != expected_ids {
        return Err(RuntimeError::Validation(
            "vocab ids must be unique and contiguous from 0".to_string(),
        ));
    }

    let piece_count = model
        .vocab
        .iter()
        .map(|entry| entry.piece.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    if piece_count != model.vocab.len() {
        return Err(RuntimeError::Validation(
            "vocab pieces must be unique".to_string(),
        ));
    }

    for entry in &model.vocab {
        if let Some(special) = &entry.special {
            if !matches!(special.as_str(), "unk" | "bos" | "eos" | "pad") {
                return Err(RuntimeError::Validation(format!(
                    "unsupported special token name {special:?}"
                )));
            }
        }
    }

    for (special_name, expected_id) in [
        ("unk", model.special_token_ids.unk),
        ("bos", model.special_token_ids.bos),
        ("eos", model.special_token_ids.eos),
        ("pad", model.special_token_ids.pad),
    ] {
        let matches = model
            .vocab
            .iter()
            .filter(|entry| entry.special.as_deref() == Some(special_name))
            .collect::<Vec<_>>();

        if matches.len() != 1 {
            return Err(RuntimeError::Validation(format!(
                "expected exactly one vocab entry with special={special_name:?}"
            )));
        }

        if matches[0].id != expected_id {
            return Err(RuntimeError::Validation(format!(
                "special token {special_name:?} must use id {expected_id}"
            )));
        }
    }

    Ok(())
}
