use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::decode::{decode_ids, decode_pieces};
use crate::encode::{encode_ids, encode_pieces};
use crate::errors::RuntimeError;
use crate::model::{load_model, load_model_str};
use crate::normalize::normalize_text;
use crate::segment::segment_normalized;
use crate::trie::PieceTrie;

#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    model: crate::model::ModelV1,
    trie: PieceTrie,
}

impl PyTokenizer {
    fn from_path(path: impl AsRef<Path>) -> Result<Self, RuntimeError> {
        let model = load_model(path)?;
        let trie = PieceTrie::from_model(&model);
        Ok(Self { model, trie })
    }

    fn from_json_str(model_json: &str) -> Result<Self, RuntimeError> {
        let model = load_model_str(model_json)?;
        let trie = PieceTrie::from_model(&model);
        Ok(Self { model, trie })
    }
}

fn to_py_err(error: RuntimeError) -> PyErr {
    match error {
        RuntimeError::Io(inner) => PyOSError::new_err(inner.to_string()),
        RuntimeError::Json(inner) => PyValueError::new_err(inner.to_string()),
        RuntimeError::UnknownTokenId(token_id) => {
            PyValueError::new_err(format!("unknown token id {token_id}"))
        }
        RuntimeError::Validation(message) => PyValueError::new_err(message),
    }
}

fn count_piece_usage_dense(
    texts: Vec<String>,
    model: &crate::model::ModelV1,
    trie: &PieceTrie,
) -> Result<Vec<usize>, RuntimeError> {
    let mut counts = vec![0usize; model.vocab.len()];

    for normalized in texts {
        let segmentation = segment_normalized(normalized.as_str(), model, trie)?;
        for token_id in segmentation.ids {
            counts[token_id] += 1;
        }
    }

    Ok(counts)
}

fn enumerate_seed_candidates(
    texts: Vec<String>,
    max_piece_chars: usize,
    min_candidate_freq: usize,
    seed_candidate_limit: usize,
) -> Vec<(String, usize, bool)> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut protected_pieces: BTreeSet<String> = BTreeSet::new();

    for normalized in texts {
        if normalized.is_empty() {
            continue;
        }

        for character in normalized.chars() {
            protected_pieces.insert(character.to_string());
        }

        let mut char_offsets = normalized
            .char_indices()
            .map(|(offset, _)| offset)
            .collect::<Vec<_>>();
        char_offsets.push(normalized.len());
        let char_count = char_offsets.len().saturating_sub(1);

        for start_index in 0..char_count {
            let max_end_index = usize::min(char_count, start_index + max_piece_chars);
            if start_index >= max_end_index {
                continue;
            }

            for end_index in (start_index + 1)..=max_end_index {
                let piece =
                    normalized[char_offsets[start_index]..char_offsets[end_index]].to_string();
                *counts.entry(piece).or_insert(0) += 1;
            }
        }
    }

    let mut protected_candidates = protected_pieces
        .iter()
        .map(|piece| (piece.clone(), counts.get(piece).copied().unwrap_or(0), true))
        .collect::<Vec<_>>();

    let remaining_slots = seed_candidate_limit.saturating_sub(protected_candidates.len());
    if remaining_slots == 0 {
        return protected_candidates;
    }

    let mut ranked_candidates = counts
        .into_iter()
        .filter(|(piece, count)| !protected_pieces.contains(piece) && *count >= min_candidate_freq)
        .map(|(piece, count)| (piece, count, false))
        .collect::<Vec<_>>();
    ranked_candidates
        .sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    protected_candidates.extend(ranked_candidates.into_iter().take(remaining_slots));
    protected_candidates
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        Self::from_path(model_path).map_err(to_py_err)
    }

    #[staticmethod]
    fn from_json(model_json: &str) -> PyResult<Self> {
        Self::from_json_str(model_json).map_err(to_py_err)
    }

    fn normalize(&self, text: &str) -> String {
        normalize_text(text)
    }

    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        encode_ids(text, &self.model, &self.trie).map_err(to_py_err)
    }

    fn encode_pieces(&self, text: &str) -> PyResult<Vec<String>> {
        encode_pieces(text, &self.model, &self.trie).map_err(to_py_err)
    }

    fn encode_normalized(&self, normalized: &str) -> PyResult<Vec<usize>> {
        let segmentation =
            segment_normalized(normalized, &self.model, &self.trie).map_err(to_py_err)?;
        Ok(segmentation.ids)
    }

    fn count_piece_usage_normalized(&self, texts: Vec<String>) -> PyResult<BTreeMap<usize, usize>> {
        let dense_counts =
            count_piece_usage_dense(texts, &self.model, &self.trie).map_err(to_py_err)?;
        let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
        for (token_id, count) in dense_counts.into_iter().enumerate() {
            if count > 0 {
                counts.insert(token_id, count);
            }
        }
        Ok(counts)
    }

    fn count_piece_usage_normalized_dense(&self, texts: Vec<String>) -> PyResult<Vec<usize>> {
        count_piece_usage_dense(texts, &self.model, &self.trie).map_err(to_py_err)
    }

    fn decode(&self, ids: Vec<usize>) -> PyResult<String> {
        decode_ids(&ids, &self.model).map_err(to_py_err)
    }

    fn decode_pieces(&self, pieces: Vec<String>) -> String {
        let borrowed = pieces.iter().map(String::as_str).collect::<Vec<_>>();
        decode_pieces(&borrowed, self.model.normalization.boundary_marker.as_str())
    }
}

#[pyfunction]
fn load_tokenizer(model_path: &str) -> PyResult<PyTokenizer> {
    PyTokenizer::from_path(model_path).map_err(to_py_err)
}

#[pyfunction]
fn enumerate_seed_candidates_normalized(
    texts: Vec<String>,
    max_piece_chars: usize,
    min_candidate_freq: usize,
    seed_candidate_limit: usize,
) -> Vec<(String, usize, bool)> {
    enumerate_seed_candidates(
        texts,
        max_piece_chars,
        min_candidate_freq,
        seed_candidate_limit,
    )
}

#[pymodule]
pub fn bwiza_tokenizer_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(load_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_seed_candidates_normalized, m)?)?;
    Ok(())
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use pyo3::prelude::*;
    use pyo3::types::PyModule;

    fn model_json() -> &'static str {
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
}"#
    }

    fn temp_model_path() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "bwiza-tokenizer-runtime-pyo3-{}.json",
            std::process::id()
        ));
        fs::write(&path, model_json()).expect("temp model should be written");
        path
    }

    #[test]
    fn python_module_exposes_tokenizer_api() {
        Python::with_gil(|py| -> PyResult<()> {
            let path = temp_model_path();
            let module = PyModule::new(py, "bwiza_tokenizer_runtime")?;
            super::bwiza_tokenizer_runtime(&module)?;

            let tokenizer_type = module.getattr("Tokenizer")?;
            let tokenizer = tokenizer_type.call1((path.to_string_lossy().as_ref(),))?;

            let normalized: String = tokenizer
                .call_method1("normalize", ("Muraho neza",))?
                .extract()?;
            assert_eq!(normalized, "▁Muraho▁neza");

            let ids: Vec<usize> = tokenizer
                .call_method1("encode", ("Muraho neza",))?
                .extract()?;
            assert_eq!(ids, vec![7, 8, 9]);

            let pieces: Vec<String> = tokenizer
                .call_method1("encode_pieces", ("Muraho neza",))?
                .extract()?;
            assert_eq!(
                pieces,
                vec!["▁Mur".to_string(), "aho".to_string(), "▁neza".to_string()]
            );

            let decoded: String = tokenizer
                .call_method1("decode", (vec![7usize, 8, 9],))?
                .extract()?;
            assert_eq!(decoded, "Muraho neza");

            let decoded_pieces: String = tokenizer
                .call_method1(
                    "decode_pieces",
                    (vec![
                        "▁Mur".to_string(),
                        "aho".to_string(),
                        "▁neza".to_string(),
                    ],),
                )?
                .extract()?;
            assert_eq!(decoded_pieces, "Muraho neza");

            fs::remove_file(path).ok();
            Ok(())
        })
        .expect("python bindings should behave like the reference API");
    }

    #[test]
    fn python_module_exposes_loader_function() {
        Python::with_gil(|py| -> PyResult<()> {
            let path = temp_model_path();
            let module = PyModule::new(py, "bwiza_tokenizer_runtime")?;
            super::bwiza_tokenizer_runtime(&module)?;

            let tokenizer = module
                .getattr("load_tokenizer")?
                .call1((path.to_string_lossy().as_ref(),))?;
            let ids: Vec<usize> = tokenizer.call_method1("encode", ("Muraho",))?.extract()?;
            assert_eq!(ids, vec![7, 8]);

            fs::remove_file(path).ok();
            Ok(())
        })
        .expect("loader function should return a working tokenizer");
    }

    #[test]
    fn python_module_exposes_json_loader_and_normalized_counting() {
        Python::with_gil(|py| -> PyResult<()> {
            let module = PyModule::new(py, "bwiza_tokenizer_runtime")?;
            super::bwiza_tokenizer_runtime(&module)?;

            let tokenizer_type = module.getattr("Tokenizer")?;
            let tokenizer = tokenizer_type.call_method1("from_json", (model_json(),))?;

            let ids: Vec<usize> = tokenizer
                .call_method1("encode_normalized", ("▁Muraho▁neza",))?
                .extract()?;
            assert_eq!(ids, vec![7, 8, 9]);

            let counts: std::collections::BTreeMap<usize, usize> = tokenizer
                .call_method1(
                    "count_piece_usage_normalized",
                    (vec!["▁Muraho".to_string(), "▁Muraho▁neza".to_string()],),
                )?
                .extract()?;
            assert_eq!(counts.get(&7), Some(&2));
            assert_eq!(counts.get(&8), Some(&2));
            assert_eq!(counts.get(&9), Some(&1));

            let dense_counts: Vec<usize> = tokenizer
                .call_method1(
                    "count_piece_usage_normalized_dense",
                    (vec!["▁Muraho".to_string(), "▁Muraho▁neza".to_string()],),
                )?
                .extract()?;
            assert_eq!(dense_counts[7], 2);
            assert_eq!(dense_counts[8], 2);
            assert_eq!(dense_counts[9], 1);

            let candidates: Vec<(String, usize, bool)> = module
                .getattr("enumerate_seed_candidates_normalized")?
                .call1((
                    vec!["aba".to_string(), "aba".to_string()],
                    2usize,
                    1usize,
                    4usize,
                ))?
                .extract()?;
            assert_eq!(
                candidates,
                vec![
                    ("a".to_string(), 4, true),
                    ("b".to_string(), 2, true),
                    ("ab".to_string(), 2, false),
                    ("ba".to_string(), 2, false),
                ]
            );

            Ok(())
        })
        .expect("json loader and normalized counting should behave like the reference API");
    }
}
