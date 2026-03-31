use std::path::Path;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::decode::{decode_ids, decode_pieces};
use crate::encode::{encode_ids, encode_pieces};
use crate::errors::RuntimeError;
use crate::model::load_model;
use crate::normalize::normalize_text;
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

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        Self::from_path(model_path).map_err(to_py_err)
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

#[pymodule]
pub fn bwiza_tokenizer_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(load_tokenizer, m)?)?;
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

            let normalized: String = tokenizer.call_method1("normalize", ("Muraho neza",))?.extract()?;
            assert_eq!(normalized, "▁Muraho▁neza");

            let ids: Vec<usize> = tokenizer.call_method1("encode", ("Muraho neza",))?.extract()?;
            assert_eq!(ids, vec![7, 8, 9]);

            let pieces: Vec<String> = tokenizer
                .call_method1("encode_pieces", ("Muraho neza",))?
                .extract()?;
            assert_eq!(pieces, vec!["▁Mur".to_string(), "aho".to_string(), "▁neza".to_string()]);

            let decoded: String = tokenizer.call_method1("decode", (vec![7usize, 8, 9],))?.extract()?;
            assert_eq!(decoded, "Muraho neza");

            let decoded_pieces: String = tokenizer
                .call_method1(
                    "decode_pieces",
                    (vec!["▁Mur".to_string(), "aho".to_string(), "▁neza".to_string()],),
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
}
