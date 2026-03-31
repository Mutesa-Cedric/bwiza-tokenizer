use crate::errors::RuntimeError;
use crate::model::ModelV1;
use crate::normalize::normalize_text;
use crate::segment::segment_normalized;
use crate::trie::PieceTrie;

pub fn encode_ids(
    text: &str,
    model: &ModelV1,
    trie: &PieceTrie,
) -> Result<Vec<usize>, RuntimeError> {
    let normalized = normalize_text(text);
    let segmentation = segment_normalized(normalized.as_str(), model, trie)?;
    Ok(segmentation.ids)
}

pub fn encode_pieces(
    text: &str,
    model: &ModelV1,
    trie: &PieceTrie,
) -> Result<Vec<String>, RuntimeError> {
    let normalized = normalize_text(text);
    let segmentation = segment_normalized(normalized.as_str(), model, trie)?;
    Ok(segmentation.pieces)
}
