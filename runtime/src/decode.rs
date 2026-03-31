use crate::errors::RuntimeError;
use crate::model::ModelV1;

pub fn decode_pieces(pieces: &[&str], boundary_marker: &str) -> String {
    let merged = pieces.join("").replace(boundary_marker, " ");
    if let Some(stripped) = merged.strip_prefix(' ') {
        return stripped.to_string();
    }

    merged
}

pub fn decode_ids(ids: &[usize], model: &ModelV1) -> Result<String, RuntimeError> {
    let pieces = ids
        .iter()
        .map(|token_id| {
            model
                .vocab
                .iter()
                .find(|entry| entry.id == *token_id)
                .map(|entry| entry.piece.as_str())
                .ok_or(RuntimeError::UnknownTokenId(*token_id))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(decode_pieces(
        &pieces,
        model.normalization.boundary_marker.as_str(),
    ))
}
