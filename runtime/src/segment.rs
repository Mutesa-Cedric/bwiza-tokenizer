use crate::errors::RuntimeError;
use crate::model::{ModelV1, VocabEntry};
use crate::trie::PieceTrie;

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationResult {
    pub score: f64,
    pub ids: Vec<usize>,
    pub pieces: Vec<String>,
}

fn entry_by_id<'a>(
    entries_by_id: &'a [Option<&'a VocabEntry>],
    token_id: usize,
) -> Result<&'a VocabEntry, RuntimeError> {
    entries_by_id
        .get(token_id)
        .and_then(|entry| *entry)
        .ok_or(RuntimeError::UnknownTokenId(token_id))
}

fn is_better(
    left_id: usize,
    left_next: usize,
    left_score: f64,
    left_token_count: usize,
    right_id: Option<usize>,
    right_next: usize,
    right_score: f64,
    right_token_count: usize,
    entries_by_id: &[Option<&VocabEntry>],
    best_ids: &[Option<usize>],
    best_next_offsets: &[usize],
) -> Result<bool, RuntimeError> {
    let Some(mut current_right_id) = right_id else {
        return Ok(true);
    };

    if left_score != right_score {
        return Ok(left_score > right_score);
    }

    if left_token_count != right_token_count {
        return Ok(left_token_count < right_token_count);
    }

    let mut current_left_id = left_id;
    let mut current_left_next = left_next;
    let mut current_right_next = right_next;

    loop {
        if current_left_id != current_right_id {
            let left_entry = entry_by_id(entries_by_id, current_left_id)?;
            let right_entry = entry_by_id(entries_by_id, current_right_id)?;
            let left_len = left_entry.piece.chars().count();
            let right_len = right_entry.piece.chars().count();

            if left_len != right_len {
                return Ok(left_len > right_len);
            }

            return Ok(current_left_id < current_right_id);
        }

        let Some(next_left_id) = best_ids[current_left_next] else {
            return Ok(false);
        };
        let Some(next_right_id) = best_ids[current_right_next] else {
            return Ok(false);
        };

        current_left_id = next_left_id;
        current_left_next = best_next_offsets[current_left_next];
        current_right_id = next_right_id;
        current_right_next = best_next_offsets[current_right_next];
    }
}

pub fn segment_normalized(
    normalized: &str,
    model: &ModelV1,
    trie: &PieceTrie,
) -> Result<SegmentationResult, RuntimeError> {
    let mut entries_by_id: Vec<Option<&VocabEntry>> = vec![None; model.vocab.len()];
    for entry in &model.vocab {
        entries_by_id[entry.id] = Some(entry);
    }

    let unknown_id = model.special_token_ids.unk;
    let unknown_entry = entry_by_id(&entries_by_id, unknown_id)?;
    let text_length = normalized.len();
    let mut best_scores = vec![f64::NEG_INFINITY; text_length + 1];
    let mut best_token_counts = vec![0usize; text_length + 1];
    let mut best_ids = vec![None; text_length + 1];
    let mut best_next_offsets = vec![text_length; text_length + 1];
    best_scores[text_length] = 0.0;

    for offset in (0..text_length).rev() {
        if !normalized.is_char_boundary(offset) {
            continue;
        }

        let mut best_id = None;
        let mut best_next_offset = text_length;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_token_count = 0usize;
        let candidate_ids = trie.candidate_ids_at(normalized, offset);

        if candidate_ids.is_empty() {
            let next_offset = normalized[offset..]
                .chars()
                .next()
                .map(|ch| offset + ch.len_utf8())
                .unwrap_or(offset);
            best_id = Some(unknown_id);
            best_next_offset = next_offset;
            best_score = unknown_entry.score + best_scores[next_offset];
            best_token_count = 1 + best_token_counts[next_offset];
        } else {
            for token_id in candidate_ids {
                let entry = entry_by_id(&entries_by_id, token_id)?;
                let next_offset = offset + entry.piece.len();
                let candidate_score = entry.score + best_scores[next_offset];
                let candidate_token_count = 1 + best_token_counts[next_offset];

                if is_better(
                    token_id,
                    next_offset,
                    candidate_score,
                    candidate_token_count,
                    best_id,
                    best_next_offset,
                    best_score,
                    best_token_count,
                    &entries_by_id,
                    &best_ids,
                    &best_next_offsets,
                )? {
                    best_id = Some(token_id);
                    best_next_offset = next_offset;
                    best_score = candidate_score;
                    best_token_count = candidate_token_count;
                }
            }
        }

        best_ids[offset] = best_id;
        best_next_offsets[offset] = best_next_offset;
        best_scores[offset] = best_score;
        best_token_counts[offset] = best_token_count;
    }

    let mut ids = Vec::new();
    let mut pieces = Vec::new();
    let mut offset = 0usize;

    while offset < text_length {
        let Some(token_id) = best_ids[offset] else {
            break;
        };
        let entry = entry_by_id(&entries_by_id, token_id)?;
        ids.push(token_id);
        pieces.push(entry.piece.clone());
        let next_offset = best_next_offsets[offset];
        if next_offset <= offset {
            return Err(RuntimeError::Validation(
                "segmentation did not advance; trie is invalid".to_string(),
            ));
        }
        offset = next_offset;
    }

    Ok(SegmentationResult {
        score: best_scores[0],
        ids,
        pieces,
    })
}
