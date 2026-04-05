use crate::errors::RuntimeError;
use crate::model::{ModelV1, VocabEntry};
use crate::trie::PieceTrie;

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationResult {
    pub score: f64,
    pub ids: Vec<usize>,
    pub pieces: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepChoice {
    Token(usize),
    ByteFallback { ids: [usize; 4], len: usize },
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

fn build_entries_by_id(model: &ModelV1) -> Vec<Option<&VocabEntry>> {
    let mut entries_by_id = vec![None; model.vocab.len()];
    for entry in &model.vocab {
        entries_by_id[entry.id] = Some(entry);
    }

    entries_by_id
}

fn build_piece_scores(model: &ModelV1) -> Vec<f64> {
    let mut scores = vec![0.0; model.vocab.len()];
    for entry in &model.vocab {
        scores[entry.id] = entry.score;
    }

    scores
}

fn build_piece_byte_lengths(model: &ModelV1) -> Vec<usize> {
    let mut lengths = vec![0usize; model.vocab.len()];
    for entry in &model.vocab {
        lengths[entry.id] = entry.piece.len();
    }

    lengths
}

fn build_piece_surface_lengths(model: &ModelV1) -> Vec<usize> {
    let mut lengths = vec![0usize; model.vocab.len()];
    for entry in &model.vocab {
        lengths[entry.id] = entry.piece.chars().count();
    }

    lengths
}

fn is_better(
    left_step: &StepChoice,
    left_next: usize,
    left_score: f64,
    left_token_count: usize,
    right_step: Option<&StepChoice>,
    right_next: usize,
    right_score: f64,
    right_token_count: usize,
    piece_surface_lengths: &[usize],
    best_steps: &[Option<StepChoice>],
    best_next_offsets: &[usize],
) -> Result<bool, RuntimeError> {
    let Some(mut current_right_step) = right_step.copied() else {
        return Ok(true);
    };

    if left_score != right_score {
        return Ok(left_score > right_score);
    }

    if left_token_count != right_token_count {
        return Ok(left_token_count < right_token_count);
    }

    let mut current_left_step = *left_step;
    let mut current_left_next = left_next;
    let mut current_right_next = right_next;

    loop {
        if current_left_step != current_right_step {
            let left_len = step_surface_length(&current_left_step, piece_surface_lengths);
            let right_len = step_surface_length(&current_right_step, piece_surface_lengths);

            if left_len != right_len {
                return Ok(left_len > right_len);
            }

            return Ok(step_ids(&current_left_step) < step_ids(&current_right_step));
        }

        let Some(next_left_step) = best_steps[current_left_next].as_ref() else {
            return Ok(false);
        };
        let Some(next_right_step) = best_steps[current_right_next].as_ref() else {
            return Ok(false);
        };

        current_left_step = *next_left_step;
        current_left_next = best_next_offsets[current_left_next];
        current_right_step = *next_right_step;
        current_right_next = best_next_offsets[current_right_next];
    }
}

fn step_ids(step: &StepChoice) -> &[usize] {
    match step {
        StepChoice::Token(token_id) => std::slice::from_ref(token_id),
        StepChoice::ByteFallback { ids, len } => &ids[..*len],
    }
}

fn step_token_count(step: &StepChoice) -> usize {
    step_ids(step).len()
}

fn step_surface_length(step: &StepChoice, piece_surface_lengths: &[usize]) -> usize {
    match step {
        StepChoice::Token(token_id) => piece_surface_lengths[*token_id],
        StepChoice::ByteFallback { .. } => 1,
    }
}

fn byte_fallback_step(ch: char, byte_fallback_ids: &[usize; 256]) -> StepChoice {
    let mut ids = [0usize; 4];
    let mut len = 0usize;

    for byte in ch.to_string().into_bytes() {
        ids[len] = byte_fallback_ids[byte as usize];
        len += 1;
    }

    StepChoice::ByteFallback { ids, len }
}

fn segment_state<'a>(
    normalized: &str,
    model: &'a ModelV1,
    trie: &PieceTrie,
    piece_scores: &[f64],
    piece_byte_lengths: &[usize],
    piece_surface_lengths: &[usize],
) -> Result<(Vec<f64>, Vec<Option<StepChoice>>, Vec<usize>), RuntimeError> {
    let unknown_id = model.special_token_ids.unk;
    let text_length = normalized.len();
    let mut best_scores = vec![f64::NEG_INFINITY; text_length + 1];
    let mut best_token_counts = vec![0usize; text_length + 1];
    let mut best_steps = vec![None; text_length + 1];
    let mut best_next_offsets = vec![text_length; text_length + 1];
    best_scores[text_length] = 0.0;
    let byte_fallback_ids = trie.byte_fallback_ids();

    for offset in (0..text_length).rev() {
        if !normalized.is_char_boundary(offset) {
            continue;
        }

        let mut best_step = None;
        let mut best_next_offset = text_length;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_token_count = 0usize;
        let mut saw_candidate = false;

        trie.for_each_candidate_id_at(normalized, offset, |token_id| {
            saw_candidate = true;
            let step = StepChoice::Token(token_id);
            let next_offset = offset + piece_byte_lengths[token_id];
            let candidate_score = piece_scores[token_id] + best_scores[next_offset];
            let candidate_token_count = 1 + best_token_counts[next_offset];

            if is_better(
                &step,
                next_offset,
                candidate_score,
                candidate_token_count,
                best_step.as_ref(),
                best_next_offset,
                best_score,
                best_token_count,
                piece_surface_lengths,
                &best_steps,
                &best_next_offsets,
            )
            .expect("candidate ids must be valid")
            {
                best_step = Some(step);
                best_next_offset = next_offset;
                best_score = candidate_score;
                best_token_count = candidate_token_count;
            }
        });

        if !saw_candidate {
            let ch = normalized[offset..]
                .chars()
                .next()
                .ok_or(RuntimeError::Validation(
                    "segmentation reached an invalid offset".to_string(),
                ))?;
            let next_offset = offset + ch.len_utf8();
            let fallback_step = if let Some(byte_ids) = byte_fallback_ids {
                byte_fallback_step(ch, byte_ids)
            } else {
                StepChoice::Token(unknown_id)
            };
            best_next_offset = next_offset;
            best_score = match fallback_step {
                StepChoice::Token(token_id) => piece_scores[token_id],
                StepChoice::ByteFallback { ids, len } => ids[..len]
                    .iter()
                    .map(|token_id| piece_scores[*token_id])
                    .sum(),
            } + best_scores[next_offset];
            best_token_count = step_token_count(&fallback_step) + best_token_counts[next_offset];
            best_step = Some(fallback_step);
        }

        best_steps[offset] = best_step;
        best_next_offsets[offset] = best_next_offset;
        best_scores[offset] = best_score;
        best_token_counts[offset] = best_token_count;
    }

    Ok((best_scores, best_steps, best_next_offsets))
}

pub fn count_segmented_ids_normalized(
    normalized: &str,
    model: &ModelV1,
    trie: &PieceTrie,
    piece_scores: &[f64],
    piece_byte_lengths: &[usize],
    piece_surface_lengths: &[usize],
    counts: &mut [usize],
) -> Result<f64, RuntimeError> {
    let (best_scores, best_steps, best_next_offsets) = segment_state(
        normalized,
        model,
        trie,
        piece_scores,
        piece_byte_lengths,
        piece_surface_lengths,
    )?;
    let text_length = normalized.len();
    let mut offset = 0usize;

    while offset < text_length {
        let Some(step) = best_steps[offset].as_ref() else {
            break;
        };
        for token_id in step_ids(step) {
            counts[*token_id] += 1;
        }
        let next_offset = best_next_offsets[offset];
        if next_offset <= offset {
            return Err(RuntimeError::Validation(
                "segmentation did not advance; trie is invalid".to_string(),
            ));
        }
        offset = next_offset;
    }

    Ok(best_scores[0])
}

pub fn segment_normalized(
    normalized: &str,
    model: &ModelV1,
    trie: &PieceTrie,
) -> Result<SegmentationResult, RuntimeError> {
    let entries_by_id = build_entries_by_id(model);
    let piece_scores = build_piece_scores(model);
    let piece_byte_lengths = build_piece_byte_lengths(model);
    let piece_surface_lengths = build_piece_surface_lengths(model);
    let (best_scores, best_steps, best_next_offsets) = segment_state(
        normalized,
        model,
        trie,
        &piece_scores,
        &piece_byte_lengths,
        &piece_surface_lengths,
    )?;
    let text_length = normalized.len();
    let mut ids = Vec::new();
    let mut pieces = Vec::new();
    let mut offset = 0usize;

    while offset < text_length {
        let Some(step) = best_steps[offset].as_ref() else {
            break;
        };
        for token_id in step_ids(step) {
            let entry = entry_by_id(&entries_by_id, *token_id)?;
            ids.push(*token_id);
            pieces.push(entry.piece.clone());
        }
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
