use crate::errors::RuntimeError;
use crate::model::{ModelV1, VocabEntry};
use crate::trie::PieceTrie;

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationResult {
    pub score: f64,
    pub ids: Vec<usize>,
    pub pieces: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

fn is_better(
    left_step: &StepChoice,
    left_next: usize,
    left_score: f64,
    left_token_count: usize,
    right_step: Option<&StepChoice>,
    right_next: usize,
    right_score: f64,
    right_token_count: usize,
    entries_by_id: &[Option<&VocabEntry>],
    best_steps: &[Option<StepChoice>],
    best_next_offsets: &[usize],
) -> Result<bool, RuntimeError> {
    let Some(mut current_right_step) = right_step.cloned() else {
        return Ok(true);
    };

    if left_score != right_score {
        return Ok(left_score > right_score);
    }

    if left_token_count != right_token_count {
        return Ok(left_token_count < right_token_count);
    }

    let mut current_left_step = left_step.clone();
    let mut current_left_next = left_next;
    let mut current_right_next = right_next;

    loop {
        if current_left_step != current_right_step {
            let left_len = step_surface_length(&current_left_step, entries_by_id)?;
            let right_len = step_surface_length(&current_right_step, entries_by_id)?;

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

        current_left_step = next_left_step.clone();
        current_left_next = best_next_offsets[current_left_next];
        current_right_step = next_right_step.clone();
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

fn step_surface_length(
    step: &StepChoice,
    entries_by_id: &[Option<&VocabEntry>],
) -> Result<usize, RuntimeError> {
    match step {
        StepChoice::Token(token_id) => {
            Ok(entry_by_id(entries_by_id, *token_id)?.piece.chars().count())
        }
        StepChoice::ByteFallback { .. } => Ok(1),
    }
}

fn step_score(
    step: &StepChoice,
    entries_by_id: &[Option<&VocabEntry>],
) -> Result<f64, RuntimeError> {
    match step {
        StepChoice::Token(token_id) => Ok(entry_by_id(entries_by_id, *token_id)?.score),
        StepChoice::ByteFallback { ids, len } => {
            ids[..*len].iter().try_fold(0.0, |score, token_id| {
                Ok(score + entry_by_id(entries_by_id, *token_id)?.score)
            })
        }
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
        let candidate_ids = trie.candidate_ids_at(normalized, offset);

        if candidate_ids.is_empty() {
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
            best_score = step_score(&fallback_step, &entries_by_id)? + best_scores[next_offset];
            best_token_count = step_token_count(&fallback_step) + best_token_counts[next_offset];
            best_step = Some(fallback_step);
        } else {
            for token_id in candidate_ids {
                let entry = entry_by_id(&entries_by_id, token_id)?;
                let step = StepChoice::Token(token_id);
                let next_offset = offset + entry.piece.len();
                let candidate_score = step_score(&step, &entries_by_id)? + best_scores[next_offset];
                let candidate_token_count =
                    step_token_count(&step) + best_token_counts[next_offset];

                if is_better(
                    &step,
                    next_offset,
                    candidate_score,
                    candidate_token_count,
                    best_step.as_ref(),
                    best_next_offset,
                    best_score,
                    best_token_count,
                    &entries_by_id,
                    &best_steps,
                    &best_next_offsets,
                )? {
                    best_step = Some(step);
                    best_next_offset = next_offset;
                    best_score = candidate_score;
                    best_token_count = candidate_token_count;
                }
            }
        }

        best_steps[offset] = best_step;
        best_next_offsets[offset] = best_next_offset;
        best_scores[offset] = best_score;
        best_token_counts[offset] = best_token_count;
    }

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
