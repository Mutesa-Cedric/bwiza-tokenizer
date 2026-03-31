use std::collections::BTreeMap;

use crate::errors::RuntimeError;
use crate::model::{ModelV1, VocabEntry};
use crate::trie::PieceTrie;

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationResult {
    pub score: f64,
    pub ids: Vec<usize>,
    pub pieces: Vec<String>,
}

#[derive(Debug, Clone)]
struct CandidatePath {
    score: f64,
    ids: Vec<usize>,
    pieces: Vec<String>,
}

pub fn segment_normalized(
    normalized: &str,
    model: &ModelV1,
    trie: &PieceTrie,
) -> Result<SegmentationResult, RuntimeError> {
    let entries_by_id: BTreeMap<usize, &VocabEntry> =
        model.vocab.iter().map(|entry| (entry.id, entry)).collect();
    let unknown_id = model.special_token_ids.unk;
    let unknown_entry = entries_by_id
        .get(&unknown_id)
        .ok_or_else(|| RuntimeError::Validation("model is missing the required <unk> entry".to_string()))?;

    let mut memo: Vec<Option<CandidatePath>> = vec![None; normalized.len() + 1];

    fn best_from(
        offset: usize,
        normalized: &str,
        trie: &PieceTrie,
        entries_by_id: &BTreeMap<usize, &VocabEntry>,
        unknown_entry: &VocabEntry,
        memo: &mut Vec<Option<CandidatePath>>,
    ) -> Result<CandidatePath, RuntimeError> {
        if let Some(result) = &memo[offset] {
            return Ok(result.clone());
        }

        if offset >= normalized.len() {
            let result = CandidatePath {
                score: 0.0,
                ids: Vec::new(),
                pieces: Vec::new(),
            };
            memo[offset] = Some(result.clone());
            return Ok(result);
        }

        let candidate_ids = trie.candidate_ids_at(normalized, offset);
        let result = if candidate_ids.is_empty() {
            let next_offset = normalized[offset..]
                .chars()
                .next()
                .map(|ch| offset + ch.len_utf8())
                .unwrap_or(offset);
            let suffix = best_from(
                next_offset,
                normalized,
                trie,
                entries_by_id,
                unknown_entry,
                memo,
            )?;

            let mut ids = Vec::with_capacity(suffix.ids.len() + 1);
            ids.push(unknown_entry.id);
            ids.extend(suffix.ids.iter().copied());

            let mut pieces = Vec::with_capacity(suffix.pieces.len() + 1);
            pieces.push(unknown_entry.piece.clone());
            pieces.extend(suffix.pieces.iter().cloned());

            CandidatePath {
                score: unknown_entry.score + suffix.score,
                ids,
                pieces,
            }
        } else {
            let mut best: Option<CandidatePath> = None;

            for token_id in candidate_ids {
                let entry = entries_by_id
                    .get(&token_id)
                    .copied()
                    .ok_or(RuntimeError::UnknownTokenId(token_id))?;
                let suffix = best_from(
                    offset + entry.piece.len(),
                    normalized,
                    trie,
                    entries_by_id,
                    unknown_entry,
                    memo,
                )?;

                let mut ids = Vec::with_capacity(suffix.ids.len() + 1);
                ids.push(entry.id);
                ids.extend(suffix.ids.iter().copied());

                let mut pieces = Vec::with_capacity(suffix.pieces.len() + 1);
                pieces.push(entry.piece.clone());
                pieces.extend(suffix.pieces.iter().cloned());

                let candidate = CandidatePath {
                    score: entry.score + suffix.score,
                    ids,
                    pieces,
                };

                if best
                    .as_ref()
                    .map(|current| is_better(&candidate, current))
                    .unwrap_or(true)
                {
                    best = Some(candidate);
                }
            }

            best.expect("candidate set was not empty")
        };

        memo[offset] = Some(result.clone());
        Ok(result)
    }

    let best = best_from(
        0,
        normalized,
        trie,
        &entries_by_id,
        unknown_entry,
        &mut memo,
    )?;

    Ok(SegmentationResult {
        score: best.score,
        ids: best.ids,
        pieces: best.pieces,
    })
}

fn is_better(left: &CandidatePath, right: &CandidatePath) -> bool {
    if left.score != right.score {
        return left.score > right.score;
    }

    if left.ids.len() != right.ids.len() {
        return left.ids.len() < right.ids.len();
    }

    for ((left_id, left_piece), (right_id, right_piece)) in left
        .ids
        .iter()
        .zip(left.pieces.iter())
        .zip(right.ids.iter().zip(right.pieces.iter()))
    {
        if left_id == right_id {
            continue;
        }

        let left_len = left_piece.chars().count();
        let right_len = right_piece.chars().count();
        if left_len != right_len {
            return left_len > right_len;
        }

        return left_id < right_id;
    }

    false
}
