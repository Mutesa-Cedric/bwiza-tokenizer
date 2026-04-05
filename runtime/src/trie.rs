use std::collections::BTreeMap;

use crate::byte_fallback::{BYTE_FALLBACK_COUNT, parse_byte_fallback_piece};
use crate::model::ModelV1;

#[derive(Debug, Default)]
struct TrieNode {
    children: BTreeMap<u8, usize>,
    terminal_ids: Vec<usize>,
}

#[derive(Debug, Default)]
pub struct PieceTrie {
    nodes: Vec<TrieNode>,
    byte_fallback_ids: Option<[usize; BYTE_FALLBACK_COUNT]>,
}

impl PieceTrie {
    pub fn from_model(model: &ModelV1) -> Self {
        let mut trie = Self {
            nodes: vec![TrieNode::default()],
            byte_fallback_ids: None,
        };
        let mut byte_fallback_ids = [0usize; BYTE_FALLBACK_COUNT];
        let mut has_byte_fallback = false;

        for entry in &model.vocab {
            if entry.special.is_some() {
                continue;
            }

            if let Some(byte_value) = parse_byte_fallback_piece(entry.piece.as_str()) {
                byte_fallback_ids[byte_value as usize] = entry.id;
                has_byte_fallback = true;
                continue;
            }

            trie.insert(entry.id, entry.piece.as_bytes());
        }

        for node in &mut trie.nodes {
            node.terminal_ids.sort_unstable();
        }

        if has_byte_fallback {
            trie.byte_fallback_ids = Some(byte_fallback_ids);
        }

        trie
    }

    fn insert(&mut self, token_id: usize, piece: &[u8]) {
        let mut node_index = 0;

        for byte in piece {
            let next_index = if let Some(child_index) = self.nodes[node_index].children.get(byte) {
                *child_index
            } else {
                let new_index = self.nodes.len();
                self.nodes.push(TrieNode::default());
                self.nodes[node_index].children.insert(*byte, new_index);
                new_index
            };

            node_index = next_index;
        }

        self.nodes[node_index].terminal_ids.push(token_id);
    }

    pub fn candidate_ids_at(&self, text: &str, offset: usize) -> Vec<usize> {
        if !text.is_char_boundary(offset) {
            return Vec::new();
        }

        let bytes = text.as_bytes();
        let mut node_index = 0;
        let mut index = offset;
        let mut matches = Vec::new();

        while index < bytes.len() {
            let byte = bytes[index];
            let Some(next_index) = self.nodes[node_index].children.get(&byte).copied() else {
                break;
            };

            node_index = next_index;
            index += 1;
            matches.extend(self.nodes[node_index].terminal_ids.iter().copied());
        }

        matches
    }

    pub fn byte_fallback_ids(&self) -> Option<&[usize; BYTE_FALLBACK_COUNT]> {
        self.byte_fallback_ids.as_ref()
    }
}
