from __future__ import annotations

from dataclasses import dataclass, field

from ..types import ModelV1, VocabEntry


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    score: float
    ids: tuple[int, ...]
    pieces: tuple[str, ...]


@dataclass(slots=True)
class _TrieNode:
    children: dict[str, int] = field(default_factory=dict)
    terminal_entries: list[VocabEntry] = field(default_factory=list)


@dataclass(slots=True)
class PieceTrie:
    nodes: list[_TrieNode] = field(default_factory=lambda: [_TrieNode()])

    @classmethod
    def from_model(cls, model: ModelV1) -> PieceTrie:
        trie = cls()

        for entry in model.vocab:
            if entry.special is not None:
                continue
            trie.insert(entry)

        for node in trie.nodes:
            node.terminal_entries.sort(key=lambda entry: entry.id)

        return trie

    def insert(self, entry: VocabEntry) -> None:
        node_index = 0

        for char in entry.piece:
            next_index = self.nodes[node_index].children.get(char)
            if next_index is None:
                next_index = len(self.nodes)
                self.nodes.append(_TrieNode())
                self.nodes[node_index].children[char] = next_index
            node_index = next_index

        self.nodes[node_index].terminal_entries.append(entry)

    def candidate_entries_at(self, text: str, offset: int) -> list[VocabEntry]:
        if offset < 0 or offset >= len(text):
            return []

        node_index = 0
        matches: list[VocabEntry] = []

        for char in text[offset:]:
            next_index = self.nodes[node_index].children.get(char)
            if next_index is None:
                break
            node_index = next_index
            matches.extend(self.nodes[node_index].terminal_entries)

        return matches


@dataclass(frozen=True, slots=True)
class ModelIndex:
    trie: PieceTrie
    unknown_entry: VocabEntry


def _unknown_entry(model: ModelV1) -> VocabEntry:
    unknown_id = model.special_token_ids["unk"]

    for entry in model.vocab:
        if entry.id == unknown_id:
            return entry

    raise ValueError("model is missing the required <unk> entry")


def build_model_index(model: ModelV1) -> ModelIndex:
    return ModelIndex(
        trie=PieceTrie.from_model(model),
        unknown_entry=_unknown_entry(model),
    )


def _match_candidates(
    normalized: str,
    offset: int,
    model_index: ModelIndex,
) -> list[VocabEntry]:
    return model_index.trie.candidate_entries_at(normalized, offset)


def _path_is_better(
    *,
    left_entry: VocabEntry,
    left_next: int,
    left_score: float,
    left_token_count: int,
    right_entry: VocabEntry | None,
    right_next: int,
    right_score: float,
    right_token_count: int,
    best_entries: list[VocabEntry | None],
    best_next_offsets: list[int],
) -> bool:
    if right_entry is None:
        return True

    if left_score != right_score:
        return left_score > right_score

    if left_token_count != right_token_count:
        return left_token_count < right_token_count

    current_left_entry = left_entry
    current_left_next = left_next
    current_right_entry = right_entry
    current_right_next = right_next

    while True:
        if current_left_entry.id != current_right_entry.id:
            if len(current_left_entry.piece) != len(current_right_entry.piece):
                return len(current_left_entry.piece) > len(current_right_entry.piece)

            return current_left_entry.id < current_right_entry.id

        next_left_entry = best_entries[current_left_next]
        next_right_entry = best_entries[current_right_next]

        if next_left_entry is None or next_right_entry is None:
            return False

        current_left_entry = next_left_entry
        current_left_next = best_next_offsets[current_left_next]
        current_right_entry = next_right_entry
        current_right_next = best_next_offsets[current_right_next]


def segment_normalized(
    normalized: str,
    model: ModelV1,
    model_index: ModelIndex | None = None,
) -> SegmentationResult:
    if not isinstance(normalized, str):
        raise TypeError("segment_normalized expects a normalized string")

    if not normalized:
        return SegmentationResult(score=0.0, ids=(), pieces=())

    resolved_index = model_index or build_model_index(model)
    unknown = resolved_index.unknown_entry
    text_length = len(normalized)
    best_scores = [float("-inf")] * (text_length + 1)
    best_token_counts = [0] * (text_length + 1)
    best_entries: list[VocabEntry | None] = [None] * (text_length + 1)
    best_next_offsets = [text_length] * (text_length + 1)
    best_scores[text_length] = 0.0

    for offset in range(text_length - 1, -1, -1):
        best_entry: VocabEntry | None = None
        best_next_offset = text_length
        best_score = float("-inf")
        best_token_count = 0
        candidates = _match_candidates(normalized, offset, resolved_index)

        if not candidates:
            next_offset = offset + 1
            best_entry = unknown
            best_next_offset = next_offset
            best_score = unknown.score + best_scores[next_offset]
            best_token_count = 1 + best_token_counts[next_offset]
        else:
            for entry in candidates:
                next_offset = offset + len(entry.piece)
                candidate_score = entry.score + best_scores[next_offset]
                candidate_token_count = 1 + best_token_counts[next_offset]
                if _path_is_better(
                    left_entry=entry,
                    left_next=next_offset,
                    left_score=candidate_score,
                    left_token_count=candidate_token_count,
                    right_entry=best_entry,
                    right_next=best_next_offset,
                    right_score=best_score,
                    right_token_count=best_token_count,
                    best_entries=best_entries,
                    best_next_offsets=best_next_offsets,
                ):
                    best_entry = entry
                    best_next_offset = next_offset
                    best_score = candidate_score
                    best_token_count = candidate_token_count

        best_entries[offset] = best_entry
        best_next_offsets[offset] = best_next_offset
        best_scores[offset] = best_score
        best_token_counts[offset] = best_token_count

    pieces: list[str] = []
    ids: list[int] = []
    offset = 0
    while offset < text_length:
        entry = best_entries[offset]
        if entry is None:
            break
        pieces.append(entry.piece)
        ids.append(entry.id)
        next_offset = best_next_offsets[offset]
        if next_offset <= offset:
            raise ValueError("segmentation did not advance; model index is invalid")
        offset = next_offset

    return SegmentationResult(
        score=best_scores[0],
        ids=tuple(ids),
        pieces=tuple(pieces),
    )
