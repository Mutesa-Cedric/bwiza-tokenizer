from __future__ import annotations

from dataclasses import dataclass, field

from ..byte_fallback import is_byte_fallback_piece, parse_byte_fallback_piece
from ..types import ModelV1, VocabEntry


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    score: float
    ids: tuple[int, ...]
    pieces: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PathStep:
    score: float
    ids: tuple[int, ...]
    pieces: tuple[str, ...]
    surface_length: int


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
            if entry.special is not None or is_byte_fallback_piece(entry.piece):
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

    def iter_candidate_entries_at(self, text: str, offset: int):
        if offset < 0 or offset >= len(text):
            return

        node_index = 0

        for char_index in range(offset, len(text)):
            char = text[char_index]
            next_index = self.nodes[node_index].children.get(char)
            if next_index is None:
                break
            node_index = next_index
            terminal_entries = self.nodes[node_index].terminal_entries
            if terminal_entries:
                yield from terminal_entries


@dataclass(frozen=True, slots=True)
class ModelIndex:
    trie: PieceTrie
    unknown_entry: VocabEntry
    byte_fallback_entries: list[VocabEntry | None] | None


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
        byte_fallback_entries=_byte_fallback_entries(model),
    )


def _byte_fallback_entries(model: ModelV1) -> list[VocabEntry | None] | None:
    entries: list[VocabEntry | None] = [None] * 256
    has_byte_fallback = False

    for entry in model.vocab:
        byte_value = parse_byte_fallback_piece(entry.piece)
        if byte_value is None:
            continue
        entries[byte_value] = entry
        has_byte_fallback = True

    if not has_byte_fallback:
        return None

    return entries


def _normal_step(entry: VocabEntry) -> _PathStep:
    return _PathStep(
        score=entry.score,
        ids=(entry.id,),
        pieces=(entry.piece,),
        surface_length=len(entry.piece),
    )


def _byte_fallback_step(
    char: str,
    byte_entries: list[VocabEntry | None],
) -> _PathStep | None:
    ids: list[int] = []
    pieces: list[str] = []
    score = 0.0

    for byte_value in char.encode("utf-8"):
        entry = byte_entries[byte_value]
        if entry is None:
            return None
        ids.append(entry.id)
        pieces.append(entry.piece)
        score += entry.score

    return _PathStep(
        score=score,
        ids=tuple(ids),
        pieces=tuple(pieces),
        surface_length=1,
    )


def _path_is_better(
    *,
    left_step: _PathStep,
    left_next: int,
    left_score: float,
    left_token_count: int,
    right_step: _PathStep | None,
    right_next: int,
    right_score: float,
    right_token_count: int,
    best_steps: list[_PathStep | None],
    best_next_offsets: list[int],
) -> bool:
    if right_step is None:
        return True

    if left_score != right_score:
        return left_score > right_score

    if left_token_count != right_token_count:
        return left_token_count < right_token_count

    current_left_step = left_step
    current_left_next = left_next
    current_right_step = right_step
    current_right_next = right_next

    while True:
        if (
            current_left_step.surface_length != current_right_step.surface_length
            or current_left_step.ids != current_right_step.ids
        ):
            if current_left_step.surface_length != current_right_step.surface_length:
                return current_left_step.surface_length > current_right_step.surface_length

            return current_left_step.ids < current_right_step.ids

        next_left_step = best_steps[current_left_next]
        next_right_step = best_steps[current_right_next]

        if next_left_step is None or next_right_step is None:
            return False

        current_left_step = next_left_step
        current_left_next = best_next_offsets[current_left_next]
        current_right_step = next_right_step
        current_right_next = best_next_offsets[current_right_next]


def _segment_state(
    normalized: str,
    model: ModelV1,
    model_index: ModelIndex | None = None,
) -> tuple[float, list[_PathStep | None], list[int]]:
    if not isinstance(normalized, str):
        raise TypeError("segment_normalized expects a normalized string")

    if not normalized:
        return 0.0, [], []

    resolved_index = model_index or build_model_index(model)
    unknown = resolved_index.unknown_entry
    text_length = len(normalized)
    best_scores = [float("-inf")] * (text_length + 1)
    best_token_counts = [0] * (text_length + 1)
    best_steps: list[_PathStep | None] = [None] * (text_length + 1)
    best_next_offsets = [text_length] * (text_length + 1)
    best_scores[text_length] = 0.0

    for offset in range(text_length - 1, -1, -1):
        best_step: _PathStep | None = None
        best_next_offset = text_length
        best_score = float("-inf")
        best_token_count = 0
        saw_candidate = False

        for entry in resolved_index.trie.iter_candidate_entries_at(normalized, offset):
            saw_candidate = True
            step = _normal_step(entry)
            next_offset = offset + len(entry.piece)
            candidate_score = step.score + best_scores[next_offset]
            candidate_token_count = len(step.ids) + best_token_counts[next_offset]
            if _path_is_better(
                left_step=step,
                left_next=next_offset,
                left_score=candidate_score,
                left_token_count=candidate_token_count,
                right_step=best_step,
                right_next=best_next_offset,
                right_score=best_score,
                right_token_count=best_token_count,
                best_steps=best_steps,
                best_next_offsets=best_next_offsets,
            ):
                best_step = step
                best_next_offset = next_offset
                best_score = candidate_score
                best_token_count = candidate_token_count

        if not saw_candidate:
            next_offset = offset + 1
            byte_fallback = resolved_index.byte_fallback_entries
            fallback_step = (
                _byte_fallback_step(normalized[offset], byte_fallback)
                if byte_fallback is not None
                else None
            )
            if fallback_step is None:
                fallback_step = _normal_step(unknown)
            best_step = fallback_step
            best_next_offset = next_offset
            best_score = fallback_step.score + best_scores[next_offset]
            best_token_count = len(fallback_step.ids) + best_token_counts[next_offset]

        best_steps[offset] = best_step
        best_next_offsets[offset] = best_next_offset
        best_scores[offset] = best_score
        best_token_counts[offset] = best_token_count

    return best_scores[0], best_steps, best_next_offsets


def _iter_best_path_steps(
    normalized: str,
    best_steps: list[_PathStep | None],
    best_next_offsets: list[int],
):
    text_length = len(normalized)
    offset = 0

    while offset < text_length:
        step = best_steps[offset]
        if step is None:
            break
        yield step
        next_offset = best_next_offsets[offset]
        if next_offset <= offset:
            raise ValueError("segmentation did not advance; model index is invalid")
        offset = next_offset


def iter_segment_ids(
    normalized: str,
    model: ModelV1,
    model_index: ModelIndex | None = None,
):
    _, best_steps, best_next_offsets = _segment_state(normalized, model, model_index)

    for step in _iter_best_path_steps(normalized, best_steps, best_next_offsets):
        yield from step.ids


def segment_normalized(
    normalized: str,
    model: ModelV1,
    model_index: ModelIndex | None = None,
) -> SegmentationResult:
    score, best_steps, best_next_offsets = _segment_state(normalized, model, model_index)

    pieces: list[str] = []
    ids: list[int] = []

    for step in _iter_best_path_steps(normalized, best_steps, best_next_offsets):
        pieces.extend(step.pieces)
        ids.extend(step.ids)

    return SegmentationResult(
        score=score,
        ids=tuple(ids),
        pieces=tuple(pieces),
    )
