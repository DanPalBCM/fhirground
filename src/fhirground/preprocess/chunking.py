"""Chunking with exact offset bookkeeping.

The invariant this module exists to guarantee:

    original_text[chunk.start_offset + i] == chunk.text[i]    for all valid i

Every evidence offset fhirground emits traces back to the original note through
that identity. Chunk text is therefore a **verbatim slice** -- never stripped,
normalized, or re-joined. A single stripped character would shift every offset
downstream of it, and the resulting FHIR would point at the wrong words while
looking perfectly valid.

A property test asserts the identity over generated text and chunk parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from fhirground.exceptions import ConfigError
from fhirground.preprocess.sections import Section, detect_sections, section_at

#: Sentence-ish boundary: punctuation then whitespace, or a line break. Used to
#: prefer splitting between sentences rather than mid-word.
_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+|\n+")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A verbatim slice of the original note text.

    Attributes:
        chunk_id: Stable identifier, ``<note_id>#<index>``.
        index: Position in the note, 0-based.
        text: The slice itself, byte-for-byte from the original.
        start_offset: Where the slice starts in the original text.
        end_offset: Where it ends (exclusive).
        section: Name of the section the slice starts in, if any.
    """

    chunk_id: str
    index: int
    text: str
    start_offset: int
    end_offset: int
    section: str | None = None

    def __len__(self) -> int:
        return len(self.text)

    def to_original(self, local_offset: int) -> int:
        """Map an offset inside this chunk to the original text.

        Raises:
            ValueError: If the offset is outside the chunk, which would
                otherwise produce a plausible-looking but wrong global offset.
        """
        if not 0 <= local_offset <= len(self.text):
            raise ValueError(
                f"local offset {local_offset} is outside chunk {self.chunk_id} "
                f"of length {len(self.text)}"
            )
        return self.start_offset + local_offset

    def verify_against(self, original: str) -> None:
        """Assert this chunk is a verbatim slice of ``original``.

        Raises:
            ValueError: If the slice does not match, which means offsets are
                corrupt and nothing downstream can be trusted.
        """
        actual = original[self.start_offset : self.end_offset]
        if actual != self.text:
            raise ValueError(
                f"chunk {self.chunk_id} is not a verbatim slice of the note: "
                f"offsets [{self.start_offset}:{self.end_offset}] give "
                f"{len(actual)} chars but the chunk holds {len(self.text)}"
            )


@dataclass
class ChunkPlan:
    """The chunks for one note, plus the sections they were cut against."""

    chunks: list[Chunk] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    def coverage(self, text_length: int) -> float:
        """Fraction of the note covered by at least one chunk.

        Should be 1.0. Anything less means text was dropped, which would make
        a finding unextractable for a reason nobody would notice.
        """
        if text_length == 0:
            return 1.0
        covered = bytearray(text_length)
        for chunk in self.chunks:
            for i in range(chunk.start_offset, min(chunk.end_offset, text_length)):
                covered[i] = 1
        return sum(covered) / text_length


def _boundaries(text: str) -> list[int]:
    """Candidate split points: after sentence punctuation or a line break."""
    return [m.end() for m in _BOUNDARY.finditer(text)]


def _choose_split(text: str, start: int, hard_end: int, min_chars: int) -> int:
    """Pick the best split point at or before ``hard_end``.

    Prefers the latest sentence boundary that leaves a chunk of at least
    ``min_chars``; falls back to the latest whitespace; falls back to a hard
    cut. A hard cut is legal -- grounding tolerates a split mention because the
    overlap window catches it -- but it is the last resort.
    """
    if hard_end >= len(text):
        return len(text)

    window = text[start:hard_end]
    candidates = [start + b for b in _boundaries(window)]
    usable = [c for c in candidates if c - start >= min_chars]
    if usable:
        return usable[-1]

    space = window.rfind(" ")
    if space > min_chars:
        return start + space + 1
    return hard_end


def chunk_note(
    text: str,
    note_id: str,
    *,
    max_chars: int = 2000,
    overlap_chars: int = 200,
    section_headers: tuple[str, ...] | None = None,
    respect_sections: bool = True,
) -> ChunkPlan:
    """Split a note into overlapping, offset-tracked chunks.

    Overlap matters for recall: a finding whose sentence straddles a boundary
    would otherwise be invisible to both chunks. The cost is that the same
    mention may be extracted twice, which :mod:`fhirground.dedup` merges.

    Args:
        text: The original note text, used verbatim.
        note_id: Used to build chunk ids.
        max_chars: Maximum chunk length. Chunks may be shorter when a sentence
            boundary or a section edge falls earlier.
        overlap_chars: How far each chunk reaches back into the previous one.
        section_headers: Header list for section detection.
        respect_sections: When true, chunks do not span a section boundary, so
            a mention's section is unambiguous.

    Returns:
        A :class:`ChunkPlan` whose chunks tile the note with overlap.

    Raises:
        ConfigError: If the chunk parameters cannot make progress.
    """
    if max_chars < 50:
        raise ConfigError(f"max_chars must be at least 50, got {max_chars}")
    if overlap_chars < 0:
        raise ConfigError(f"overlap_chars must be non-negative, got {overlap_chars}")
    if overlap_chars >= max_chars:
        raise ConfigError(
            f"overlap_chars ({overlap_chars}) must be smaller than max_chars "
            f"({max_chars}), or chunking cannot advance"
        )

    sections = detect_sections(text, section_headers) if section_headers else detect_sections(text)
    if not text:
        return ChunkPlan(chunks=[], sections=sections)

    # Hard edges the chunker will not cross when respecting sections.
    edges: list[int] = [len(text)]
    if respect_sections and sections:
        edges = sorted({s.header_start for s in sections if s.header_start > 0} | {len(text)})

    chunks: list[Chunk] = []
    cursor = 0
    min_chars = max(1, min(max_chars // 4, 200))

    while cursor < len(text):
        next_edge = next((e for e in edges if e > cursor), len(text))
        hard_end = min(cursor + max_chars, next_edge)
        end = _choose_split(text, cursor, hard_end, min_chars)
        if end <= cursor:  # pragma: no cover - guarded by the parameter checks
            end = min(cursor + max_chars, len(text))

        chunks.append(
            Chunk(
                chunk_id=f"{note_id}#{len(chunks)}",
                index=len(chunks),
                text=text[cursor:end],
                start_offset=cursor,
                end_offset=end,
                section=section_at(sections, cursor),
            )
        )

        if end >= len(text):
            break
        # Step forward, then reach back by the overlap -- but never behind the
        # start of the chunk just emitted, or chunking would not terminate.
        cursor = max(cursor + 1, end - overlap_chars) if end < next_edge else end

    plan = ChunkPlan(chunks=chunks, sections=sections)
    for chunk in chunks:
        chunk.verify_against(text)
    return plan
