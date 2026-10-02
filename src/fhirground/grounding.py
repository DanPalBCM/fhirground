"""Grounding: locating a mention's evidence quote in the original note.

A mention whose evidence cannot be found in the note is **dropped**, not
emitted with a guess. That is design principle 2, and it is the difference
between an auditable extraction and a plausible one: every resource fhirground
emits carries offsets that index the real note text, so a reviewer can always
go read the sentence the claim came from.

Three strategies, tried in order, each with a lower confidence contribution:

1. **exact** -- the quote appears verbatim. Score 1.0.
2. **normalized** -- it appears once whitespace is collapsed and case folded.
   Models routinely re-space and re-capitalize what they "copied". Score 0.9.
3. **fuzzy** -- rapidfuzz ``partial_ratio`` over a sliding window clears the
   threshold. Score ``ratio/100 * 0.8``.

Below the threshold, the mention is dropped and counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from rapidfuzz import fuzz

from fhirground.preprocess.chunking import Chunk

#: Default minimum rapidfuzz partial_ratio for a fuzzy hit, as SPEC §3.5.
DEFAULT_FUZZY_THRESHOLD = 90.0

_WS = re.compile(r"\s+")


class GroundingStrategy(StrEnum):
    """How an evidence quote was located. Recorded on the evidence extension."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    FUZZY = "fuzzy"


@dataclass(frozen=True, slots=True)
class Grounding:
    """Where a mention's evidence sits in the original note.

    Attributes:
        start: Offset in the **original note text**, not the chunk.
        end: Exclusive end offset in the original text.
        quote: The text actually found at those offsets, which may differ from
            the model's quote under normalized or fuzzy matching. Storing what
            is really there, rather than what the model claimed, is what makes
            the offset/quote invariant checkable.
        strategy: Which strategy succeeded.
        score: Grounding confidence component in [0, 1].
        ambiguous: True when the quote occurs more than once in the chunk and
            the first occurrence was taken.
        chunk_id: The chunk the match was found in.
    """

    start: int
    end: int
    quote: str
    strategy: GroundingStrategy
    score: float
    ambiguous: bool = False
    chunk_id: str | None = None

    @property
    def length(self) -> int:
        """Character length of the grounded span."""
        return self.end - self.start

    def verify(self, original: str) -> bool:
        """Whether ``original[start:end]`` still equals the stored quote."""
        return 0 <= self.start <= self.end <= len(original) and (
            original[self.start : self.end] == self.quote
        )


def normalize(text: str) -> str:
    """Collapse whitespace and fold case, for normalized matching."""
    return _WS.sub(" ", text).strip().lower()


def _find_exact(haystack: str, needle: str) -> tuple[int, bool] | None:
    """First exact occurrence and whether there are more."""
    first = haystack.find(needle)
    if first < 0:
        return None
    second = haystack.find(needle, first + 1)
    return first, second >= 0


def _find_normalized(haystack: str, needle: str) -> tuple[int, int, bool] | None:
    """Locate ``needle`` ignoring case and whitespace runs.

    Builds an index from normalized positions back to original positions, so
    the returned span is in **original** coordinates. Matching on a normalized
    copy and then reusing those offsets would be off by the number of
    collapsed characters.
    """
    norm_chars: list[str] = []
    back: list[int] = []
    previous_space = False
    for i, ch in enumerate(haystack):
        if ch.isspace():
            if previous_space:
                continue
            norm_chars.append(" ")
            back.append(i)
            previous_space = True
        else:
            norm_chars.append(ch.lower())
            back.append(i)
            previous_space = False

    norm = "".join(norm_chars).strip()
    # `.strip()` may have removed leading entries; recompute the alignment.
    lead = len("".join(norm_chars)) - len("".join(norm_chars).lstrip())
    back = back[lead : lead + len(norm)]

    target = normalize(needle)
    if not target:
        return None
    pos = norm.find(target)
    if pos < 0:
        return None
    more = norm.find(target, pos + 1) >= 0

    start = back[pos]
    last = back[min(pos + len(target) - 1, len(back) - 1)]
    return start, last + 1, more


def _find_fuzzy(haystack: str, needle: str, threshold: float) -> tuple[int, int, float] | None:
    """Best fuzzy alignment of ``needle`` within ``haystack``.

    Uses ``rapidfuzz.fuzz.partial_ratio_alignment``, which returns the aligned
    destination span alongside the score. An earlier implementation slid a
    window and kept the best ``partial_ratio``, then tried to tighten it --
    that located the right *score* in the wrong *place*, because
    ``partial_ratio`` reports the best alignment anywhere inside the window and
    says nothing about where. Asking the library for the alignment removes the
    guesswork entirely.
    """
    if not needle or not haystack:
        return None

    alignment = fuzz.partial_ratio_alignment(needle.lower(), haystack.lower())
    if alignment is None or alignment.score < threshold:
        return None

    start, end = alignment.dest_start, alignment.dest_end
    if end <= start:
        return None
    return start, end, float(alignment.score)


def ground_in_chunk(
    evidence: str,
    chunk: Chunk,
    original: str,
    *,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> Grounding | None:
    """Locate ``evidence`` inside one chunk, returning original-text offsets.

    Args:
        evidence: The model's quote.
        chunk: The chunk the mention was extracted from.
        original: The full original note text, used to store the real quote.
        fuzzy_threshold: Minimum ``partial_ratio`` for a fuzzy hit.

    Returns:
        A :class:`Grounding`, or None if the quote could not be located.
    """
    if not evidence or not evidence.strip():
        return None

    hay = chunk.text

    hit = _find_exact(hay, evidence)
    if hit is not None:
        local, more = hit
        start = chunk.to_original(local)
        end = start + len(evidence)
        return Grounding(
            start=start,
            end=end,
            quote=original[start:end],
            strategy=GroundingStrategy.EXACT,
            score=1.0,
            ambiguous=more,
            chunk_id=chunk.chunk_id,
        )

    norm_hit = _find_normalized(hay, evidence)
    if norm_hit is not None:
        local_start, local_end, more = norm_hit
        start, end = chunk.to_original(local_start), chunk.to_original(local_end)
        return Grounding(
            start=start,
            end=end,
            quote=original[start:end],
            strategy=GroundingStrategy.NORMALIZED,
            score=0.9,
            ambiguous=more,
            chunk_id=chunk.chunk_id,
        )

    fuzzy_hit = _find_fuzzy(hay, evidence, fuzzy_threshold)
    if fuzzy_hit is not None:
        local_start, local_end, ratio = fuzzy_hit
        start, end = chunk.to_original(local_start), chunk.to_original(local_end)
        return Grounding(
            start=start,
            end=end,
            quote=original[start:end],
            strategy=GroundingStrategy.FUZZY,
            score=round((ratio / 100.0) * 0.8, 4),
            ambiguous=False,
            chunk_id=chunk.chunk_id,
        )

    return None


def ground_mention(
    evidence: str,
    chunks: list[Chunk],
    original: str,
    *,
    preferred_chunk_id: str | None = None,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> Grounding | None:
    """Locate evidence across chunks, preferring the chunk it came from.

    The preferred chunk is tried first because overlap means the same sentence
    may appear in two chunks, and the offsets should attribute to the chunk the
    model actually saw. Other chunks are then tried so that a quote spanning a
    boundary still grounds.

    Returns:
        The best :class:`Grounding` found, or None.
    """
    ordered = list(chunks)
    if preferred_chunk_id is not None:
        ordered.sort(key=lambda c: c.chunk_id != preferred_chunk_id)

    best: Grounding | None = None
    for chunk in ordered:
        found = ground_in_chunk(evidence, chunk, original, fuzzy_threshold=fuzzy_threshold)
        if found is None:
            continue
        if found.strategy is GroundingStrategy.EXACT:
            return found
        if best is None or found.score > best.score:
            best = found
    return best
