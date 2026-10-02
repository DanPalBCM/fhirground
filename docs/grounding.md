# Grounding

**A mention whose evidence cannot be found in the note is dropped.** Not
emitted with a guess, not emitted without offsets — dropped and counted. That
is design principle 2, and it is the difference between an auditable
extraction and a plausible one.

## Three strategies

Tried in order, each contributing less confidence:

| Strategy | Condition | Score |
| --- | --- | --- |
| `exact` | the quote appears verbatim | 1.0 |
| `normalized` | it appears once whitespace is collapsed and case folded | 0.9 |
| `fuzzy` | `partial_ratio` clears the threshold (default 90) | `ratio/100 * 0.8` |

Normalized matching exists because models routinely re-space and
re-capitalize what they "copied". Fuzzy exists because some re-tokenize and
lose exact positions.

```python
from fhirground.grounding import ground_mention
g = ground_mention("type 2  DIABETES, poorly controlled", chunks, note_text)
g.strategy    # GroundingStrategy.NORMALIZED
g.quote       # 'Type 2 diabetes, poorly controlled'  -- what the NOTE says
g.verify(note_text)   # True
```

## The stored quote is the note's, not the model's

Under normalized or fuzzy matching the model's wording differs from the note's.
The `Grounding` stores **what is actually at those offsets**, which is what
makes the offset/quote invariant checkable at all:

```
note_text[start:end] == quote      # a project validation rule
```

If the quote stored were the model's paraphrase, this rule could never hold and
nothing downstream could be verified.

## Offsets are original-note offsets

Chunking guarantees `original[chunk.start_offset + i] == chunk.text[i]`, so
chunk text is a **verbatim slice** — never stripped, normalized, or re-joined.
A single stripped character would shift every offset after it, and the
resulting FHIR would point at the wrong words while looking perfectly valid. A
property test asserts the identity over generated text and chunk parameters.

`Chunk.to_original()` raises rather than returning a plausible-but-wrong global
offset for an out-of-range local one.

## Ambiguity is recorded

When a quote occurs more than once in a chunk, the **first** occurrence is
taken and `ambiguous=True` is recorded. Picking silently would make two
different sentences indistinguishable in the audit trail.

## One fixed bug worth knowing

The first fuzzy implementation slid a window, kept the best `partial_ratio`,
and tried to tighten it. That found the right *score* in the wrong *place*:
`"Start metformin 500mg PO BID"` grounded to `"tes, poorly controlled. A1c "`.
`partial_ratio` reports the best alignment anywhere inside the window and says
nothing about where it was.

Grounding now uses `rapidfuzz.fuzz.partial_ratio_alignment`, which returns the
aligned span alongside the score. See [decisions](decisions.md) D-020 — a
reminder that a plausible score is not a correct location.

## Chunking

```python
chunk_note(text, note_id, max_chars=2000, overlap_chars=200,
           respect_sections=True)
```

Chunks overlap so a finding whose sentence straddles a boundary is not
invisible to both; the cost is a duplicate mention, which
[deduplication](formats.md) merges. With `respect_sections` on, chunks do not
span a section boundary, so a mention's section is unambiguous.

Notes that cannot be placed are kept, not dropped, with a reason:
`outside_all_windows`, `no_note_datetime`, `patient_not_in_index`,
`no_patient_id`.
