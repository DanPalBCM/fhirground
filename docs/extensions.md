# Extensions reference

Three extensions carry everything fhirground adds beyond plain FHIR. The
StructureDefinitions are **not shipped** in v1; the URLs identify the
extensions and this page defines their semantics.

## The base URL

```bash
fhirground extract --extension-base https://example.org/fhir/StructureDefinition/ ...
```

The default is an explicit placeholder. Publishing resources under someone
else's StructureDefinition URL would be worse than a visible placeholder, so
set this to something you control before sharing output.

## extraction-confidence

```json
{"url": ".../extraction-confidence",
 "valueDecimal": 0.8734,
 "extension": [{"url": "calibrated", "valueBoolean": false}]}
```

Exactly one per resource — a project rule enforces it. `valueDecimal` is in
`[0, 1]`; out of range raises at build time.

`calibrated` is the field to read. `false` means the score is a weighted mean
of heuristics and **not a probability**. See [confidence](confidence.md).

## evidence

Complex, and repeated once per source mention when deduplication merged
several.

```json
{"url": ".../evidence", "extension": [
  {"url": "source", "valueReference": {"reference": "DocumentReference/docref-n1"}},
  {"url": "start", "valueInteger": 12},
  {"url": "end", "valueInteger": 46},
  {"url": "quote", "valueString": "Type 2 diabetes, poorly controlled"},
  {"url": "grounding", "valueCode": "exact"}
]}
```

| Sub-extension | Meaning |
| --- | --- |
| `source` | the source `DocumentReference`, when known |
| `start` / `end` | character offsets into the **original note text** |
| `quote` | what is actually at those offsets; omitted when `include_quotes: false` |
| `grounding` | `exact`, `normalized`, or `fuzzy` |

At least one per resource — a project rule enforces that too, because an
ungrounded resource is not auditable.

The invariant: `note_text[start:end]` normalized equals `quote` normalized.
With `include_quotes: false` the quote is gone but the **offsets must still be
in range**, so the claim stays checkable against the note without the resource
carrying the text.

A merged resource carries one `evidence` extension per source mention. The
audit trail does not shrink when the resource count does.

## effective-inferred

```json
{"url": ".../effective-inferred", "valueBoolean": true}
```

Observation only. Present when `effectiveDateTime` came from the **note
timestamp** rather than from the note text, because the text had no resolvable
date.

This flag is why inferring an Observation time is acceptable while inferring
`authoredOn` is not: the inference is recorded, so a reader can tell. There is
no equivalent FHIR flag for an order date. See
[assertion and mapping](assertion-mapping.md).

Suppress with `mark_inferred_effective=False` for consumers that reject unknown
extensions — but then the inference becomes invisible, which is a worse
trade-off than it looks.

## Reading them back

```python
from fhirground.mapping import read_calibrated, read_confidence, read_evidence

read_confidence(resource)   # 0.8734 or None
read_calibrated(resource)   # False or None
read_evidence(resource)     # [{"start": 12, "end": 46, "quote": ..., ...}]
```

All three return `None` or `[]` when the extension is absent, rather than
raising — a resource from another producer is a legitimate input.
