# Data formats

## Input: notes JSONL

```jsonc
{"note_id": "n1",
 "patient_ref": "Patient/123",            // required: every resource needs a subject
 "note_datetime": "2026-03-14T10:30:00Z", // anchors relative date resolution
 "text": "Assessment: Type 2 diabetes...",
 "encounter_ref": null,                   // optional
 "document_reference_id": "docref-n1"}    // optional; referenced by evidence
```

A bare id in `patient_ref` is rejected: `123` is not a FHIR reference and would
produce a resource whose subject does not resolve. Unknown fields are rejected,
so a typo is caught rather than ignored. Duplicate `note_id` values are
rejected.

## Input: FHIR DocumentReference

```bash
fhirground extract --input synthea/fhir/ --input-format fhir ...
```

Accepts a file or a directory, bare resources or Bundles. A `text/plain`
attachment is base64-decoded. Attachments of other types, and resources that
are not DocumentReferences, are **skipped rather than treated as errors** — a
Synthea bundle is full of both.

A DocumentReference with no `subject.reference`, or with neither `date` nor
`context.period.start`, **is** an error: without them the emitted resources
would have no patient, or relative dates could not be anchored.

## Mention schema (what the LLM returns)

See [the mention schema](mention-schema.md). Inspect it with
`fhirground schema --print`.

```json
{"mentions": [
  {"mention_type": "condition", "text": "type 2 diabetes",
   "evidence": "Type 2 diabetes, poorly controlled", "assertion": "present",
   "clinical_status_hint": null, "onset_text": null}
]}
```

## Value sets

```csv
system,code,display,synonyms
http://snomed.info/sct,44054006,Type 2 diabetes mellitus,type 2 diabetes|T2DM|DM2
http://www.nlm.nih.gov/research/umls/rxnorm,860975,metformin hydrochloride 500 MG Oral Tablet,metformin 500|glucophage 500
```

`synonyms` is pipe-separated. JSON is also accepted. Missing `system`, `code`,
or `display` is an error reporting the line number.

## Outputs

```
out/
  bundles/<note_id>.json
  ndjson/Condition.ndjson
  ndjson/MedicationRequest.ndjson
  ndjson/Observation.ndjson
  ndjson/Provenance.ndjson
  clinval_predictions.jsonl
  debug.jsonl
  run_summary.json
```

A resource type with no resources still gets an **empty NDJSON file**, so a
consumer can tell "none found" from "this run did not produce the type".

### Bundle

`type: collection` — not `transaction`, because fhirground does not write to a
server and a transaction Bundle would imply an intent to POST that v1 does not
support.

### debug.jsonl

One row per mention. The first file to read when a result disappoints.

```jsonc
{"note_id": "n1", "chunk_id": "n1#0",
 "mention": {"mention_type": "condition", "text": "type 2 diabetes", ...},
 "grounding": {"start": 12, "end": 46, "strategy": "exact", "score": 1.0,
               "ambiguous": false},
 "resolution": {"coded": true, "system": "http://snomed.info/sct",
                "code": "44054006", "score": 1.0, "resolver": "conds",
                "method": "synonym"},
 "decision": {"emit": true, "reason": null, "note": "present condition"},
 "confidence_components": {"g": 1.0, "c": null, "t": 1.0, "v": 1.0, "l": null},
 "score": 0.9857, "calibrated": false,
 "resource_type": "Condition", "resource_id": "n1-cond-0",
 "drop_reason": null, "n_duplicates": 1,
 "validation": {"ok": true, "n_errors": 0, "n_warnings": 0, "findings": []}}
```

Exactly one of `resource_id` and `drop_reason` is non-null on every row — a
test asserts it.

### run_summary.json

```jsonc
{"n_notes": 25, "n_mentions": 199,
 "resource_counts": {"Condition": 43, "MedicationRequest": 49, "Observation": 67},
 "drop_counts": {"negated": 20, "family": 20, "ungrounded": 0, ...},
 "repairs": 0, "tokens_in": 61220, "tokens_out": 12410,
 "n_validation_errors": 0, "n_validation_warnings": 0,
 "backend": {...}, "resolvers": {...}, "cache": {"hits": 0, "misses": 25},
 "versions": {"fhirground": "1.0.0", "prompt": "mention-v1",
              "fhir_library": "fhir.resources 8.3.0 (R4B)"},
 "cost_estimate": {"currency": "USD", "total": 0.3698}}
```

**Every drop reason appears, including the zeros.** A zero means checked and
not found, which is different from not checked.

With no token counts from the backend, `cost_estimate` becomes
`{"status": "unknown", "reason": "the backend reported no token counts"}`
rather than `0` — a zero would make the least-instrumented backend look
cheapest.

### clinval predictions

See [evaluating with clinval](clinval.md).

## Provenance

One per note, linking every emitted resource to its source and recording what
produced it:

```jsonc
{"resourceType": "Provenance", "id": "prov-n1",
 "target": [{"reference": "Condition/n1-cond-0"}, ...],
 "recorded": "2026-10-02T16:21:43+00:00",
 "activity": {"coding": [{"code": "DERIVE", ...}]},
 "agent": [{"who": {"display": "fhirground 1.0.0 | backend=mock | model=... | prompt=mention-v1 | schema=a1b2c3 | fhir.resources 8.3.0 (R4B)"}}],
 "entity": [{"role": "source", "what": {"reference": "DocumentReference/docref-n1"}}]}
```

The schema digest is in there so two runs that decoded against different
mention schemas are distinguishable. `agent.who` uses `display` rather than a
`Device/<id>` reference, because inventing a reference that resolves to nothing
would be worse than a descriptive string.
