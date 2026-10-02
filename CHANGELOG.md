# Changelog

All notable changes to fhirground are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-10-02

First release.

### Added

**The two-stage design**

- A mention schema with **no code fields**, so a model cannot emit a medical
  code, plus mapping functions that accept only `Resolution` objects, so a code
  cannot reach a resource without a terminology resolver. A property test
  drives adversarial input through the pipeline to check it.
- Backend-safe JSON Schema generation: refs inlined, `anyOf` collapsed to type
  unions, and the result re-checked against the subset Bedrock and vLLM both
  accept. The generated schema is checked in and CI asserts it matches the
  model.

**Input**

- Notes JSONL and FHIR `DocumentReference` (file, directory, or Bundle), with
  base64 `text/plain` attachments. `patient_ref` is required.
- Errors report the file and 1-based line number.

**Preprocessing**

- Configurable regex section detection, with family-history sections treated as
  a non-patient context.
- Offset-tracked chunking with overlap. Chunk text is a verbatim slice, and a
  property test asserts `original[chunk.start + i] == chunk.text[i]`.

**Extraction**

- `mock` (deterministic, offline, fixture or rule driven),
  `openai_compatible` (vLLM; `response_format: json_schema` or `guided_json`),
  and `bedrock` (Converse with a single forced tool).
- A repair loop that re-prompts with the validation error, capped at
  `max_repairs`, with the error written to be read by a model. Transport and
  auth failures do not consume repair attempts.
- An on-disk SQLite call cache keyed by backend, prompt, temperature, sample
  count, and schema digest.

**Grounding**

- `exact`, `normalized`, and `fuzzy` strategies, the last using
  `rapidfuzz.partial_ratio_alignment` so the located span is the aligned one.
- Ungrounded mentions are dropped and counted. Ambiguity is recorded.
- The stored quote is what the **note** says, not the model's paraphrase, which
  is what makes the offset/quote invariant checkable.

**Assertion handling**

- The full SPEC §4.4 table, every row tested, with a section override so a bare
  mention under *Family History* is treated as family history.
- Nine drop reasons, all reported including the zeros.

**Normalization**

- 58 UCUM codes. An unrecognized unit is carried as text with **no** system or
  code, enforced by a validation rule.
- Numbers, comparators, ranges (low bound plus a note), and blood-pressure
  pairs (string plus a note).
- Explicit and relative dates with precision preserved; ambiguous slash dates
  refused rather than guessed.

**Terminology**

- `LocalValueSet` (CSV or JSON, exact then token-set fuzzy), `NullResolver`,
  `Chain` (first or best), and an opt-in cached `RxNavResolver`.
- Per-mention-type routing. Display strings come from the value set, never from
  the model.

**Mapping and validation**

- Condition, MedicationRequest, and Observation to FHIR **R4B**, plus one
  `Provenance` per note and a `collection` Bundle.
- `authoredOn` is never inferred; an Observation's effective time may be, and
  carries the `effective-inferred` flag when it is.
- Model parsing, six project rules, the five relevant R4 invariants with text
  quoted verbatim from the StructureDefinitions, and an optional HL7 validator
  jar hook.

**Confidence**

- Five components with weights renormalized over those actually available, so
  an absent component is not scored as zero.
- `Calibrator` with isotonic and Platt methods, saved as **JSON, never
  pickle**, refusing to fit on fewer than 10 examples or a single class.

**Output**

- Bundles, NDJSON per type, `clinval_predictions.jsonl`, `debug.jsonl`, and
  `run_summary.json`. Empty NDJSON files are still written.
- `include_quotes=False` removes evidence quotes while keeping offsets.

**Benchmark and scoring**

- `fhirground.synth` renders notes with exact gold from the same source
  records, injecting negated, family-history, and hedged distractors, unit
  variants, comparators, abbreviations, and typos.
- A resource-level scorer with one-to-one matching and the precision caveat
  embedded in its own report.

**CLI**

- `extract`, `validate`, `synth`, `score`, `calibrate`, `schema`, `version`,
  with SPEC §5.1 exit codes (0, 1, 2, 4).

**Project**

- 618 tests at 95% coverage; ≥90% on grounding, assertion, mapping, validate,
  and confidence as §8.5 requires.
- `demo.py`, which is also an acceptance test: it exits non-zero if the schema
  accepts a code field, if any evidence offset disagrees with its quote, or if
  any emitted code is not in a value set.
- CI on Python 3.11–3.13 with jobs for the demo, the clinval contract, schema
  sync, and a guard against committed data or cache files.

### Known limitations

- Three resource types only; no AllergyIntolerance, Procedure,
  FamilyMemberHistory, MedicationStatement, or DiagnosticReport.
- No US Core or other profile conformance claims. R4B models are used because
  R4 models exist only in a pydantic-v1-only release line; for these three
  types the field sets are identical, which was measured.
- No cross-note or patient-level reconciliation.
- No de-identification, no UI, no FHIR server client.
- Synthea is not run and no Synthea fixtures are committed: it needs a JVM this
  environment lacks. `scripts/regenerate_synthea.sh` documents the path.
- No live-backend benchmark artifact is committed; the committed `score.json`
  is from the mock backend and labelled as such.
- Not published to PyPI and no documentation site deployed.

[1.0.0]: https://github.com/DanPalBCM/fhirground/releases/tag/v1.0.0
