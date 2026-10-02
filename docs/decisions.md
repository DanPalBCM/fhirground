# Decisions and deviations

Every decision that departs from the specification, resolves an ambiguity in
it, or rests on something **verified** about an external library. Each entry
says what was checked, when, and how to re-check it.

---

## D-001 — The name `fhirground` is available

**Status:** resolved, no approval stop needed.
**Checked:** 2 October 2026.

- **PyPI:** `fhirground` returns 404. Four candidate alternatives were also
  free, so no fallback was needed.
- **GitHub:** `search/repositories?q=fhirground+in:name` returns
  `total_count: 0`.

Re-check with:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/fhirground/json
gh api 'search/repositories?q=fhirground+in:name' --jq '.total_count'
```

---

## D-002 — R4 models are pydantic-v1 only, so fhirground uses **R4B**

**Status:** resolved. §0 explicitly sanctions R4B with documentation, so no
approval stop was required.
**Checked:** 2 October 2026, by installing and introspecting the releases
rather than reading about them.

§0 warned that `fhir.resources` has changed default FHIR versions. What is
actually true:

| Release line | Top-level models | Subpackages | Pydantic |
| --- | --- | --- | --- |
| 6.x (e.g. 6.5.0) | **R4 (4.0.1)** | `core` | **v1 only** |
| 7.x | R5 | `DSTU2`, `R4B`, `STU3`, `core` | v2 |
| 8.x (8.3.0) | R5 | `R4B`, `STU3` | v2 |

So **no released version offers R4 models under pydantic v2**, which §7
requires. Installing `fhir.resources==6.5.0` beside pydantic v2 fails at
import:

```
ImportError: cannot import name 'ROOT_VALIDATOR_CONFIG_KEY' from 'pydantic.class_validators'
```

**Decision:** pin `fhir.resources>=8.0,<9` and import from
`fhir.resources.R4B.*`.

### The R4 / R4B difference, measured

Rather than cite differences from memory, the field sets were dumped from both
lines (6.5.0 under pydantic v1, 8.3.0 under pydantic v2) and compared:

| Resource | R4 fields | R4B fields | Only in R4 | Only in R4B |
| --- | --- | --- | --- | --- |
| Condition | 42 | 41 | `resource_type` | — |
| MedicationRequest | 54 | 53 | `resource_type` | — |
| Observation | 58 | 57 | `resource_type` | — |

**For the three resource types fhirground maps, R4 and R4B are field-identical.**
The sole difference, `resource_type`, is a library implementation detail — 6.x
exposed it as a pydantic field, 8.x does not — and is not a FHIR difference.

Two further checks:

- R4B retains the R4 choice element **`medicationCodeableConcept`**. R5
  replaced it with a single `medication` of type `CodeableReference`, so taking
  the library's default (R5) would have broken §4.4's field mapping.
- R4B `Observation` and `Condition` carry every field §4.4 needs:
  `valueQuantity`, `valueString`, `effectiveDateTime`, `category`,
  `clinicalStatus`, `verificationStatus`, `onsetString`.

Consequently fhirground makes **no US Core or profile conformance claim**, and
the HL7 validator hook defaults to validating against `4.0.1`.

---

## D-003 — Both backends' structured-output shapes, verified

**Status:** resolved.
**Checked:** 2 October 2026.

### Bedrock Converse

Verified against **botocore 1.43.107's own service model** for
`bedrock-runtime`, not against documentation:

```python
model = botocore.session.get_session().get_service_model("bedrock-runtime")
shape = model.operation_model("Converse").input_shape
```

- `Converse` accepts `toolConfig`, whose members are `tools` (required) and
  `toolChoice`.
- `toolSpec` requires `name` and `inputSchema`; it also offers `description`
  and `strict`.
- `inputSchema` is a union whose only member is `json`, a free-form structure —
  so the mention schema goes in unchanged.
- `toolChoice` supports `any`, `auto`, and `tool: {name}`.

Converse has **no** `response_format`-style structured output, so the
forced-single-tool approach §3.3 describes is the correct one and is what
fhirground implements.

### OpenAI-compatible / vLLM

Verified against the OpenAI client's own type definitions
(`openai.types.shared_params.response_format_json_schema`, client 3.24.0):

```python
response_format = {
    "type": "json_schema",
    "json_schema": {"name": str, "schema": dict, "strict": bool | None},
}
```

fhirground also supports vLLM's older `guided_json` extra-body form for
servers that predate `response_format`.

### `strict` is **off** by default on both

OpenAI's strict mode requires every property to appear in `required`. The
mention schema has four required fields and thirteen optional ones **on
purpose** — a model forced to fill every field invents values for the ones it
cannot support, and an invented dose is worse than a missing one. vLLM's guided
decoding constrains to the schema without that requirement. So strict buys
nothing here and costs correctness; it remains configurable.

Both request builders are unit-tested with the real mention schema
(`tests/test_backends.py`), and the Bedrock request is additionally validated
by **botocore's `Stubber`**, which rejects a malformed request shape — the
strongest check available without an AWS account.

---

## D-004 — The §6.1 schema has a typo: `onset_text`

**Status:** deviation from the spec text, deliberately.

SPEC §6.1 declares:

```json
"onset_text": {"enum": ["string", "null"]}
```

As written, `onset_text` could only ever hold the **literal strings**
`"string"` or `"null"`. Every other nullable string field in the same schema
uses `{"type": ["string", "null"]}`, so this is plainly a typo rather than an
intent.

fhirground generates `{"type": ["string", "null"], "maxLength": 200}`, and a
test asserts it (`test_onset_text_is_a_nullable_string`).

Two related fixes in the generated schema:

- Pydantic emits `$defs` and `$ref` for enums and `anyOf` for optionality,
  none of which is in the subset §6.1 requires. `schemas/jsonschema.py` inlines
  refs, collapses `anyOf: [X, null]` into `type: [X, "null"]`, and re-checks
  the result with `assert_backend_safe`.
- A nullable enum needs its **`type`** widened too. Emitting
  `{"enum": [..., null], "type": "string"}` declares a type that contradicts
  the null it just allowed, and a strict validator rejects the value. The
  generator produces `"type": ["string", "null"]`.

---

## D-005 — FHIR invariants, quoted verbatim from the StructureDefinitions

**Status:** §4.4 requires verifying each invariant's exact text against the
spec. Done, and one of them does not say what it might be assumed to say.
**Checked:** 2 October 2026, against
`https://hl7.org/fhir/R4/condition.profile.json` and
`observation.profile.json`.

| Key | Severity | Verbatim text |
| --- | --- | --- |
| **con-3** | *warning* | "Condition.clinicalStatus SHALL be present if verificationStatus is not entered-in-error and category is problem-list-item" |
| **con-4** | error | "If condition is abated, then clinicalStatus must be either inactive, resolved, or remission" |
| **con-5** | error | "Condition.clinicalStatus SHALL NOT be present if verification Status is entered-in-error" |
| **obs-6** | error | "dataAbsentReason SHALL only be present if Observation.value[x] is not present" |
| **obs-7** | error | "If Observation.code is the same as an Observation.component.code then the value element associated with the code SHALL NOT be present" |

Three things worth recording:

1. **con-3's polarity.** It binds when category **is** problem-list-item — not
   "is not". Since §4.4 omits `category` by default, con-3 does not bind and
   `clinicalStatus` may be absent, which is what the mapper does. Turning on
   `include_category` makes con-3 bind, so the mapper then **refuses** to emit
   a Condition without a `clinical_status_hint` rather than produce a resource
   that trips a spec warning.
2. **con-3 is a warning, not an error.** It is reported at warning severity and
   does not block emission.
3. **The spacing typo in con-5** ("verification Status") is the spec's own and
   is reproduced verbatim so the text is searchable against the source.

con-1 and con-2 govern `Condition.stage` and `Condition.evidence`, which v1
never emits. obs-3 governs `referenceRange`, likewise. All five checks live in
`validate.check_fhir_invariants` with positive and negative tests.

---

## D-006 — Synthea is not run; records are generated instead

**Status:** deviation from §8.3, with the regeneration path documented.

§8.3 asks for a committed Synthea set (~50 patients, FHIR R4, fixed seed).
Synthea is a **Java** program, and fhirground's development and CI
environments have no JVM (`which java` → nothing).

Rather than commit fixtures nobody in this environment can regenerate, or skip
the benchmark:

- `fhirground.synth.generate_records` produces source records of the **same
  shape** that `records_from_synthea` extracts from a real Synthea bundle, so
  the renderer and the gold generator do not care which produced them;
- `fhirground.synth.records_from_synthea` is implemented and tested against a
  Synthea-shaped bundle, so the Synthea path works for anyone who has Java;
- `scripts/regenerate_synthea.sh` documents the pinned Synthea version
  (3.3.0) and the exact arguments;
- `examples/data/manifest.json` records `"source": "generated"` so no reader
  mistakes the committed set for Synthea output.

The benchmark property §8.3 actually depends on is preserved: gold FHIR is
generated **from the same source records** the notes were rendered from, so it
is exact by construction rather than annotated.

Synthea `DocumentReference` notes (§3.1, §8.3) are supported as an input format
and tested, but no such corpus is committed for the same reason.

---

## D-007 — A third error type is not needed, but `MATCH` is not used either

**Status:** clarification, for readers coming from the sibling project.

The sibling `clinval` adds a `MATCH` error type to its own taxonomy.
fhirground has no equivalent concept: a mention either becomes a resource or is
dropped with one of the `DropReason` values, and `NoteResult.dropped` plus the
emitted resources account for every group. A property test asserts that
identity.

---

## D-008 — `authoredOn` is never inferred

**Status:** implementation of §4.4, with the reasoning recorded.

§4.4 says `authoredOn` "is only set from an explicit resolved date, never
inferred". fhirground enforces this by **rejecting an inferred date** at the
mapper: `build_medication_request` ignores a `ResolvedDate` whose `inferred`
flag is set.

The asymmetry with Observation is deliberate. An Observation's
`effectiveDateTime` may be inferred from the note timestamp *because a flag
records that it was* (`effective-inferred`). `authoredOn` is the date a
prescriber wrote an order; defaulting it to "when the note was typed"
fabricates a clinical fact that looks authoritative and has no corresponding
flag in FHIR.

---

## D-009 — Ambiguous slash dates are refused

**Status:** resolution of an ambiguity in §3.9.

§3.9 lists `MM/DD/YYYY` as a supported format. `03/04/2026` is a valid date
under both US and day-first readings, and they differ by a month.

`resolve_date` therefore takes a `day_first` parameter. With it unset — the
default — month-first is used, **except** where both readings are valid and
different, in which case the date resolves to `None`. A missing
`effectiveDateTime` is recoverable; one that is a month off is a silently wrong
clinical fact.

Date **precision** is also preserved rather than padded: "since 2019" becomes
`2019`, "March 2019" becomes `2019-03`. Padding to `2019-01-01` would fabricate
precision the note never had.

---

## D-010 — Blood pressure pairs stay as text

**Status:** scope decision within §3.8.

`120/80` is two measurements. FHIR represents it with `Observation.component`,
which is out of v1 scope, and a single `valueQuantity` cannot hold it. So it
becomes a `valueString` **with an `Observation.note`** saying why:

> Reported as a pair '120/80', which is two measurements. v1 keeps it as
> valueString rather than splitting it into components.

Splitting it into invented systolic and diastolic components would be
hallucinating structure. Dropping the note would make the lossiness invisible.

Ranges (`5-10`) are handled the same way: §3.8 specifies the low bound becomes
the value, and fhirground attaches a note recording that the upper bound is not
represented.

---

## D-011 — Deduplication keys on the code when there is one

**Status:** implementation of §3.10.

§3.10 says merge on "the same type, normalized concept (code if resolved, else
normalized text), and assertion". Two consequences worth stating:

- With a resolver, `T2DM` and `type 2 diabetes` **merge** because they share a
  code. Without one they do **not**, because nothing established that they are
  the same concept. That is the conservative and correct outcome, not a
  limitation to work around.
- Assertion is part of the key, so "possible pneumonia" and "pneumonia" stay
  separate. They are different claims.

**Every evidence span survives a merge.** A merged resource carries one
`evidence` extension per source mention, so the audit trail does not shrink
when the resource count does.

---

## D-012 — The LLM-call cache contains note text

**Status:** documented hazard, per §3.16.

The cache is keyed by a hash of backend config, prompt, temperature, sample
count, and schema digest. The **value** includes the prompt, and the prompt
includes the chunk — so the SQLite file contains note text and is
note-derived data subject to the same handling rules as the notes.

It is off by default (`PipelineConfig.cache_path=None`), enabled with
`--cache`, and the PHI documentation says so in these words. The same applies
to the optional RxNav cache, which holds medication text taken from notes.

---

## D-013 — RxNav sends note-derived text to a third party

**Status:** documented hazard.

The RxNav resolver queries a public NLM service with **medication text
extracted from a clinical note**. That text is a note fragment. A site whose
data-use agreement forbids sending note content outside its boundary must not
enable it.

It is therefore opt-in with no default: `RxNavResolver` must be constructed
explicitly, it is never part of a default `ResolverConfig`, and it supports
`offline=True` to answer only from cache.

---

## D-014 — Logs carry ids, counts, timings, and hashes — never note text

**Status:** implementation of the §0 rule.

No module in `fhirground` logs note text. `NoteInput.text_hash()` exists so a
log line can identify a note without quoting it. The one place text
deliberately travels is `RawResponse.text`, because the repair loop needs the
model's own output to feed back; it is not logged.

`include_quotes=False` additionally removes evidence quotes from emitted
resources while **keeping the offsets**, so the claim stays checkable against
the note without the resource carrying the text (§4.5).

---

## D-015 — The committed value sets are not terminology

**Status:** scope boundary, per §8.3 and §10.

`examples/data/valuesets/*.csv` are derived from the codes the committed
synthetic records use — 22 concepts in total. They exist so the tests and the
demo can resolve codes at all.

They are **not** SNOMED CT, LOINC, or RxNorm content. Those carry licensing
obligations and are never bundled. A real deployment supplies its own
crosswalk via `LocalValueSet.from_csv`, and the docs say so on the terminology
page.

---

## D-016 — No live-backend benchmark artifact is committed

**Status:** §11 definition-of-done item **not met**, deliberately.

§11 asks for "a sample `score.json` from one live run committed under
`docs/benchmarks/` with model, date, and versions". That requires calling a
paid, credentialed LLM service, which is outside what this build can do without
the owner's credentials and consent.

What is committed instead: `docs/benchmarks/score-mock.json`, a full score
report from the **mock** backend, clearly labelled as such, with the model id
`mock-deterministic-v1`. The commands for producing a live artifact are in
`docs/benchmark.md`.

A mock-backend score measures the harness, not a model. It is committed so the
format and the pipeline are demonstrable, not as a performance claim.

---

## D-017 — `strip_to_json` must handle arrays, not only objects

**Status:** bug found by a test, fixed.

Recovering JSON from a wrapped response originally searched for the first `{`
and last `}`. Given a bare array — `[{...}, {...}]`, a common near-miss where
the model omits the `mentions` wrapper — that reduced the payload to its
**first element**, silently discarding every other mention while appearing to
succeed.

`strip_to_json` now recognizes objects and arrays and takes whichever bracket
opens first. Covered by `test_bare_array_accepted`.

---

## D-018 — Deduplication compared against a list it had already modified

**Status:** bug found by a test, fixed.

The representative mention of a group is promoted when a new mention is better
grounded. The check compared `grounding.score` against
`existing.best_grounding.score` — but the new grounding had already been
appended to `existing.groundings`, so it was inside the max it was being
compared to and could never win. A better-grounded duplicate therefore never
became the representative.

The incumbent's score is now captured before the append. Covered by
`test_best_grounding_wins_the_representative`.

---

## D-019 — Section labels measure from the header, not the body

**Status:** bug found by a test, fixed.

Chunks are cut **at** section headers, so a chunk's first offset is the header
offset. With section spans measured from the body start, that offset belonged
to no section and every chunk came back with `section=None` — which disabled
the family-history override that §4.4 depends on.

`Section.contains` now measures from `header_start`. `contains_body` is kept
for callers that want the body only.

---

## D-020 — Fuzzy grounding asks rapidfuzz where the match is

**Status:** bug found by a test, fixed.

The first fuzzy implementation slid a window, kept the best `partial_ratio`,
then tried to tighten it. That located the right *score* in the wrong *place*:
`"Start metformin 500mg PO BID"` grounded to `"tes, poorly controlled. A1c "`
with a score of 0.77. `partial_ratio` reports the best alignment anywhere
inside the window and says nothing about where it was.

Grounding now uses `rapidfuzz.fuzz.partial_ratio_alignment`, which returns the
aligned destination span alongside the score. Covered by
`test_near_miss_grounds_to_the_right_place`.

---

## D-021 — Release scope: GitHub only

**Status:** decided by the project owner.

Tagged `v1.0.0` on GitHub with documentation sources committed and buildable
locally. **Not** published to PyPI and no documentation site deployed, so
§11's "published to PyPI, docs deployed" items are deferred rather than met.
Nothing about the package blocks either step; a PyPI name and version are
effectively permanent, so that call is left to a deliberate later decision.
