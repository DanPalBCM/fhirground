# fhirground

**Grounded, validated FHIR R4 extraction from clinical notes.**

[![tests](https://github.com/DanPalBCM/fhirground/actions/workflows/ci.yml/badge.svg)](https://github.com/DanPalBCM/fhirground/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Health systems want structured data from clinical notes, and the format is
increasingly FHIR. The existing options each fail differently:

| Approach | What goes wrong |
| --- | --- |
| Ask an LLM for FHIR directly | invalid resources, misplaced fields, **hallucinated codes** |
| General extraction libraries | grounded spans, but custom JSON — not FHIR |
| Rule and dictionary NLP | concepts, but not validated resources with assertion status and provenance |

fhirground splits the problem in two.

---

## The two-stage design

```
                    ┌──────────────────────────────┐
   clinical note ──►│  1. the LLM says MENTIONS    │
                    │                              │
                    │  plain language + a verbatim │
                    │  evidence quote + assertion  │
                    │                              │
                    │  the schema has NO code      │
                    │  fields, so a model has      │
                    │  nowhere to put one          │
                    └──────────────┬───────────────┘
                                   │  mentions
                                   ▼
                    ┌──────────────────────────────┐
                    │  2. CODE says FHIR           │
                    │                              │
                    │  ground to offsets           │
                    │  apply the assertion policy  │
                    │  normalize units and dates   │
                    │  resolve codes via resolvers │
                    │  map, validate, score        │
                    └──────────────┬───────────────┘
                                   ▼
            Bundle · NDJSON · clinval predictions · debug traces
```

**FHIR structure never depends on the model's formatting.** That is the whole
idea, and everything below follows from it.

---

## Four guarantees, each enforced in code

**Grounded.** Every emitted resource carries evidence offsets that index the
original note. A mention whose quote cannot be found is dropped and counted —
not emitted with a guess.

**No invented codes.** Two mechanisms, together. The mention schema has no code
fields, so a model cannot produce one. And every mapping function accepts a
`Resolution` object rather than a string, so a code cannot reach a resource
without having come from a terminology resolver. A property test drives
adversarial mentions — text containing `"SNOMED 44054006"` — through the
pipeline and asserts every emitted `coding.code` exists in the active value
sets.

**Validated.** Every resource parses under the pinned FHIR R4B models and
passes the project rules, or it is dropped with a reason. The three R4
invariants that bind on these resource types are encoded with their text
quoted verbatim from the StructureDefinitions.

**Assertion-aware.** Negated, family-history, and hypothetical mentions are
never emitted as affirmed facts.

---

## Quickstart

```bash
git clone https://github.com/DanPalBCM/fhirground
cd fhirground
uv sync
uv run python demo.py
```

The demo runs the whole pipeline on the toy benchmark committed in
`examples/data/` using the deterministic `mock` backend — no API key, no
network. It also tries to smuggle a hallucinated code past the schema and shows
it being rejected.

### With a vLLM server

```bash
uv run fhirground extract \
  --input notes.jsonl --backend openai_compatible \
  --base-url http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct \
  --value-sets valuesets/ --out out/ --format bundle,ndjson,clinval,debug
```

### With Bedrock

```bash
uv run fhirground extract \
  --input synthea/fhir/ --input-format fhir --backend bedrock \
  --model "$FHIRGROUND_BEDROCK_MODEL_ID" --region us-east-1 --samples 3 --out out/
```

Credentials come from the standard boto3 chain. fhirground never holds AWS keys.

---

## The worked example

```python
import fhirground as fg
from fhirground.terminology import Chain, LocalValueSet, ResolverConfig

resolvers = ResolverConfig(
    condition=Chain([LocalValueSet.from_csv("valuesets/conditions.csv")]),
    medication=Chain([LocalValueSet.from_csv("valuesets/meds.csv")]),
    observation=Chain([LocalValueSet.from_csv("valuesets/obs.csv")]),
)

pipe = fg.Pipeline(
    fg.PipelineConfig(samples=1, max_repairs=2, emit_refuted=False),
    backend=fg.backends.MockBackend(),
    resolvers=resolvers,
)

note = fg.NoteInput(
    note_id="n1",
    patient_ref="Patient/123",
    note_datetime="2026-03-14T10:30:00Z",
    text="Assessment: Type 2 diabetes, poorly controlled. A1c 9.2% today. "
         "Start metformin 500 mg PO BID. Denies chest pain. Mother has hypertension.",
)

result = pipe.run_note(note)
print(result.bundle)

for mention, reason in result.dropped:
    print(reason.value, mention.text)
```

Produces **exactly**:

| Resource | Code | Detail |
| --- | --- | --- |
| `Condition` | SNOMED `44054006` | `verificationStatus: confirmed`, no `clinicalStatus` (no hint given) |
| `Observation` | LOINC `4548-4` | `valueQuantity` 9.2 `%` with UCUM `system`/`code`, `effectiveDateTime` 2026-03-14 |
| `MedicationRequest` | RxNorm `860975` | `active`/`order`, dose 500 mg, route PO |
| `Provenance` | — | targets all three, records model, prompt version, and FHIR library |

and drops:

```
negated   chest pain
family    hypertension
```

`clinicalStatus` is **absent**, not defaulted to `active`. FHIR invariant
con-3 only requires it when `category` is `problem-list-item`, which
fhirground omits by default — so guessing a clinical course the note never
stated would be fabrication. See [`docs/decisions.md`](docs/decisions.md) D-005.

This example is a test (`tests/test_pipeline.py::TestSpecExample`).

---

## What the demo reports

Real output on the 25 committed synthetic notes:

```
  resource type               n
  ---------------------- ------
  Condition                  43
  MedicationRequest          49
  Observation                67
  Provenance                 25
  ✓ every emitted resource parses as FHIR R4B and passes the project rules

    negated                    20  ████████████████████
    family                     20  ████████████████████
    ungrounded               none
    hypothetical             none

  ✓ 159 evidence spans verified: note_text[start:end] equals the stored quote
  ✓ all 22 distinct emitted codes came from a terminology resolver

  | resource type | gold | predicted | TP | FP | FN | precision | recall |  F1   |
  | Condition         |  48 |  43 | 31 | 12 | 17 | 0.721 | 0.646 | 0.681 |
  | MedicationRequest |  49 |  49 | 49 |  0 |  0 | 1.000 | 1.000 | 1.000 |
  | Observation       |  67 |  67 | 67 |  0 |  0 | 1.000 | 1.000 | 1.000 |
  | **all**           | 164 | 159 |147 | 12 | 17 | 0.925 | 0.896 | 0.910 |
```

Those scores measure the **mock backend**, which is a deterministic rule
extractor — they say nothing about any LLM. They are shown because the
*harness* behaviour is what matters here: 100% valid FHIR, every span
verifiable, every code traceable, and 40 distractors declined rather than
emitted.

---

## Interoperability

fhirground is one of three projects that share a data contract:

- **fhirground** (this project) — notes to grounded FHIR.
- **[clinval](https://github.com/DanPalBCM/clinval)** — the validation harness.
  fhirground writes `clinval_predictions.jsonl`, so clinval evaluates its
  output **with no translation step**. Verified end to end: clinval parses
  fhirground's output directly and the shared `run.call_id` makes its cost
  accounting charge one model call once rather than once per label.
- **[ehrlint](https://github.com/DanPalBCM/ehrlint)** — leakage auditor for EHR
  prediction tasks.

```bash
fhirground extract --input notes.jsonl --out out/ --format clinval
clinval run --config eval.yaml --pred out/clinval_predictions.jsonl --gold gold.jsonl --out report/
```

---

## Limitations

Stated plainly, because a pipeline that oversells itself in this domain is
worse than useless.

- **Not a clinical decision support tool.** Outputs are not for direct patient
  care without human review.
- **No accuracy guarantee.** fhirground guarantees *structure, grounding, and
  provenance*. Accuracy is measured — by the bundled scorer or by clinval — not
  promised.
- **Three resource types** in v1: Condition, MedicationRequest, Observation.
  No AllergyIntolerance, Procedure, FamilyMemberHistory, MedicationStatement,
  or DiagnosticReport.
- **No profile conformance claims.** Not US Core, not any other profile.
  fhirground uses the **R4B** models because R4 models exist only in a
  pydantic-v1-only release line; for these three resource types the two are
  field-identical, which was measured rather than assumed
  ([D-002](docs/decisions.md)).
- **Uncalibrated confidence is not a probability.** It is a weighted mean of
  heuristics. Every resource records whether it was calibrated; fit a
  calibrator on labelled data to make the number mean something.
- **Not a terminology server.** The committed value sets cover 22 synthetic
  concepts. SNOMED CT, LOINC, and RxNorm carry licensing obligations and are
  never bundled — supply your own crosswalk.
- **No cross-note reconciliation.** Two notes mentioning one diagnosis produce
  two Conditions. Patient-level merging is its own problem.
- **`120/80` stays a string.** A blood pressure pair is two measurements, and
  `Observation.component` is out of v1 scope. An `Observation.note` records
  why.
- **The LLM-call cache contains note text**, because the prompt does. It is off
  by default. So does the optional RxNav cache, which additionally **sends
  medication text to a third party** — opt-in only
  ([D-012, D-013](docs/decisions.md)).

---

## Formats

```jsonc
// input: notes.jsonl
{"note_id": "n1", "patient_ref": "Patient/123",
 "note_datetime": "2026-03-14T10:30:00Z", "text": "Assessment: ...",
 "encounter_ref": null, "document_reference_id": "docref-n1"}

// what the LLM returns -- note the complete absence of code fields
{"mentions": [
  {"mention_type": "condition", "text": "type 2 diabetes",
   "evidence": "Type 2 diabetes, poorly controlled", "assertion": "present",
   "clinical_status_hint": null, "onset_text": null}
]}

// value sets: system,code,display,synonyms  (synonyms pipe-separated)
// http://www.nlm.nih.gov/research/umls/rxnorm,860975,metformin ...,metformin 500|glucophage 500
```

Outputs: `bundles/<note_id>.json`, `ndjson/<Type>.ndjson`,
`clinval_predictions.jsonl`, `debug.jsonl`, `run_summary.json`.

Full reference: [`docs/formats.md`](docs/formats.md). Inspect the decoding
schema with `fhirground schema --print`.

---

## No real data, anywhere

Every dataset in this repository is **generated**. `fhirground.synth` assembles
notes from fixed concept banks with sequential identifiers (`synth-00001`) and
dates derived from a constant epoch. Gold FHIR is generated from the same
source records the notes were rendered from, so the benchmark is exact by
construction.

Synthea is **not** run: it is a Java program and this environment has no JVM,
so records are generated internally instead and
[`scripts/regenerate_synthea.sh`](scripts/regenerate_synthea.sh) documents the
pinned version and arguments for anyone who wants the Synthea path
([D-006](docs/decisions.md)).

No test makes a network call. Live-backend tests are marked and skipped unless
their environment variable is set.

---

## Development

```bash
uv sync --all-extras
uv run pytest                 # 618 tests, 95% coverage
uv run pytest --cov=fhirground --cov-report=term-missing
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run fhirground schema --check   # the checked-in schema matches the model
```

Property tests (hypothesis) cover the four §8.2 invariants: no invented codes,
FHIR validity, the grounding offset/quote identity, and that dropped plus
emitted mentions account for every group. Request builders are tested against
the real mention schema, with Bedrock additionally validated by botocore's
`Stubber` and the OpenAI-compatible backend driven through `respx`.

Every decision that departs from the specification, resolves an ambiguity, or
rests on something verified about an external library is recorded in
[`docs/decisions.md`](docs/decisions.md) — including four bugs that tests
caught and what they were.

---

## Citing

See [`CITATION.cff`](CITATION.cff).

```bibtex
@software{fhirground,
  author  = {Palacios, Daniel},
  title   = {fhirground: grounded, validated FHIR R4 extraction from clinical notes},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/DanPalBCM/fhirground}
}
```

## License

[Apache-2.0](LICENSE). Chosen for the patent grant and health-enterprise
acceptability, matching Synthea and much of the FHIR Python ecosystem.
Terminology content is never bundled beyond synthetic fixtures; see
[`docs/terminology.md`](docs/terminology.md) for third-party licensing notes.
