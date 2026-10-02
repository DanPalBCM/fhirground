# Getting started

## Install

fhirground needs Python 3.11 or newer.

```bash
git clone https://github.com/DanPalBCM/fhirground
cd fhirground
uv sync
```

Optional extras:

```bash
uv sync --extra bedrock      # boto3, for the Bedrock backend
uv sync --extra calibrate    # scikit-learn, for confidence calibration
uv sync --all-extras
```

## Run the demo

```bash
uv run python demo.py
```

It runs the whole pipeline on the toy benchmark in `examples/data/` using the
deterministic `mock` backend — no API key, no network. It also tries to smuggle
a hallucinated code past the mention schema and shows it being rejected, then
scores the output against gold.

Output lands in `demo_output/`.

## Your own notes

### 1. Prepare the input

One JSON object per line. `patient_ref` is **required** — every resource needs
a subject, and failing at read time beats discovering it during mapping.

```jsonc
{"note_id": "n1", "patient_ref": "Patient/123",
 "note_datetime": "2026-03-14T10:30:00Z",
 "text": "Assessment: Type 2 diabetes...",
 "encounter_ref": null, "document_reference_id": "docref-n1"}
```

Synthea FHIR output works directly:

```bash
fhirground extract --input synthea/fhir/ --input-format fhir ...
```

### 2. Supply value sets

Without them every concept is text-only. A directory with
`conditions.csv`, `medications.csv`, and `observations.csv`:

```csv
system,code,display,synonyms
http://snomed.info/sct,44054006,Type 2 diabetes mellitus,type 2 diabetes|T2DM
```

`synonyms` is pipe-separated. A missing file means that mention type stays
text-only, which is a legitimate configuration. See
[terminology](terminology.md) for licensing.

### 3. Extract

```bash
# offline, deterministic -- good for a dry run
fhirground extract --input notes.jsonl --value-sets valuesets/ --out out/

# a vLLM server
fhirground extract --input notes.jsonl --backend openai_compatible \
  --base-url http://localhost:8000/v1 --model meta-llama/Llama-3.1-8B-Instruct \
  --value-sets valuesets/ --out out/ --format bundle,ndjson,clinval,debug

# Bedrock, with three samples for a self-consistency score
fhirground extract --input notes.jsonl --backend bedrock \
  --model "$MODEL_ID" --region us-east-1 --samples 3 --temperature 0.4 \
  --value-sets valuesets/ --out out/
```

### 4. Check the output

```bash
fhirground validate out/bundles --notes notes.jsonl
```

With `--notes`, evidence offsets are checked against the note text. Add
`--hl7-validator /path/validator_cli.jar` to also run the official HL7
validator, if you have the jar.

## Commands

| Command | What it does |
| --- | --- |
| `fhirground extract` | notes to FHIR |
| `fhirground validate` | model parse, project rules, optional HL7 jar |
| `fhirground synth` | generate synthetic notes with exact gold |
| `fhirground score` | score predictions against gold FHIR |
| `fhirground calibrate` | fit a confidence calibrator on labelled data |
| `fhirground schema` | print, write, or check the decoding schema |
| `fhirground version` | versions of fhirground and its FHIR library |

Exit codes: `0` ok, `1` some notes failed, `2` usage error, `4` backend or auth
error.

## Reading the output

```
out/
  bundles/<note_id>.json          one FHIR Bundle per note
  ndjson/Condition.ndjson         bulk-data style, one file per type
  clinval_predictions.jsonl       for the clinval harness
  debug.jsonl                     one trace per mention -- read this first
  run_summary.json                counts, drop reasons, tokens, versions
```

`debug.jsonl` is the file that makes a disappointing result diagnosable: one
row per mention with its grounding strategy, resolution, confidence components,
and either the emitted resource id or the drop reason.

## Limitations

- **Not a clinical decision support tool.** Outputs need human review.
- **No accuracy guarantee.** fhirground guarantees structure, grounding, and
  provenance; accuracy is *measured*, not promised.
- **Three resource types** in v1: Condition, MedicationRequest, Observation.
- **No profile conformance claims** — not US Core, not any other.
- **Uncalibrated confidence is not a probability.** See
  [confidence](confidence.md).
- **Not a terminology server.** Bring your own value sets.
- **No cross-note reconciliation.**
- **The LLM-call cache contains note text.** See [PHI](phi.md).
