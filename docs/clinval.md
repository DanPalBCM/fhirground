# Evaluating with clinval

[clinval](https://github.com/DanPalBCM/clinval) is the sibling validation
harness. fhirground writes its prediction format directly, so there is **no
translation step**.

```bash
fhirground extract --input notes.jsonl --value-sets valuesets/ \
  --out out/ --format clinval

clinval run --config eval.yaml \
  --pred out/clinval_predictions.jsonl --gold gold.jsonl \
  --index index.csv --out report/ --i-understand-this-is-dev
```

## The contract

| clinval field | fhirground value |
| --- | --- |
| `system` | `fhirground-<backend>` |
| `note_id` | the note id |
| `patient_id` | `patient_ref`, e.g. `Patient/123` |
| `label` | `<ResourceType>:<code or normalized text>` |
| `value` | always `true` |
| `span` | from the first evidence extension |
| `confidence` | the `extraction-confidence` value |
| `run` | model, prompt version, tokens, latency, `call_id` |

```json
{"system": "fhirground-mock", "note_id": "synth-note-00001",
 "patient_id": "Patient/synth-00001", "label": "Condition:44054006",
 "value": true, "confidence": 0.9857,
 "span": {"start": 134, "end": 177, "text": "Type 2 diabetes, stable on current therapy."},
 "run": {"model": "mock-deterministic-v1", "prompt_version": "mention-v1",
         "call_id": "synth-note-00001", "tokens_in": 2439, "tokens_out": 495}}
```

## Two details that matter

**`label` is colon-qualified and high-cardinality.** clinval treats labels as
**opaque strings** and never splits them, so `Condition:K59.1` is safe. But
because the label embeds the code, a cohort with two hundred distinct
conditions produces two hundred labels, most with a handful of positives.
Per-label metrics on five positives are not informative — set `labels:` in your
clinval config to the conditions your study is actually about.

**`run.call_id` is the note id.** One pipeline pass produces every record for a
note, so they share a call id and clinval charges the model call **once** rather
than once per label. Without it, cost would be multiplied by the number of
resources per note.

## Only positives are emitted

fhirground writes no `value: false` records. A drop means fhirground **declined
to assert** a finding, which is not the same as asserting its absence — and
clinval reads a missing record as *unscored*, not as a negative.

Materializing negatives needs the label universe, which only your study knows:

```bash
clinval negatives --from out/clinval_predictions.jsonl --notes notes.jsonl \
  --labels "Condition:44054006,Condition:38341003" --out filled.jsonl
```

Without that step clinval will warn that coverage is below 100%, which is the
correct signal rather than a problem to suppress.

## Reading an ontology_mismatch

fhirground's hard rule is that the **LLM never emits a code**: the model
extracts plain-language mentions, and codes come only from the terminology
resolver.

So when clinval tags an `ontology_mismatch`, the model did not hallucinate a
code — it cannot. The **resolver** mapped a mention to a different concept than
the annotator did. That is a terminology-mapping problem, not a prompting one,
and the fix is in your value sets rather than your prompt.

For category-level agreement, use ICD-10 prefix matching in the clinval config:

```yaml
matching:
  ontology:
    icd-10:
      prefix_depth: 3     # E11.9 and E11.65 both reduce to E11
```

## Which tool answers which question

| Question | Tool |
| --- | --- |
| Is the FHIR valid, grounded, and code-clean? | fhirground's own validation |
| Did we recover the source records? | `fhirground score` |
| Does the system agree with clinician annotators, with CIs? | **clinval** |
| Where do the residual errors come from, by mechanism? | **clinval** |
| Is the dev/test protocol honest? | **clinval** |

`fhirground score` measures recovery of structured records a synthetic note was
rendered from. clinval measures agreement with human judgement. They are
different questions and neither substitutes for the other.
