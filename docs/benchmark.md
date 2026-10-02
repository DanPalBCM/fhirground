# The synthetic benchmark

```bash
fhirground synth --out examples/data --n-notes 50 --seed 0
```

## How it works

Take structured source records, render a note that states them in clinical
prose, and emit the FHIR those records imply. Because both sides come from the
**same records**, the gold is exact by construction rather than annotated —
which is what makes an offline, reproducible benchmark possible at all.

```
source records ──┬──► rendered note (prose, with distractors)
                 └──► gold FHIR (exact)
```

## What gets injected

The templates deliberately include what makes extraction hard:

| Injected | Why |
| --- | --- |
| **negated distractors** | findings the record does *not* contain, stated as denials. A pipeline ignoring negation turns these into false positives, and there is no way to score that without them. |
| **family history** | conditions attributed to a relative |
| **hedged mentions** | "possible", "rule out" — should become `provisional`, not dropped |
| **unit spelling variants** | `mg/dL`, `mg/dl`, `milligrams per deciliter` |
| **comparators** | `<0.01`, `>140` |
| **abbreviations** | `T2DM`, `HTN`, `A1c` |
| **light typos** | a transposition or doubled letter — what fuzzy grounding exists to survive |

Every distractor is recorded in `manifest.json` with what should happen to it,
so a scorer can verify that negated findings were **dropped** rather than
merely absent.

## Scoring

```bash
fhirground score --pred out/ndjson --gold examples/data/gold \
  --summary out/run_summary.json --out score.json
```

A match needs all of: same resource type, same code (or same normalized text
when neither side is coded), compatible status fields, and numeric values
within tolerance — 1% relative by default, **exact for integers**, because 72
and 73 bpm are different observations however small the relative difference.

Matching is greedy and **one-to-one**, so emitting a finding twice costs a
false positive rather than earning two true positives.

!!! warning "Read precision next to recall and the drop table"
    A pipeline that drops everything scores perfect precision. `--summary`
    carries the drop counts into the report for exactly this reason, and the
    report embeds the caveat in its own JSON.

## Committed artifacts

`docs/benchmarks/score-mock.json` is a full score report from the **mock**
backend — a deterministic rule extractor. It is committed so the format and the
pipeline are demonstrable, and it is labelled in its own `provenance` block:

```json
{"provenance": {"backend": "mock", "model_id": "mock-deterministic-v1",
  "note": "...This measures the harness, not an LLM..."}}
```

**A mock score is not a performance claim.** No live-backend artifact is
committed; see [decisions](decisions.md) D-016. To produce one:

```bash
fhirground synth --out bench/ --n-notes 200 --seed 0
fhirground extract --input bench/notes.jsonl --value-sets bench/valuesets \
  --backend bedrock --model "$MODEL_ID" --region us-east-1 \
  --samples 3 --temperature 0.4 --out bench_out/
fhirground score --pred bench_out/ndjson --gold bench/gold \
  --summary bench_out/run_summary.json --out docs/benchmarks/score-live.json
```

Record the model id, the date, and the output of `fhirground version` alongside
it.

## Synthea

fhirground does **not** require Synthea, and does not run it: Synthea is a Java
program and this project's environments have no JVM. Records are generated
internally instead.

The Synthea path is implemented and tested, for anyone who has Java:

```bash
./scripts/regenerate_synthea.sh            # pins Synthea 3.3.0
fhirground synth --synthea-dir tests/fixtures/synthea/fhir \
  --n-notes 50 --seed 0 --out bench/
```

`manifest.json` records `"source": "generated"` or `"synthea"` so no reader
mistakes one for the other. See [decisions](decisions.md) D-006.

## What this benchmark does not measure

It measures **recovery of structured records a synthetic note was rendered
from**. It does not measure performance on real clinical prose, which is
messier, more elliptical, and more varied than any template.

For agreement with human judgement on real notes, use
[clinval](clinval.md).
