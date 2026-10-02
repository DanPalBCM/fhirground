# Why two stages

## The failure mode

Ask an LLM for a FHIR `Condition` and you get something that looks like one.
Sometimes it is valid. Often `clinicalStatus` is a bare string where a
`CodeableConcept` belongs, `subject` is a patient name rather than a reference,
and — worst — `code.coding[0].code` is a plausible SNOMED identifier that
denotes something else entirely, or nothing at all.

That last failure is the dangerous one, because the output is *structurally
perfect*. A reviewer sees a well-formed resource with a real-looking code and
has no signal that anything is wrong. Invalid JSON fails loudly; a hallucinated
code fails silently and propagates into a data warehouse.

## The split

```
    the model's job                     code's job
    ───────────────                     ──────────
    read the note                       locate the evidence
    name the finding in plain words     apply the assertion policy
    quote the evidence verbatim         parse values, units, dates
    say how it is asserted              resolve codes via resolvers
                                        build and validate the resource
                                        score confidence, record provenance
```

The model is asked only for things a language model is actually good at:
reading prose and reporting what it says. Every operation with a right answer —
offsets, code lookup, unit conversion, FHIR field placement, invariant checking
— is done by code that can be tested.

## What this buys

**A hallucinated code becomes impossible rather than unlikely.** Not through
prompt discipline, which degrades, but structurally: the
[mention schema](mention-schema.md) has no code fields, so there is nowhere to
put one, and the mapping functions accept only `Resolution` objects, so a
string cannot become a `coding`. Both halves are needed — either alone leaves a
gap.

**Invalid FHIR becomes a code bug, not a model variance problem.** The mappers
are deterministic. If a `Condition` fails to parse, that is a defect with a
stack trace, not something to re-prompt around.

**Every claim is auditable.** The resource carries offsets into the note, so a
reviewer can read the sentence. A project rule asserts
`note_text[start:end] == quote`, so the offsets cannot rot.

**The model is swappable.** Changing backend, model, or prompt cannot change
FHIR structure, because structure is not downstream of the model. It changes
only *which findings* are reported — which is exactly what you want to measure
when comparing models.

## What it costs

**Two passes of engineering.** The mention schema has to be designed, the
mappers written, the invariants encoded. There is no way to skip this and keep
the guarantees.

**Recall depends on the model naming things the resolvers know.** A model that
writes "sugar problem" where the value set has "Type 2 diabetes mellitus" gets
a text-only `CodeableConcept`. That is the honest outcome — better than a
guessed code — but it is a miss, and it shows up in the terminology confidence
component.

**Codes are only as good as your value sets.** fhirground cannot resolve what
you have not licensed. The [terminology page](terminology.md) is explicit about
this.

## Where the stages meet

The whole interface is one JSON object. That is the narrow waist:

```json
{"mentions": [
  {"mention_type": "condition", "text": "type 2 diabetes",
   "evidence": "Type 2 diabetes, poorly controlled", "assertion": "present"}
]}
```

Four required fields. The model cannot say more than this, and does not need
to.
