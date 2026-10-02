# The mention schema

This is the narrow waist of the whole design: the only thing the LLM is allowed
to say.

```bash
fhirground schema --print
```

## There are no code fields

Not "the prompt asks the model not to emit codes" — the schema has **nowhere to
put one**. A model that tries is rejected by its own decoding constraint:

```python
{"mention_type": "condition", "text": "type 2 diabetes",
 "evidence": "Type 2 diabetes", "assertion": "present",
 "snomed_code": "44054006"}
#  -> Additional properties are not allowed ('snomed_code' was unexpected)
```

`additionalProperties: false` at both levels makes this structural. A second
test asserts no *field name* in the model contains `code`, `coding`, `snomed`,
`loinc`, `rxnorm`, `icd`, or `system`, so a code field cannot be added by
accident later.

That is only half the guarantee. The other half is that every mapping function
takes a `Resolution` object rather than a string, so even a code reaching
Python cannot become a `coding`. See [terminology](terminology.md).

## The fields

Four required:

| Field | Values |
| --- | --- |
| `mention_type` | `condition`, `medication`, `observation` |
| `text` | the concept in plain language, ≤200 chars |
| `evidence` | a **verbatim** quote from the note, ≤300 chars |
| `assertion` | `present`, `negated`, `possible`, `historical`, `family`, `hypothetical` |

Thirteen optional, all nullable: `clinical_status_hint`, `onset_text`,
`med_action`, `dose_value`, `dose_unit`, `route`, `frequency_text`,
`obs_category`, `value_number`, `value_comparator`, `value_text`,
`value_unit`, `date_text`.

!!! note "Why so few are required"
    A model forced to fill every field invents values for the ones it cannot
    support. An invented dose is worse than a missing one. The prompt says so
    explicitly: *leave an optional field null rather than guessing*.

An empty optional string is treated as absent. Models routinely emit `""`
where they mean `null`, and carrying that through would produce a resource with
an empty `route.text`.

## The backend-safe subset

The schema is the constrained-decoding target for both backends, so it stays
inside what **both** accept: flat objects, enums, nullable fields via type
unions, no `oneOf`/`anyOf` on objects, no `$ref`.

Pydantic does not produce that shape — it emits `$defs` with `$ref` for enums
and `anyOf` for optionality. `fhirground.schemas.jsonschema` inlines the refs,
collapses `anyOf: [X, null]` into `type: [X, "null"]`, trims docstrings to
their first line, and then re-checks the result:

```python
from fhirground.schemas import assert_backend_safe, build_mention_schema
assert_backend_safe(build_mention_schema())   # raises, naming the JSON path
```

If a future pydantic release changes its output shape, that check fails here
rather than producing an opaque error at a backend.

## The checked-in file

`src/fhirground/schemas/mention.schema.json` is generated and committed,
because it is the artifact backends decode against. A test and a CI job assert
it still matches the model:

```bash
fhirground schema --check
```

Regenerate after changing the Pydantic model:

```bash
fhirground schema --out src/fhirground/schemas/mention.schema.json
```

The schema's short digest goes into every `Provenance`, so two runs that
decoded against different schemas are distinguishable in the output.

## One spec correction

SPEC §6.1 declares `"onset_text": {"enum": ["string", "null"]}`, which would
accept only the literal strings `"string"` and `"null"`. Every other nullable
string field uses `{"type": ["string", "null"]}`, so this is a typo.
fhirground generates the type union. See [decisions](decisions.md) D-004.
