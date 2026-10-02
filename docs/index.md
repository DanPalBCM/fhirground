# fhirground

**Grounded, validated FHIR R4 extraction from clinical notes.**

The design in one sentence: **the LLM produces mentions, deterministic code
produces FHIR.** Asking a model for FHIR directly yields invalid resources,
misplaced fields, and hallucinated codes. Splitting the problem means FHIR
structure never depends on the model's formatting.

## Four guarantees

**[Grounded](grounding.md).** Every emitted resource carries evidence offsets
that index the original note. A mention whose quote cannot be found is dropped
and counted.

**[No invented codes](mention-schema.md).** The mention schema has no code
fields, and every mapping function accepts a `Resolution` rather than a string.
A code cannot reach a resource without having come from a
[terminology resolver](terminology.md). A property test drives adversarial
input through the pipeline to check it.

**Validated.** Every resource parses under the pinned FHIR R4B models and
passes the project rules, or it is dropped with a reason. The R4 invariants
that bind are encoded with their text quoted verbatim from the
StructureDefinitions.

**[Assertion-aware](assertion-mapping.md).** Negated, family-history, and
hypothetical mentions are never emitted as affirmed facts.

## Start here

- [Getting started](getting-started.md) — install, run the demo, use your notes
- [Why two stages](design.md) — the reasoning, and what it buys
- [Assertion and mapping rules](assertion-mapping.md) — the normative table
- [PHI and security](phi.md) — logging, the caches, `include_quotes`

## What it will not do

fhirground guarantees structure, grounding, and provenance. It makes **no
accuracy guarantee** and **no profile conformance claim**, it is not a
terminology server, and it is not a clinical decision support tool. The
[limitations](getting-started.md#limitations) are deliberately blunt.
