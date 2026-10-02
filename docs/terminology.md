# Terminology and licensing

## fhirground bundles no terminology

SNOMED CT, LOINC, and RxNorm each carry licensing obligations. fhirground ships
**22 synthetic concepts** derived from the codes its own test fixtures use, and
nothing more.

!!! warning "You must supply your own value sets"
    The committed CSVs in `examples/data/valuesets/` exist so the tests and the
    demo can resolve codes at all. They are not terminology and must not be
    used as such.

    - **SNOMED CT** — requires an affiliate licence; free in member countries
      via your national release centre. See the SNOMED International website.
    - **LOINC** — free, but requires accepting the LOINC licence.
    - **RxNorm** — free from the NLM under its terms of service; the UMLS
      Metathesaurus requires a licence.

    fhirground does not check your licences. That is your responsibility.

## The only door a code comes through

Two mechanisms, and both are needed:

1. The [mention schema](mention-schema.md) has **no code fields**, so a model
   cannot produce one.
2. Every mapping function accepts a **`Resolution` object, never a string**:

```python
def build_condition(mention, resolution: Resolution, ...) -> dict: ...
```

A caller cannot pass a code that did not come from a resolver, because the
parameter type will not accept one. That is why §4.5's rule — "every coding
came from a `Resolution`" — is enforceable: no post-hoc inspection of a dict
could tell where a string came from.

A property test drives adversarial mentions through the pipeline, with `text`
containing `"SNOMED 44054006"`, `"ICD-10 E11.9"`, and
`'{"system": "http://loinc.org", "code": "1234-5"}'`, and asserts every emitted
`coding.code` exists in the active value sets.

## Resolvers

### LocalValueSet

The one most deployments use.

```python
from fhirground.terminology import LocalValueSet
conditions = LocalValueSet.from_csv("valuesets/conditions.csv")
```

```csv
system,code,display,synonyms
http://snomed.info/sct,44054006,Type 2 diabetes mellitus,type 2 diabetes|T2DM|DM2
```

Matching is two-stage: normalized exact match on display or any synonym
(scores 1.0 and 0.95), then rapidfuzz `token_set_ratio` above a threshold
(default 88). Token-set is right for clinical text because it ignores word
order — "diabetes mellitus type 2" and "type 2 diabetes mellitus" are the same
concept and a plain ratio would score them poorly.

**The `display` on the emitted coding comes from the value set, never from the
model's wording.** Otherwise a resource would carry a real code beside an
invented label. `code.text` preserves what the note said.

### NullResolver

Not a placeholder — the correct choice when no licensed terminology is
available. Every concept becomes text-only, the terminology confidence
component is **absent rather than zero**, and the output is honest about
carrying no codes.

### RxNavResolver

!!! danger "This sends note-derived text to a third party"
    It queries NLM RxNav with **medication text extracted from a clinical
    note**. That text is a note fragment. A site whose data-use agreement
    forbids sending note content outside its boundary **must not enable this**.

    It is opt-in with no default, is never part of a default `ResolverConfig`,
    and supports `offline=True` to answer only from cache. The cache contains
    the same note-derived text — see [PHI](phi.md).

```python
from fhirground.terminology import RxNavResolver
resolver = RxNavResolver(cache_path=".cache/rxnav.sqlite")
```

### Chaining

```python
from fhirground.terminology import Chain, ResolverConfig

resolvers = ResolverConfig(
    medication=Chain([
        LocalValueSet.from_csv("valuesets/meds.csv"),   # fast, offline, licensed
        RxNavResolver(cache_path=".cache/rxnav.sqlite"),  # the long tail
    ]),
)
```

Local first, network second. The other order would make every run depend on a
remote service and make results irreproducible.

`strategy="best"` queries every resolver and keeps the highest score, for
comparing resolvers. `min_score` treats a weak hit as a miss so the chain
continues.

## Per-type routing

`ResolverConfig` is per mention type rather than global, because the
terminologies differ: conditions to SNOMED CT or ICD-10, medications to
RxNorm, observations to LOINC. One resolver over all three would search every
value set for every mention — slower, and more likely to cross-match.

## When nothing resolves

The resource gets a **text-only `CodeableConcept`**:

```json
{"text": "fractured femur"}
```

Design principle 3. The confidence reflects it: the terminology component is
absent, so the weighted mean renormalizes over the components that *are*
available rather than scoring a configuration choice as a failure. See
[confidence](confidence.md).
