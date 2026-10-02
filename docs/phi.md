# PHI and security considerations

fhirground reads clinical notes. This page is about where that text can end up.

## Logs never contain note text

No module in fhirground logs note text. Log lines carry ids, counts, timings,
and hashes only.

`NoteInput.text_hash()` exists so a log line can identify a note without
quoting it:

```python
note.text_hash()   # 'a3f2c9e1b8d47506'
```

The one place text deliberately travels is `RawResponse.text`, because the
repair loop needs the model's own output to feed back. It is not logged.

## The LLM call cache contains note text

```bash
fhirground extract --cache .cache/llm.sqlite ...
```

The cached **value** includes the prompt, and the prompt includes the note
chunk. So the SQLite file is note-derived data and is subject to the same
handling rules as the notes themselves.

- It is **off by default**.
- `.gitignore` blocks `*.sqlite`, `.cache/`, and `llm_cache*`, and a CI job
  fails if one is ever committed.
- Treat it as you would the note corpus: same storage, same retention, same
  disposal.

## The RxNav resolver sends text off-site

!!! danger "Opt-in only, and deliberately so"
    `RxNavResolver` queries a public NLM service with **medication text
    extracted from a clinical note**. That text is a note fragment leaving your
    boundary.

    A site whose data-use agreement forbids sending note content to a third
    party must not enable it. It is never part of a default `ResolverConfig`
    and must be constructed explicitly.

    Its cache holds the same text. `offline=True` answers only from cache and
    never calls out.

## Quote minimization

```bash
fhirground extract --no-quotes ...
```

Omits `quote` from every `evidence` extension while **keeping the offsets**. A
reviewer with access to the source note can still verify the claim; the
resource itself carries no note text.

The project validation rule adapts: with quotes present it checks
`note_text[start:end] == quote`; with them absent it still requires the offsets
to be in range. The claim stays checkable either way.

Use this when Bundles leave the environment the notes live in.

## What the outputs contain

| File | Note text? |
| --- | --- |
| `bundles/*.json` | **yes** — evidence quotes, unless `--no-quotes` |
| `ndjson/*.ndjson` | **yes** — same resources |
| `clinval_predictions.jsonl` | **yes** — `span.text` is the evidence quote |
| `debug.jsonl` | **yes** — the raw mention, including its `evidence` field |
| `run_summary.json` | no — counts, timings, versions only |
| `score.json` | no — counts and metrics only |

`debug.jsonl` is the most text-bearing output, since it carries every mention
including dropped ones. It is also the most useful for diagnosis. Keep it
inside the boundary.

## Backends send notes to a model

Obvious but worth stating. With `--backend bedrock` or
`--backend openai_compatible`, note text goes to that service.

- **Bedrock**: subject to your AWS agreement and the region you choose.
  fhirground never holds credentials; they come from the standard boto3 chain.
- **vLLM**: `--base-url` is wherever you point it. A server you run inside your
  own boundary keeps the text there, which is the main reason to prefer it.
- **mock**: no network at all. Useful for a dry run that exercises the whole
  pipeline on real notes without any egress.

## Synthetic data only in this repository

Every dataset committed here is generated. `fhirground.synth` builds notes from
fixed concept banks with sequential identifiers and dates derived from a
constant epoch. Tests assert that identifiers match a placeholder pattern and
that no note text contains a long digit run or an `MRN` token.

A CI job fails if any data file appears outside `examples/data/`,
`tests/fixtures/`, or `docs/benchmarks/`.

## What fhirground does not do

- **No de-identification.** It is not a scrubber. Notes in are notes out.
- **No access control, no encryption at rest, no audit logging.** Those belong
  to the system you run it in.
- **No PHI detection.** It will not warn you that a note contains an
  identifier.
